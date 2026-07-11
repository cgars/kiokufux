from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .autotagging import EmbeddingAutoTagger, normalize_candidate_tags, propose_tags
from .catalog import Catalog
from .config import KiokuFuxConfig, catalog_path, ensure_workspace, load_config, write_default_config
from .embeddings import backend_from_options, generate_embeddings
from .models import Photo, SearchResult, TagProposal, TagProposalSummary, TagVocabularyEntry
from .scanner import scan as scan_folder
from .search import search as run_search
from .sidecar import export_sidecars
from .thumbnails import generate_thumbnails
from .vlm import backend_from_name, parse_image_analysis

LOGGER_NAME = "kiokufux"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
PRIVACY_LOCAL_NOTICE = "Online services: no photo, metadata, or query data will be sent; processing is local."
OPENCLIP_DOWNLOAD_NOTICE = (
    "Online services: no photo, metadata, or query data will be sent; "
    "OpenCLIP may contact the network only to download model weights if they are not cached."
)


def _catalog(root: Path) -> tuple[Path, Catalog]:
    ws = ensure_workspace(root)
    cat = Catalog(catalog_path(root))
    cat.init_schema()
    return ws, cat


def _setup_logging(ws: Path, verbose: int = 0) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    file_handler = logging.FileHandler(ws / "logs" / "kiokufux.log")
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    if verbose:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG if verbose > 1 else logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(console_handler)
    return logger


def _add_embedding_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-backend", choices=["auto", "openclip", "simple"], default=None)
    parser.add_argument("--openclip-model", help="OpenCLIP model architecture, e.g. ViT-B-32")
    parser.add_argument("--openclip-pretrained", help="OpenCLIP pretrained weights tag, e.g. laion2b_s34b_b79k")


def _embedding_backend(args: argparse.Namespace, config: KiokuFuxConfig):
    backend = args.embedding_backend or config.embeddings.backend
    model = args.openclip_model or config.embeddings.openclip_model
    pretrained = args.openclip_pretrained or config.embeddings.openclip_pretrained
    return backend_from_options(backend, model, pretrained)


def _privacy_notice(args: argparse.Namespace, config: KiokuFuxConfig | None = None) -> str:
    if getattr(args, "cmd", None) == "vlm-analyze":
        return PRIVACY_LOCAL_NOTICE
    if getattr(args, "cmd", None) not in {"embed", "search", "auto-tag"}:
        return PRIVACY_LOCAL_NOTICE
    configured_backend = config.embeddings.backend if config is not None else None
    backend = getattr(args, "embedding_backend", None) or configured_backend
    if backend in {"auto", "openclip"}:
        return OPENCLIP_DOWNLOAD_NOTICE
    return PRIVACY_LOCAL_NOTICE


def _format_search_result(index: int, result: SearchResult, summary: bool = False) -> str:
    stats = (
        f"{index}\traw_score={result.score:.4f}\tmatch={result.match_label}"
        f"\ttop_percent={result.top_percent:.1f}\trobust_z={result.robust_z_score:.2f}"
        f"\tnormalized_relative={result.normalized_score:.4f}"
    )
    if summary:
        return f"{stats}\tname={result.source_path.name}"
    return (
        f"{stats}\tpath={result.source_path}\tthumb={result.thumbnail_path}"
        f"\tmetadata={result.metadata_summary}"
    )


def _proposal_photo_label(photo_id: str, photos: dict[str, Photo]) -> str:
    photo = photos.get(photo_id)
    filename = Path(photo.relative_path).name if photo is not None else "(unknown)"
    return f"{photo_id[:7]}\t{filename}"


def _print_tag_proposal_summary(rows: list[TagProposalSummary]) -> None:
    for row in rows:
        print(
            f"{row.tag}\tphotos={row.photo_count}\tproposals={row.proposal_count}"
            f"\tavg_confidence={row.avg_confidence:.2f}\tmax_confidence={row.max_confidence:.2f}"
            f"\tstatus={row.status}\tsource={row.source}"
        )


def _print_vocabulary(rows: list[TagVocabularyEntry]) -> None:
    for row in rows:
        aliases = ",".join(row.aliases) if row.aliases else "-"
        parent = row.parent or "-"
        notes = row.notes or ""
        print(
            f"{row.tag}\tcategory={row.category}\tscope={row.scope}\tstatus={row.status}"
            f"\tparent={parent}\taliases={aliases}\tnotes={notes}"
        )


def _print_tag_proposals(rows: list[TagProposal], photos: dict[str, Photo]) -> None:
    current_photo_id: str | None = None
    for row in rows:
        if row.photo_id != current_photo_id:
            current_photo_id = row.photo_id
            print(f"{_proposal_photo_label(row.photo_id, photos)}:")
        print(f"  - {row.tag}\tconfidence={row.confidence:.2f}\tstatus={row.status}\tsource={row.source}")


def _extract_verbose_args(argv: list[str]) -> tuple[list[str], int]:
    cleaned: list[str] = []
    verbose = 0
    for arg in argv:
        if arg == "--verbose":
            verbose += 1
        elif arg == "-v":
            verbose += 1
        elif arg.startswith("-v") and set(arg[1:]) == {"v"}:
            verbose += len(arg) - 1
        else:
            cleaned.append(arg)
    return cleaned, verbose


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kiokufux")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Print verbose log messages to stderr; use -vv for debug logs")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ["init", "scan", "thumbnails", "export-sidecars"]:
        sp = sub.add_parser(name)
        sp.add_argument("path", type=Path)
    tag = sub.add_parser("tag")
    tag.add_argument("path", type=Path)
    tag.add_argument("photo_id")
    tag.add_argument("tags", nargs="+")
    untag = sub.add_parser("untag")
    untag.add_argument("path", type=Path)
    untag.add_argument("photo_id")
    untag.add_argument("tags", nargs="+")
    tags = sub.add_parser("tags")
    tags.add_argument("path", type=Path)
    tags.add_argument("photo_id", nargs="?")
    auto_tag = sub.add_parser("auto-tag")
    auto_tag.add_argument("path", type=Path)
    auto_tag.add_argument("--candidate-tags", help="Comma-separated candidate tags for zero-shot AI tagging")
    auto_tag.add_argument("--top-k", type=int, help="Maximum AI tag proposals per photo")
    auto_tag.add_argument("--min-score", type=float, help="Minimum image/text similarity for AI tag proposals")
    _add_embedding_options(auto_tag)
    summary = sub.add_parser("tag-summary")
    summary.add_argument("path", type=Path)
    summary.add_argument("--status", default="pending", choices=["pending", "accepted", "rejected", "all"])
    vocab = sub.add_parser("vocab")
    vocab.add_argument("path", type=Path)
    vocab.add_argument("--status", default="all", choices=["proposed", "accepted", "rejected", "all"])
    vocab_propose = sub.add_parser("vocab-propose")
    vocab_propose.add_argument("path", type=Path)
    vocab_propose.add_argument("--min-photos", type=int, default=1)
    vocab_accept = sub.add_parser("vocab-accept")
    vocab_accept.add_argument("path", type=Path)
    vocab_accept.add_argument("tag")
    vocab_accept.add_argument("--category", default="uncategorized")
    vocab_accept.add_argument("--scope", default="optional", choices=["core", "collection-specific", "optional"])
    vocab_accept.add_argument("--parent")
    vocab_accept.add_argument("--alias", action="append", default=[])
    vocab_accept.add_argument("--notes")
    vocab_reject = sub.add_parser("vocab-reject")
    vocab_reject.add_argument("path", type=Path)
    vocab_reject.add_argument("tag")
    vocab_reject.add_argument("--notes")
    vocab_merge = sub.add_parser("vocab-merge")
    vocab_merge.add_argument("path", type=Path)
    vocab_merge.add_argument("alias")
    vocab_merge.add_argument("canonical")
    vocab_apply = sub.add_parser("vocab-apply")
    vocab_apply.add_argument("path", type=Path)
    vlm = sub.add_parser("vlm-analyze")
    vlm.add_argument("path", type=Path)
    vlm.add_argument("--vlm-backend", default="fake", choices=["fake", "ollama"])
    vlm.add_argument("--ollama-url", default="http://localhost:11434", help="Base URL for local or LAN Ollama server")
    vlm.add_argument("--ollama-model", default="llava", help="Ollama vision model name")
    vlm.add_argument("--vlm-timeout", type=float, default=120.0, help="VLM request timeout in seconds")
    vlm.add_argument("--limit", type=int)
    vlm.add_argument("--force", action="store_true", help="Re-analyze photos that already have VLM analysis")
    proposals = sub.add_parser("tag-proposals", aliases=["tag-review"])
    proposals.add_argument("path", type=Path)
    proposals.add_argument("photo_id", nargs="?")
    proposals.add_argument("--status", default="pending", choices=["pending", "accepted", "rejected", "all"])
    accept = sub.add_parser("accept-tag")
    accept.add_argument("path", type=Path)
    accept.add_argument("photo_id", nargs="?")
    accept.add_argument("tag", nargs="?")
    accept.add_argument("--all", action="store_true", help="Accept all pending tag proposals, optionally limited to PHOTO_ID")
    reject = sub.add_parser("reject-tag")
    reject.add_argument("path", type=Path)
    reject.add_argument("photo_id")
    reject.add_argument("tag")
    e = sub.add_parser("embed")
    e.add_argument("path", type=Path)
    _add_embedding_options(e)
    s = sub.add_parser("search")
    s.add_argument("path", type=Path)
    s.add_argument("query")
    s.add_argument("--top-k", type=int)
    s.add_argument("--summary", action="store_true", help="Only show search statistics and image file name")
    s.add_argument("--min-raw-score", type=float, help="Minimum raw similarity required for confident matches")
    s.add_argument("--min-robust-z", type=float, help="Minimum robust z-score required for confident matches")
    _add_embedding_options(s)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    cleaned_argv, extracted_verbose = _extract_verbose_args(raw_argv)
    parser = _build_parser()
    args = parser.parse_args(cleaned_argv)
    args.verbose = max(args.verbose, extracted_verbose)
    root = args.path.expanduser().resolve()
    ws = ensure_workspace(root)
    config = load_config(root)
    args.verbose = max(args.verbose, config.logging.verbose)
    logger = _setup_logging(ws, args.verbose)
    logger.debug("Parsed arguments: %s", args)
    print(_privacy_notice(args, config))

    if args.cmd == "init":
        Catalog(catalog_path(root)).init_schema()
        config_file = write_default_config(root)
        logger.info("Initialized KiokuFux workspace at %s", ws)
        logger.info("Wrote KiokuFux config at %s", config_file)
        print(f"Initialized KiokuFux workspace at {ws}")
        print(f"Configuration file: {config_file}")
        return 0

    cat = Catalog(catalog_path(root))
    cat.init_schema()
    logger.info("Running command %s for %s", args.cmd, root)
    with cat:
        if args.cmd == "scan":
            indexed, errors = scan_folder(root, cat, logger)
            logger.info("Scan complete: %s indexed/updated, %s errors", indexed, errors)
            print(f"Scan complete: {indexed} indexed/updated, {errors} errors")
        elif args.cmd == "thumbnails":
            generated = generate_thumbnails(cat, ws, max_size=config.thumbnails.max_size)
            logger.info("Generated %s thumbnails", generated)
            print(f"Generated {generated} thumbnails")
        elif args.cmd == "embed":
            generated = generate_embeddings(cat, ws, _embedding_backend(args, config))
            logger.info("Generated %s embeddings", generated)
            print(f"Generated {generated} embeddings")
        elif args.cmd == "search":
            results = run_search(
                cat,
                args.query,
                (args.top_k if args.top_k is not None else config.search.top_k),
                _embedding_backend(args, config),
                min_raw_score=(args.min_raw_score if args.min_raw_score is not None else config.search.min_raw_score),
                min_robust_z=(args.min_robust_z if args.min_robust_z is not None else config.search.min_robust_z),
            )
            logger.info("Search returned %s results for query %r", len(results), args.query)
            if results and not results[0].confidence_gate_passed:
                print("No confident matches found.")
                print("Showing closest available results.")
            for i, r in enumerate(results, 1):
                print(_format_search_result(i, r, summary=args.summary))
        elif args.cmd == "export-sidecars":
            exported = export_sidecars(cat)
            logger.info("Exported %s sidecars", exported)
            print(f"Exported {exported} sidecars")
        elif args.cmd == "tag":
            for tag in args.tags:
                cat.add_tag(args.photo_id, tag)
            logger.info("Added %s tags to %s", len(args.tags), args.photo_id)
            print(f"Tagged {args.photo_id}: {', '.join(t.tag for t in cat.list_tags(args.photo_id))}")
        elif args.cmd == "untag":
            for tag in args.tags:
                cat.remove_tag(args.photo_id, tag)
            logger.info("Removed %s tags from %s", len(args.tags), args.photo_id)
            remaining = cat.list_tags(args.photo_id)
            print(f"Tags for {args.photo_id}: {', '.join(t.tag for t in remaining) if remaining else '(none)'}")
        elif args.cmd == "tags":
            if args.photo_id:
                rows = cat.list_tags(args.photo_id)
                for row in rows:
                    print(f"{row.photo_id}\t{row.tag}\t{row.source}")
            else:
                rows = cat.list_all_tags()
                for row in rows:
                    print(f"{row.photo_id}\t{row.tag}\t{row.source}")
        elif args.cmd == "auto-tag":
            candidates = normalize_candidate_tags((args.candidate_tags or config.autotagging.candidate_tags).split(","))
            tagger = EmbeddingAutoTagger(
                backend=_embedding_backend(args, config),
                candidate_tags=candidates,
                top_k=(args.top_k if args.top_k is not None else config.autotagging.top_k),
                min_score=(args.min_score if args.min_score is not None else config.autotagging.min_score),
            )
            proposed = propose_tags(cat, tagger)
            logger.info("Generated %s tag proposals", proposed)
            print(f"Generated {proposed} tag proposals for review")
        elif args.cmd == "tag-summary":
            status = None if args.status == "all" else args.status
            rows = cat.summarize_tag_proposals(status=status)
            _print_tag_proposal_summary(rows)
        elif args.cmd == "vocab":
            status = None if args.status == "all" else args.status
            _print_vocabulary(cat.list_vocabulary(status=status))
        elif args.cmd == "vocab-propose":
            created = cat.propose_vocabulary_from_tag_proposals(min_photos=args.min_photos)
            logger.info("Created %s vocabulary proposals", created)
            print(f"Created {created} vocabulary proposals")
        elif args.cmd == "vocab-accept":
            cat.upsert_vocabulary_tag(args.tag, category=args.category, scope=args.scope, status="accepted", parent=args.parent, aliases=args.alias, notes=args.notes)
            logger.info("Accepted vocabulary tag %s", args.tag)
            print(f"Accepted vocabulary tag: {args.tag}")
        elif args.cmd == "vocab-reject":
            cat.upsert_vocabulary_tag(args.tag, status="rejected", notes=args.notes)
            logger.info("Rejected vocabulary tag %s", args.tag)
            print(f"Rejected vocabulary tag: {args.tag}")
        elif args.cmd == "vocab-merge":
            cat.merge_vocabulary_tag(args.alias, args.canonical)
            logger.info("Merged vocabulary alias %s into %s", args.alias, args.canonical)
            print(f"Merged vocabulary alias: {args.alias} -> {args.canonical}")
        elif args.cmd == "vocab-apply":
            changed = cat.apply_vocabulary_to_tag_proposals()
            logger.info("Applied vocabulary to %s tag proposals", changed)
            print(f"Applied vocabulary to {changed} tag proposals")
        elif args.cmd == "vlm-analyze":
            backend = backend_from_name(
                args.vlm_backend,
                ollama_url=args.ollama_url,
                ollama_model=args.ollama_model,
                timeout=args.vlm_timeout,
            )
            accepted_vocabulary = [entry.tag for entry in cat.list_vocabulary(status="accepted")]
            photos = cat.list_photos()
            logger.info(
                "VLM analyze starting: backend=%s model=%s photos=%s limit=%s force=%s accepted_vocabulary=%s",
                backend.source, backend.model_version, len(photos), args.limit, args.force, len(accepted_vocabulary),
            )
            if hasattr(backend, "endpoint_url"):
                logger.info("VLM Ollama endpoint: %s", backend.endpoint_url())
            analyzed = 0
            for photo in photos:
                if args.limit is not None and analyzed >= args.limit:
                    break
                if not args.force and cat.get_image_analysis(photo.photo_id) is not None:
                    logger.debug("Skipping existing VLM analysis for %s (%s)", photo.relative_path, photo.photo_id[:7])
                    continue
                logger.info("Analyzing %s (%s) with %s", photo.relative_path, photo.photo_id[:7], backend.source)
                try:
                    raw = backend.analyze_image(photo.source_path, accepted_vocabulary=accepted_vocabulary)
                    analysis = parse_image_analysis(
                        photo.photo_id, raw, source=backend.source,
                        model_name=backend.model_name, model_version=backend.model_version,
                    )
                    cat.upsert_image_analysis(analysis)
                except Exception as exc:
                    logger.error("VLM analysis failed for %s (%s): %s", photo.relative_path, photo.photo_id[:7], exc)
                    print(f"VLM analysis failed for {photo.relative_path}: {exc}", file=sys.stderr)
                    return 1
                analyzed += 1
                logger.info(
                    "Stored VLM analysis for %s: candidate_tags=%s uncertain_tags=%s",
                    photo.relative_path, len(analysis.candidate_tags), len(analysis.uncertain_tags),
                )
            logger.info("Generated %s VLM image analyses", analyzed)
            print(f"Generated {analyzed} VLM image analyses")
        elif args.cmd in {"tag-proposals", "tag-review"}:
            status = None if args.status == "all" else args.status
            rows = cat.list_tag_proposals(args.photo_id, status=status)
            photos = {photo.photo_id: photo for photo in cat.list_photos(include_missing=True)}
            _print_tag_proposals(rows, photos)
        elif args.cmd == "accept-tag":
            if args.photo_id:
                try:
                    args.photo_id = cat.resolve_photo_id(args.photo_id)
                except ValueError as exc:
                    parser.error(str(exc))
            if args.all:
                accepted = cat.accept_tag_proposals(args.photo_id)
                scope = args.photo_id[:7] if args.photo_id else "all images"
                logger.info("Accepted %s pending tag proposals for %s", accepted, scope)
                print(f"Accepted {accepted} pending tag proposals for {scope}")
            else:
                if not args.photo_id or not args.tag:
                    parser.error("accept-tag requires PHOTO_ID and TAG unless --all is used")
                cat.accept_tag_proposal(args.photo_id, args.tag)
                logger.info("Accepted tag proposal %s for %s", args.tag, args.photo_id)
                print(f"Accepted tag for {args.photo_id}: {args.tag}")
        elif args.cmd == "reject-tag":
            cat.reject_tag_proposal(args.photo_id, args.tag)
            logger.info("Rejected tag proposal %s for %s", args.tag, args.photo_id)
            print(f"Rejected tag for {args.photo_id}: {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
