from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from .autotagging import EmbeddingAutoTagger, normalize_candidate_tags, propose_tags
from .catalog import Catalog
from .config import KiokuFuxConfig, catalog_path, ensure_workspace, load_config, write_default_config
from .embeddings import backend_from_options, generate_embeddings
from .models import Photo, SearchResult, TagProposal, TagProposalSummary, TagVocabularyEntry
from .hashing import file_sha256
from .metadata import extract_metadata
from .rotation import VALID_ROTATION_DEGREES, RotationDetection, detect_clockwise_rotation, detect_clockwise_rotation_from_vlm_response, rotate_image
from .scanner import scan as scan_folder
from .search import search as run_search
from .sidecar import export_sidecars
from .gallery import export_gallery
from .thumbnails import generate_thumbnails
from .prompts import ROTATION_VLM_COMPARE_PROMPT, ROTATION_VLM_PROMPT, iter_prompts
from .vlm import backend_from_name, parse_image_analysis

LOGGER_NAME = "kiokufux"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
PRIVACY_LOCAL_NOTICE = "Online services: no photo, metadata, or query data will be sent; processing is local."
VLM_REMOTE_NOTICE = (
    "Online services: photos will be sent to the configured Ollama endpoint for VLM analysis; "
    "metadata and query data remain local."
)
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
    if getattr(args, "cmd", None) == "vlm-analyze" or (getattr(args, "cmd", None) == "rotate" and (getattr(args, "vlm_fallback", False) or getattr(args, "vlm_only", False))):
        backend = getattr(args, "vlm_backend", None)
        url = str(getattr(args, "ollama_url", "") or "")
        if backend == "ollama" and not (url.startswith("http://localhost") or url.startswith("http://127.0.0.1") or url.startswith("http://[::1]")):
            return VLM_REMOTE_NOTICE
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



def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    if not rows:
        print("(none)")
        return
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def _shorten(text: str | None, max_len: int = 88) -> str:
    if not text:
        return "-"
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= max_len else cleaned[: max_len - 1] + "…"

def _proposal_photo_label(photo_id: str, photos: dict[str, Photo]) -> str:
    photo = photos.get(photo_id)
    filename = Path(photo.relative_path).name if photo is not None else "(unknown)"
    return f"{photo_id[:7]}\t{filename}"


def _print_tag_proposal_summary(rows: list[TagProposalSummary]) -> None:
    _print_table(
        ["tag", "photos", "proposals", "avg", "max", "status", "source"],
        [[row.tag, str(row.photo_count), str(row.proposal_count), f"{row.avg_confidence:.2f}", f"{row.max_confidence:.2f}", row.status, row.source] for row in rows],
    )


def _print_vocabulary(rows: list[TagVocabularyEntry]) -> None:
    _print_table(
        ["tag", "category", "scope", "status", "parent", "aliases", "notes"],
        [[row.tag, row.category, row.scope, row.status, row.parent or "-", ", ".join(row.aliases) if row.aliases else "-", row.notes or "-"] for row in rows],
    )


def _print_tag_proposals(rows: list[TagProposal], photos: dict[str, Photo]) -> None:
    table_rows = []
    for row in rows:
        photo = photos.get(row.photo_id)
        filename = Path(photo.relative_path).name if photo is not None else "(unknown)"
        table_rows.append([row.photo_id[:7], filename, row.tag, f"{row.confidence:.2f}", row.status, row.source])
    _print_table(["photo", "file", "tag", "conf", "status", "source"], table_rows)


def _print_descriptions(rows: list[tuple[Photo, object]]) -> None:
    table_rows = []
    for photo, analysis in rows:
        table_rows.append([photo.photo_id[:7], Path(photo.relative_path).name, analysis.source, _shorten(analysis.caption, 48), _shorten(analysis.description)])
    _print_table(["photo", "file", "source", "caption", "description"], table_rows)


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
    gallery = sub.add_parser("export-gallery")
    gallery.add_argument("path", type=Path)
    gallery.add_argument("output", type=Path)
    gallery.add_argument("--title", default="KiokuFux Gallery")
    gallery.add_argument("--query")
    gallery.add_argument("--tag", action="append", default=[])
    gallery.add_argument("--top-k", type=int)
    gallery.add_argument("--min-tag-count", type=int, default=2)
    gallery.add_argument("--max-cloud-tags", type=int, default=40)
    gallery.add_argument("--image-max-size", type=int)
    gallery.add_argument("--overwrite", action="store_true")
    _add_embedding_options(gallery)
    rotate = sub.add_parser("rotate")
    rotate.add_argument("path", type=Path)
    rotate.add_argument("photo_id", nargs="?", help="Full photo ID or unique prefix of at least 7 characters; omit with --auto to process all indexed images")
    rotation_mode = rotate.add_mutually_exclusive_group(required=True)
    rotation_mode.add_argument("--degrees", type=int, choices=sorted(VALID_ROTATION_DEGREES), help="Clockwise rotation in degrees")
    rotation_mode.add_argument("--auto", action="store_true", help="Detect the clockwise rotation from EXIF or conservative image-content heuristics")
    rotate.add_argument("--no-backup", action="store_true", help="Do not write a same-folder .bak copy before rotating")
    rotate.add_argument("--vlm-fallback", action="store_true", help="If EXIF, stored VLM text, and local heuristics are uncertain, run a VLM analysis for rotation")
    rotate.add_argument("--vlm-only", action="store_true", help="With --auto, skip EXIF/stored text/local heuristics and use only a fresh VLM orientation check")
    rotate.add_argument("--vlm-verify", action="store_true", help="After a fresh VLM-driven rotation, ask the VLM once more whether the result looks upright; never applies a second rotation")
    rotate.add_argument("--vlm-compare", action="store_true", help="For fresh VLM checks, ask the VLM to choose among 0/90/180/270-degree candidate previews instead of judging one image")
    rotate.add_argument("--vlm-backend", default="fake", choices=["fake", "ollama"])
    rotate.add_argument("--ollama-url", default="http://localhost:11434", help="Base URL for local or LAN Ollama server used by rotation VLM options")
    rotate.add_argument("--ollama-model", default="llava", help="Ollama vision model name used by rotation VLM options")
    rotate.add_argument("--vlm-timeout", type=float, default=120.0, help="VLM request timeout in seconds for rotation VLM options")
    prompts = sub.add_parser("prompts", help="Print VLM prompts used by KiokuFux")
    prompts.add_argument("--topic", choices=["all", "rotation", "vlm-analysis"], default="all")
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
    vocab_apply.add_argument("--source", help="Only apply vocabulary to proposals from this source (default: all sources)")
    vlm = sub.add_parser("vlm-analyze")
    vlm.add_argument("path", type=Path)
    vlm.add_argument("--vlm-backend", default="fake", choices=["fake", "ollama"])
    vlm.add_argument("--ollama-url", default="http://localhost:11434", help="Base URL for local or LAN Ollama server")
    vlm.add_argument("--ollama-model", default="llava", help="Ollama vision model name")
    vlm.add_argument("--vlm-timeout", type=float, default=120.0, help="VLM request timeout in seconds")
    vlm.add_argument("--limit", type=int)
    vlm.add_argument("--force", action="store_true", help="Re-analyze photos that already have VLM analysis")
    descriptions = sub.add_parser("descriptions", aliases=["vlm-descriptions"])
    descriptions.add_argument("path", type=Path)
    descriptions.add_argument("photo_id", nargs="?")
    proposals = sub.add_parser("tag-proposals", aliases=["tag-review"])
    proposals.add_argument("path", type=Path)
    proposals.add_argument("photo_id", nargs="?")
    proposals.add_argument("--status", default="pending", choices=["pending", "accepted", "rejected", "all"])
    accept = sub.add_parser("accept-tag")
    accept.add_argument("path", type=Path)
    accept.add_argument("photo_id", nargs="?")
    accept.add_argument("tag", nargs="?")
    accept.add_argument("--all", action="store_true", help="Accept all pending tag proposals, optionally limited to PHOTO_ID")
    accept.add_argument("--source", help="Proposal source to accept (default: any source)")
    reject = sub.add_parser("reject-tag")
    reject.add_argument("path", type=Path)
    reject.add_argument("photo_id")
    reject.add_argument("tag")
    reject.add_argument("--source", help="Proposal source to reject (default: ai-zero-shot)")
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


def _analysis_description_text(analysis: object | None) -> str | None:
    if analysis is None:
        return None
    parts = [
        getattr(analysis, "caption", None),
        getattr(analysis, "description", None),
        getattr(analysis, "scene", None),
        getattr(analysis, "activity", None),
        *getattr(analysis, "warnings", []),
    ]
    return " ".join(part for part in parts if part) or None


def _rotate_photo(cat: Catalog, photo: Photo, degrees: int, create_backup: bool, logger: logging.Logger) -> str | None:
    result = rotate_image(photo.source_path, degrees, create_backup=create_backup)
    metadata = extract_metadata(photo.source_path)
    cat.mark_photo_edited(photo.photo_id, file_sha256(photo.source_path), metadata)
    logger.info("Rotated %s clockwise by %s degrees", photo.source_path, degrees)
    return str(result.backup_path) if result.backup_path else None


def _rotation_basis_label(source: str) -> str:
    labels = {
        "exif": "EXIF orientation metadata",
        "stored-vlm-description": "existing VLM description",
        "textline-heuristic": "local text-line heuristic",
        "fresh-vlm-only": "fresh VLM check only (--vlm-only)",
        "fresh-vlm-fallback": "fresh VLM fallback after cheaper checks were uncertain",
        "fresh-vlm-verification": "one-shot fresh VLM verification after rotation",
    }
    return labels.get(source, source)


def _format_rotation_decision(prefix: str, detection: RotationDetection) -> str:
    basis = _rotation_basis_label(detection.source)
    if detection.degrees is None:
        return f"{prefix}: no rotation applied; checked={basis}; confidence={detection.confidence:.2f}; why={detection.reason}"
    return f"{prefix}: will rotate {detection.degrees}° clockwise; checked={basis}; confidence={detection.confidence:.2f}; why={detection.reason}"


def _rotation_vlm_backend(args: argparse.Namespace):
    return backend_from_name(
        args.vlm_backend,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        timeout=args.vlm_timeout,
        prompt=ROTATION_VLM_COMPARE_PROMPT if args.vlm_compare else ROTATION_VLM_PROMPT,
    )


def _build_rotation_candidate_sheet(photo: Photo, target: Path) -> None:
    from PIL import Image, ImageDraw, ImageOps

    with Image.open(photo.source_path) as img:
        base = ImageOps.exif_transpose(img).convert("RGB")
    candidates = [("A", 0), ("B", 90), ("C", 180), ("D", 270)]
    thumbs = []
    for label, degrees in candidates:
        thumb = base.rotate(-degrees, expand=True)
        thumb.thumbnail((360, 360), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (380, 410), "white")
        x = (380 - thumb.width) // 2
        canvas.paste(thumb, (x, 40))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), f"{label}: {degrees}° clockwise", fill="black")
        thumbs.append(canvas)
    sheet = Image.new("RGB", (760, 820), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % 2) * 380, (index // 2) * 410))
    sheet.save(target, "JPEG", quality=90)


def _run_rotation_vlm_analysis(photo: Photo, args: argparse.Namespace, logger: logging.Logger):
    backend = _rotation_vlm_backend(args)
    if args.vlm_compare:
        logger.info("Running VLM rotation candidate comparison for %s with %s", photo.source_path, backend.source)
        with tempfile.TemporaryDirectory(prefix="kiokufux-rotation-") as tmpdir:
            sheet_path = Path(tmpdir) / f"{photo.source_path.stem}-rotation-candidates.jpg"
            _build_rotation_candidate_sheet(photo, sheet_path)
            return backend.analyze_image(sheet_path)
    logger.info("Running direct VLM rotation check for %s with %s", photo.source_path, backend.source)
    return backend.analyze_image(photo.source_path)


def _detect_rotation_for_photo(cat: Catalog, photo: Photo, args: argparse.Namespace, logger: logging.Logger):
    if args.vlm_only:
        raw = _run_rotation_vlm_analysis(photo, args, logger)
        return detect_clockwise_rotation_from_vlm_response(raw, source="fresh-vlm-only")

    detection = detect_clockwise_rotation(
        photo.source_path,
        description_text=_analysis_description_text(cat.get_image_analysis(photo.photo_id)),
    )
    if detection.degrees is not None or not args.vlm_fallback:
        return detection

    raw = _run_rotation_vlm_analysis(photo, args, logger)
    vlm_detection = detect_clockwise_rotation_from_vlm_response(raw, source="fresh-vlm-fallback")
    if vlm_detection.degrees is not None:
        return vlm_detection
    return detection


def _verify_vlm_rotation_once(photo: Photo, args: argparse.Namespace, logger: logging.Logger) -> RotationDetection:
    raw = _run_rotation_vlm_analysis(photo, args, logger)
    return detect_clockwise_rotation_from_vlm_response(raw, source="fresh-vlm-verification")


def _print_prompts(topic: str = "all") -> None:
    for prompt in iter_prompts(topic):
        print(f"[{prompt.name}]")
        print(prompt.text)
        print()


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    cleaned_argv, extracted_verbose = _extract_verbose_args(raw_argv)
    parser = _build_parser()
    args = parser.parse_args(cleaned_argv)
    args.verbose = max(args.verbose, extracted_verbose)
    if args.cmd == "prompts":
        _print_prompts(args.topic)
        return 0
    root = args.path.expanduser().resolve()
    ws = ensure_workspace(root)
    config = load_config(root)
    args.verbose = max(args.verbose, config.logging.verbose)
    logger = _setup_logging(ws, args.verbose)
    logger.debug("Parsed arguments: %s", args)
    if getattr(args, "cmd", None) == "rotate":
        auto = getattr(args, "auto", False)
        vlm_only = getattr(args, "vlm_only", False)
        vlm_fallback = getattr(args, "vlm_fallback", False)
        vlm_verify = getattr(args, "vlm_verify", False)
        vlm_compare = getattr(args, "vlm_compare", False)
        if vlm_only and not auto:
            parser.error("--vlm-only requires --auto")
        if vlm_fallback and not auto:
            parser.error("--vlm-fallback requires --auto")
        if vlm_compare and not (vlm_only or vlm_fallback):
            parser.error("--vlm-compare requires --vlm-only or --vlm-fallback")
        if vlm_verify and not auto:
            parser.error("--vlm-verify requires --auto")
        if vlm_verify and not (vlm_only or vlm_fallback):
            parser.error("--vlm-verify requires --vlm-only or --vlm-fallback")
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
            print(f"Scanning {root} recursively...", file=sys.stderr)

            def scan_progress(scanned: int, path: Path, indexed_count: int, error_count: int) -> None:
                if scanned == 1 or scanned % 25 == 0:
                    rel = path.relative_to(root) if path.is_relative_to(root) else path
                    print(
                        f"Scanned {scanned} images; indexed/updated={indexed_count}; errors={error_count}; current={rel}",
                        file=sys.stderr,
                    )

            indexed, errors = scan_folder(root, cat, logger, progress=scan_progress)
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
        elif args.cmd == "export-gallery":
            result = export_gallery(
                cat,
                args.output.expanduser().resolve(),
                title=args.title,
                query=args.query,
                tags=args.tag,
                top_k=(args.top_k if args.top_k is not None else config.search.top_k),
                min_tag_count=args.min_tag_count,
                max_cloud_tags=args.max_cloud_tags,
                image_max_size=args.image_max_size,
                overwrite=args.overwrite,
                backend=_embedding_backend(args, config) if args.query else None,
            )
            logger.info("Exported gallery to %s: %s exported, %s skipped", result.output, result.exported, result.skipped)
            print(f"Exported gallery to {result.output}: selected={result.selected}, exported={result.exported}, skipped={result.skipped}")
        elif args.cmd == "rotate":
            if args.photo_id is None:
                if not args.auto:
                    parser.error("rotate requires PHOTO_ID unless --auto is used for all indexed images")
                rotated = skipped = 0
                for photo in cat.list_photos():
                    detection = _detect_rotation_for_photo(cat, photo, args, logger)
                    print(_format_rotation_decision(photo.relative_path, detection))
                    if detection.degrees is None:
                        skipped += 1
                        continue
                    backup_path = _rotate_photo(cat, photo, detection.degrees, create_backup=not args.no_backup, logger=logger)
                    backup = f"; backup: {backup_path}" if backup_path else ""
                    print(f"Rotated {photo.relative_path} clockwise by {detection.degrees} degrees{backup}")
                    if args.vlm_verify and detection.source.startswith("fresh-vlm"):
                        verification = _verify_vlm_rotation_once(photo, args, logger)
                        print(_format_rotation_decision(f"{photo.relative_path} VLM verification after rotation", verification))
                        if verification.degrees is not None:
                            print(f"{photo.relative_path}: VLM verification still suggests rotation; no further automatic rotation will be applied.")
                    rotated += 1
                print(f"Auto-rotation complete: {rotated} rotated, {skipped} skipped")
                if rotated:
                    print("Thumbnails and embeddings were invalidated for rotated images; rerun thumbnails and embed when ready.")
                return 0

            try:
                photo_id = cat.resolve_photo_id(args.photo_id)
            except ValueError as exc:
                parser.error(str(exc))
            photo = cat.get_photo(photo_id)
            if photo is None:
                parser.error(f"No photo found for ID: {photo_id}")
            degrees = args.degrees
            if args.auto:
                detection = _detect_rotation_for_photo(cat, photo, args, logger)
                print(_format_rotation_decision("Auto-rotation", detection))
                if detection.degrees is None:
                    logger.info("No confident auto-rotation for %s", photo.source_path)
                    print("No image changes made. Use --degrees 90|180|270 if you want to rotate manually.")
                    return 0
                degrees = detection.degrees
            backup_path = _rotate_photo(cat, photo, degrees, create_backup=not args.no_backup, logger=logger)
            backup = f"; backup: {backup_path}" if backup_path else ""
            print(f"Rotated {photo.relative_path} clockwise by {degrees} degrees{backup}")
            if args.vlm_verify and args.auto and detection.source.startswith("fresh-vlm"):
                verification = _verify_vlm_rotation_once(photo, args, logger)
                print(_format_rotation_decision("VLM verification after rotation", verification))
                if verification.degrees is not None:
                    print("VLM verification still suggests rotation; no further automatic rotation will be applied.")
            print("Thumbnails and embeddings were invalidated; rerun thumbnails and embed when ready.")
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
            changed = cat.apply_vocabulary_to_tag_proposals(source=args.source)
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
        elif args.cmd in {"descriptions", "vlm-descriptions"}:
            if args.photo_id:
                try:
                    resolved = cat.resolve_photo_id(args.photo_id)
                except ValueError as exc:
                    parser.error(str(exc))
                photo = cat.get_photo(resolved)
                analysis = cat.get_image_analysis(resolved)
                _print_descriptions([(photo, analysis)] if photo is not None and analysis is not None else [])
            else:
                _print_descriptions(cat.list_image_analyses())
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
                accepted = cat.accept_tag_proposals(args.photo_id, source=args.source)
                scope = args.photo_id[:7] if args.photo_id else "all images"
                logger.info("Accepted %s pending tag proposals for %s", accepted, scope)
                print(f"Accepted {accepted} pending tag proposals for {scope}")
            else:
                if not args.photo_id or not args.tag:
                    parser.error("accept-tag requires PHOTO_ID and TAG unless --all is used")
                cat.accept_tag_proposal(args.photo_id, args.tag, source=args.source or "ai-zero-shot")
                logger.info("Accepted tag proposal %s for %s", args.tag, args.photo_id)
                print(f"Accepted tag for {args.photo_id}: {args.tag}")
        elif args.cmd == "reject-tag":
            cat.reject_tag_proposal(args.photo_id, args.tag, source=args.source or "ai-zero-shot")
            logger.info("Rejected tag proposal %s for %s", args.tag, args.photo_id)
            print(f"Rejected tag for {args.photo_id}: {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
