from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .catalog import Catalog
from .config import KiokuFuxConfig, catalog_path, ensure_workspace, load_config, write_default_config
from .embeddings import backend_from_options, generate_embeddings
from .models import SearchResult
from .scanner import scan as scan_folder
from .search import search as run_search
from .sidecar import export_sidecars
from .thumbnails import generate_thumbnails

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
    if getattr(args, "cmd", None) not in {"embed", "search"}:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
