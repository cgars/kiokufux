from __future__ import annotations

import argparse, logging
from pathlib import Path
from .catalog import Catalog
from .config import catalog_path, ensure_workspace, workspace_for
from .scanner import scan as scan_folder
from .thumbnails import generate_thumbnails
from .embeddings import backend_from_options, generate_embeddings
from .search import search as run_search
from .sidecar import export_sidecars


def _catalog(root: Path) -> tuple[Path, Catalog]:
    ws = ensure_workspace(root)
    cat = Catalog(catalog_path(root)); cat.init_schema(); return ws, cat


def _logger(ws: Path) -> logging.Logger:
    logging.basicConfig(filename=ws / "logs" / "kiokufux.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("kiokufux")


def _add_embedding_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-backend", choices=["auto", "openclip", "simple"], default="auto")
    parser.add_argument("--openclip-model", help="OpenCLIP model architecture, e.g. ViT-B-32")
    parser.add_argument("--openclip-pretrained", help="OpenCLIP pretrained weights tag, e.g. laion2b_s34b_b79k")


def _embedding_backend(args: argparse.Namespace):
    return backend_from_options(args.embedding_backend, args.openclip_model, args.openclip_pretrained)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kiokufux"); sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["init", "scan", "thumbnails", "export-sidecars"]:
        sp = sub.add_parser(name); sp.add_argument("path", type=Path)
    e = sub.add_parser("embed"); e.add_argument("path", type=Path); _add_embedding_options(e)
    s = sub.add_parser("search"); s.add_argument("path", type=Path); s.add_argument("query"); s.add_argument("--top-k", type=int, default=10); _add_embedding_options(s)
    args = p.parse_args(argv); root = args.path.expanduser().resolve()
    if args.cmd == "init":
        ws = ensure_workspace(root); Catalog(catalog_path(root)).init_schema(); print(f"Initialized KiokuFux workspace at {ws}"); return 0
    ws, cat = _catalog(root); logger = _logger(ws)
    with cat:
        if args.cmd == "scan":
            indexed, errors = scan_folder(root, cat, logger); print(f"Scan complete: {indexed} indexed/updated, {errors} errors")
        elif args.cmd == "thumbnails":
            print(f"Generated {generate_thumbnails(cat, ws)} thumbnails")
        elif args.cmd == "embed":
            print(f"Generated {generate_embeddings(cat, ws, _embedding_backend(args))} embeddings")
        elif args.cmd == "search":
            for i, r in enumerate(run_search(cat, args.query, args.top_k, _embedding_backend(args)), 1):
                print(f"{i}\tscore={r.score:.4f}\tpath={r.source_path}\tthumb={r.thumbnail_path}\tmetadata={r.metadata_summary}")
        elif args.cmd == "export-sidecars":
            print(f"Exported {export_sidecars(cat)} sidecars")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
