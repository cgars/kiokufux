from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
WORKSPACE_NAME = ".kiokufux"
CATALOG_NAME = "catalog.sqlite"


def workspace_for(root: Path, output_dir: Path | None = None) -> Path:
    return (output_dir or root / WORKSPACE_NAME).expanduser().resolve()


def ensure_workspace(root: Path, output_dir: Path | None = None) -> Path:
    ws = workspace_for(root, output_dir)
    for name in ("thumbnails", "embeddings", "indexes", "logs"):
        (ws / name).mkdir(parents=True, exist_ok=True)
    return ws


def catalog_path(root: Path, output_dir: Path | None = None) -> Path:
    return workspace_for(root, output_dir) / CATALOG_NAME
