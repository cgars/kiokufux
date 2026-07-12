from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from .catalog import Catalog
from .config import SUPPORTED_EXTENSIONS, WORKSPACE_NAME
from .hashing import file_sha256, photo_id_for_hash
from .metadata import extract_metadata
from .models import Photo


def iter_images(root: Path):
    for path in root.rglob("*"):
        if WORKSPACE_NAME in path.parts or not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def scan(root: Path, catalog: Catalog, logger: logging.Logger) -> tuple[int, int]:
    seen: list[str] = []
    indexed = errors = 0
    for path in iter_images(root):
        rel = str(path.relative_to(root))
        try:
            file_hash = file_sha256(path)
            photo_id = photo_id_for_hash(file_hash)
            seen.append(photo_id)
            existing = catalog.get_photo(photo_id)
            resolved = path.resolve()
            if existing and existing.file_hash == file_hash and not existing.error:
                if existing.missing or existing.source_path != resolved or existing.relative_path != rel:
                    catalog.upsert_photo(replace(existing, source_path=resolved, relative_path=rel, missing=False, error=None))
                    indexed += 1
                continue
            meta = extract_metadata(path)
            catalog.upsert_photo(Photo(photo_id=photo_id, source_path=resolved, relative_path=rel, file_hash=file_hash, **meta))
            indexed += 1
        except Exception as exc:
            errors += 1; logger.exception("Failed to scan %s", path)
            try:
                fh = file_sha256(path); pid = photo_id_for_hash(fh)
            except Exception:
                fh = f"unreadable:{path.resolve()}"; pid = photo_id_for_hash(fh)
            seen.append(pid); catalog.record_error(pid, path.resolve(), rel, fh, str(exc))
    catalog.mark_missing_except(seen)
    return indexed, errors
