from __future__ import annotations

import json
from pathlib import Path
from .catalog import Catalog

SCHEMA = "fotofux.sidecar.v1"


def sidecar_document(photo, embedding_model: str | None = None) -> dict:
    return {
        "schema": SCHEMA, "photo_id": photo.photo_id, "source_path": str(photo.source_path), "file_hash": photo.file_hash,
        "metadata": {"width": photo.width, "height": photo.height, "datetime_original": photo.exif_datetime_original, "gps": {"lat": photo.exif_gps_lat, "lon": photo.exif_gps_lon}},
        "semantic": {"embedding_model": embedding_model, "auto_tags": [], "caption": None, "status": photo.embedding_status},
        "review": {"state": "unreviewed"},
    }


def export_sidecars(catalog: Catalog, export_dir: Path | None = None) -> int:
    count = 0
    if export_dir: export_dir.mkdir(parents=True, exist_ok=True)
    for photo in catalog.list_photos():
        target = (export_dir / f"{photo.photo_id}.fotofux.json") if export_dir else photo.source_path.with_name(photo.source_path.name + ".fotofux.json")
        target.write_text(json.dumps(sidecar_document(photo), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        count += 1
    return count
