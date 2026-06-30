from __future__ import annotations

from pathlib import Path
from .catalog import Catalog


def generate_thumbnails(catalog: Catalog, workspace: Path, max_size: int = 512) -> int:
    out = workspace / "thumbnails"; out.mkdir(parents=True, exist_ok=True); count = 0
    for photo in catalog.list_photos():
        target = out / f"{photo.photo_id}.jpg"
        if target.exists() and photo.thumbnail_path:
            continue
        try:
            from PIL import Image, ImageOps
            with Image.open(photo.source_path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                img.save(target, "JPEG", quality=85)
            catalog.set_thumbnail(photo.photo_id, target); count += 1
        except Exception as exc:
            catalog.conn.execute("UPDATE photos SET error=? WHERE photo_id=?", (f"thumbnail: {exc}", photo.photo_id)); catalog.conn.commit()
    return count
