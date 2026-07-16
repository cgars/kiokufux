from __future__ import annotations

import html
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from string import Template

from PIL import Image, ImageOps

from .catalog import Catalog
from .models import Photo
from .embeddings import EmbeddingBackend
from .search import search as run_search

SCHEMA = "kiokufux.gallery.v1"
EXPORT_FORMAT = "kiokufux-gallery-inline-v1"


@dataclass(slots=True)
class GalleryExportResult:
    selected: int
    exported: int
    skipped: int
    output: Path


def _safe_name(photo: Photo, suffix: str | None = None) -> str:
    ext = suffix or photo.source_path.suffix or ".jpg"
    return f"{photo.photo_id}{ext.lower()}"


def _copy_image(src: Path, dst: Path, image_max_size: int | None = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if image_max_size is None:
        shutil.copy2(src, dst)
        return
    with Image.open(src) as img:
        fixed = ImageOps.exif_transpose(img)
        fixed.thumbnail((image_max_size, image_max_size), Image.Resampling.LANCZOS)
        fixed.convert("RGB").save(dst, "JPEG", quality=88)


def _write_thumbnail(photo: Photo, dst: Path, catalog: Catalog) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if photo.thumbnail_path:
        thumb = catalog.artifact_path(photo.thumbnail_path)
        if thumb.exists():
            shutil.copy2(thumb, dst)
            return
    with Image.open(photo.source_path) as img:
        fixed = ImageOps.exif_transpose(img)
        fixed.thumbnail((360, 360), Image.Resampling.LANCZOS)
        fixed.convert("RGB").save(dst, "JPEG", quality=82)


def published_tags(catalog: Catalog, photo_id: str) -> list[str]:
    tags = {catalog.canonical_tag(row.tag) for row in catalog.list_tags(photo_id) if row.source in {"manual", "auto"}}
    tags.discard("")
    return sorted(tags)


def _photos_for_export(catalog: Catalog, tags: list[str] | None = None, query_ids: set[str] | None = None) -> list[Photo]:
    photos = catalog.list_photos()
    if query_ids is not None:
        photos = [photo for photo in photos if photo.photo_id in query_ids]
    filters = [catalog.canonical_tag(tag) for tag in (tags or [])]
    if filters:
        filtered = []
        for photo in photos:
            pts = set(published_tags(catalog, photo.photo_id))
            if any(tag in pts for tag in filters):
                filtered.append(photo)
        photos = filtered
    return photos


def _tag_frequencies(items: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        for tag in set(item.get("tags", [])):
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def build_gallery_document(title: str, items: list[dict], source_summary: dict) -> dict:
    return {
        "schema": SCHEMA,
        "title": title,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_summary": source_summary,
        "item_count": len(items),
        "items": items,
        "tag_frequencies": _tag_frequencies(items),
    }


def export_gallery(
    catalog: Catalog,
    output: Path,
    *,
    title: str = "KiokuFux Gallery",
    query: str | None = None,
    tags: list[str] | None = None,
    top_k: int = 10,
    min_tag_count: int = 2,
    max_cloud_tags: int = 40,
    image_max_size: int | None = None,
    overwrite: bool = False,
    backend: EmbeddingBackend | None = None,
) -> GalleryExportResult:
    if output.exists():
        if not overwrite and not _is_legacy_gallery_export(output):
            raise FileExistsError(f"Output directory already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "thumbnails").mkdir()

    query_ids = None
    if query:
        query_ids = {r.photo_id for r in run_search(catalog, query, top_k=top_k, backend=backend)}
    photos = _photos_for_export(catalog, tags=tags, query_ids=query_ids)
    items: list[dict] = []
    skipped = 0
    for photo in photos:
        if not photo.source_path.exists():
            skipped += 1
            continue
        image_ext = ".jpg" if image_max_size else (photo.source_path.suffix or ".jpg")
        image_rel = f"images/{_safe_name(photo, image_ext)}"
        thumb_rel = f"thumbnails/{photo.photo_id}.jpg"
        try:
            _copy_image(photo.source_path, output / image_rel, image_max_size=image_max_size)
            _write_thumbnail(photo, output / thumb_rel, catalog)
        except Exception:
            skipped += 1
            continue
        analysis = catalog.get_image_analysis(photo.photo_id)
        caption = analysis.caption if analysis else None
        description = analysis.description if analysis else None
        items.append({
            "photo_id": photo.photo_id,
            "image_path": image_rel,
            "thumbnail_path": thumb_rel,
            "filename": photo.source_path.name,
            "relative_path": photo.relative_path,
            "caption": caption,
            "description": description,
            "tags": published_tags(catalog, photo.photo_id),
            "datetime_original": photo.exif_datetime_original,
            "dimensions": {"width": photo.width, "height": photo.height},
            "metadata": {"mime_type": photo.mime_type, "gps": {"lat": photo.exif_gps_lat, "lon": photo.exif_gps_lon}},
        })
    doc = build_gallery_document(title, items, {"query": query, "tags": tags or [], "selected": len(photos), "skipped": skipped})
    (output / "gallery.json").write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    _write_gallery_template_files(output, title, min_tag_count, max_cloud_tags, doc)
    return GalleryExportResult(len(photos), len(items), skipped, output)



def _template_root():
    return resources.files("kiokufux").joinpath("templates", "gallery")


def _is_legacy_gallery_export(output: Path) -> bool:
    """Recognize and replace exports whose file:// JSON fetch cannot work."""
    index = output / "index.html"
    manifest = output / "gallery.json"
    if not index.is_file() or not manifest.is_file():
        return False
    try:
        html_text = index.read_text(encoding="utf-8")
    except OSError:
        return False
    return 'fetch("gallery.json")' in html_text or '<script src="gallery.js"></script>' in html_text


def _write_gallery_template_files(
    output: Path,
    title: str,
    min_tag_count: int,
    max_cloud_tags: int,
    document: dict,
) -> None:
    template_root = _template_root()
    config_json = json.dumps({"minTagCount": min_tag_count, "maxCloudTags": max_cloud_tags})
    # Embed the data so opening index.html directly from disk does not require a
    # fetch(), which browsers block for file:// URLs. Escaping '<' prevents data
    # containing a closing script tag from ending this JSON script element.
    gallery_json = json.dumps(document, sort_keys=True).replace("<", "\\u003c")
    style_css = template_root.joinpath("style.css").read_text(encoding="utf-8")
    gallery_js = template_root.joinpath("gallery.js").read_text(encoding="utf-8")
    # Keep the browser from interpreting a future literal closing tag in an
    # asset as the end of its inline element.
    style_css = style_css.replace("</style", "<\\/style")
    gallery_js = gallery_js.replace("</script", "<\\/script")
    index_template = Template(template_root.joinpath("index.html").read_text(encoding="utf-8"))
    rendered = index_template.safe_substitute(
        title=html.escape(title),
        export_format=EXPORT_FORMAT,
        config_json=config_json,
        gallery_json=gallery_json,
        style_css=style_css,
        gallery_js=gallery_js,
    )
    (output / "index.html").write_text(rendered, encoding="utf-8")
    for asset_name in ("style.css", "gallery.js"):
        (output / asset_name).write_text(template_root.joinpath(asset_name).read_text(encoding="utf-8"), encoding="utf-8")
