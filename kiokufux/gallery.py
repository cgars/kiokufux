from __future__ import annotations

import html
import json
import logging
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path, PureWindowsPath
from string import Template

from PIL import Image, ImageOps

from .catalog import Catalog
from .models import Photo
from .embeddings import EmbeddingBackend
from .search import search as run_search
from .sidecar import FaceSidecarIndex

SCHEMA = "kiokufux.gallery.v1"
EXPORT_FORMAT = "kiokufux-gallery-inline-v1"
FACE_MODES = {"none", "confirmed", "grouped", "detected"}
logger = logging.getLogger(__name__)


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


def _write_thumbnail(photo: Photo, source: Path, dst: Path, catalog: Catalog) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if photo.thumbnail_path:
        thumb = catalog.artifact_path(photo.thumbnail_path)
        if thumb.exists():
            try:
                shutil.copy2(thumb, dst)
                return
            except OSError:
                pass
    with Image.open(source) as img:
        fixed = ImageOps.exif_transpose(img)
        fixed.thumbnail((360, 360), Image.Resampling.LANCZOS)
        fixed.convert("RGB").save(dst, "JPEG", quality=82)


def _relative_source_candidates(collection_root: Path, relative_path: str) -> list[Path]:
    """Return safe native and Windows-separator interpretations of a stored relative path."""
    native = Path(relative_path)
    windows = PureWindowsPath(relative_path)
    if native.is_absolute() or windows.anchor:
        return []
    variants = [Path(*windows.parts)] if "\\" in relative_path else [native]

    root = collection_root.resolve()
    candidates: list[Path] = []
    for variant in variants:
        try:
            candidate = (root / variant).resolve()
        except (OSError, RuntimeError):
            continue
        if candidate.is_relative_to(root) and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _source_path(photo: Photo, collection_root: Path) -> Path | None:
    """Locate a photo in the current collection before trying its indexed absolute path."""
    try:
        for relative_candidate in _relative_source_candidates(collection_root, photo.relative_path):
            if relative_candidate.is_file():
                return relative_candidate
    except (OSError, RuntimeError):
        pass
    try:
        if photo.source_path.is_file():
            return photo.source_path
    except OSError:
        pass
    return None


def _published_tags_by_photo(catalog: Catalog, photo_id: str | None = None) -> dict[str, list[str]]:
    sql = """
        SELECT pt.photo_id, COALESCE(ta.tag, pt.tag) AS canonical_tag
        FROM photo_tags pt
        LEFT JOIN tag_aliases ta ON ta.alias=pt.tag
        WHERE pt.source IN ('manual','auto')
    """
    params: tuple[str, ...] = ()
    if photo_id is not None:
        sql += " AND pt.photo_id=?"
        params = (photo_id,)
    sql += " ORDER BY pt.photo_id, canonical_tag"
    tags: dict[str, set[str]] = {}
    for row in catalog.conn.execute(sql, params):
        canonical = str(row["canonical_tag"] or "")
        if canonical:
            tags.setdefault(str(row["photo_id"]), set()).add(canonical)
    return {key: sorted(values) for key, values in tags.items()}


def published_tags(catalog: Catalog, photo_id: str) -> list[str]:
    return _published_tags_by_photo(catalog, photo_id).get(photo_id, [])


def _visible_people(faces: dict, face_mode: str) -> list[dict]:
    people: dict[str, dict] = {}
    unknown_count = 0
    for occurrence in faces.get("occurrences", []):
        identity = occurrence.get("identity", {})
        status = identity.get("status")
        if status == "confirmed" and identity.get("person_id"):
            person_id = str(identity["person_id"])
            display_name = identity.get("display_name")
            friendly_name = identity.get("friendly_name")
            identity_id = f"person:{person_id}"
            people[identity_id] = {
                "identity_id": identity_id,
                "person_id": person_id,
                "label": display_name or friendly_name or person_id,
                "display_name": display_name,
                "friendly_name": friendly_name,
                "status": "confirmed",
            }
            continue
        if status == "provisional" and face_mode in {"grouped", "detected"}:
            group_id = identity.get("group_id")
            if not group_id or identity.get("group_conflict"):
                continue
            group_id = str(group_id)
            identity_id = f"group:{group_id}"
            people[identity_id] = {
                "identity_id": identity_id,
                "group_id": group_id,
                "label": identity.get("friendly_name") or group_id,
                "friendly_name": identity.get("friendly_name"),
                "status": "provisional",
                "review_state": identity.get("group_review_state"),
            }
            continue
        if status == "ungrouped" and face_mode == "detected":
            unknown_count += 1
    if unknown_count:
        people["unknown"] = {
            "identity_id": "unknown",
            "label": "Unknown people",
            "status": "ungrouped",
            "count_in_photo": unknown_count,
        }
    status_order = {"confirmed": 0, "provisional": 1, "ungrouped": 2}
    return sorted(
        people.values(),
        key=lambda person: (status_order[person["status"]], person["label"].casefold(), person["identity_id"]),
    )


def _people_by_photo(catalog: Catalog, face_index: FaceSidecarIndex, face_mode: str) -> dict[str, list[dict]]:
    return {
        photo.photo_id: _visible_people(face_index.for_photo(photo.photo_id), face_mode)
        for photo in catalog.list_photos()
    }


def _visible_face_boxes(faces: dict, face_mode: str) -> list[dict]:
    boxes: list[dict] = []
    for occurrence in faces.get("occurrences", []):
        identity = occurrence.get("identity", {})
        status = identity.get("status")
        if status == "confirmed" and identity.get("person_id"):
            person_id = str(identity["person_id"])
            label = identity.get("display_name") or identity.get("friendly_name") or person_id
            identity_id = f"person:{person_id}"
        elif status == "provisional" and face_mode in {"grouped", "detected"} and not identity.get("group_conflict"):
            group_id = identity.get("group_id")
            if not group_id:
                continue
            identity_id = f"group:{group_id}"
            label = identity.get("friendly_name") or str(group_id)
        elif status == "ungrouped" and face_mode == "detected":
            identity_id = "unknown"
            label = "Unknown person"
        else:
            continue

        raw_box = occurrence.get("box") or {}
        try:
            values = {key: float(raw_box[key]) for key in ("x1", "y1", "x2", "y2")}
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in values.values()):
            continue
        values = {key: min(1.0, max(0.0, value)) for key, value in values.items()}
        if values["x2"] <= values["x1"] or values["y2"] <= values["y1"]:
            continue
        boxes.append({
            "identity_id": identity_id,
            "label": label,
            "status": status,
            "box": values,
        })
    return boxes


def _face_boxes_by_photo(catalog: Catalog, face_index: FaceSidecarIndex, face_mode: str) -> dict[str, list[dict]]:
    return {
        photo.photo_id: _visible_face_boxes(face_index.for_photo(photo.photo_id), face_mode)
        for photo in catalog.list_photos()
    }


def _resolve_person_ids(people_by_photo: dict[str, list[dict]], selectors: list[str] | None) -> set[str]:
    aliases: dict[str, set[str]] = {}
    for people in people_by_photo.values():
        for person in people:
            if person.get("status") != "confirmed":
                continue
            for value in (person["person_id"], person.get("display_name"), person.get("friendly_name"), person["identity_id"]):
                if value:
                    aliases.setdefault(str(value).strip().casefold(), set()).add(person["identity_id"])

    resolved: set[str] = set()
    for selector in selectors or []:
        matches = aliases.get(selector.strip().casefold(), set())
        if not matches:
            raise ValueError(f"No confirmed person matches {selector!r}")
        if len(matches) > 1:
            raise ValueError(f"Confirmed person name {selector!r} is ambiguous; use a person_id or friendly name")
        resolved.update(matches)
    return resolved


def _resolve_group_ids(people_by_photo: dict[str, list[dict]], selectors: list[str] | None) -> set[str]:
    aliases: dict[str, set[str]] = {}
    for people in people_by_photo.values():
        for person in people:
            if person.get("status") != "provisional":
                continue
            for value in (person["group_id"], person.get("friendly_name"), person["identity_id"]):
                if value:
                    aliases.setdefault(str(value).strip().casefold(), set()).add(person["identity_id"])

    resolved: set[str] = set()
    for selector in selectors or []:
        matches = aliases.get(selector.strip().casefold(), set())
        if not matches:
            raise ValueError(f"No provisional face group matches {selector!r}")
        if len(matches) > 1:
            raise ValueError(f"Provisional face group name {selector!r} is ambiguous; use a group_id")
        resolved.update(matches)
    return resolved


def _photos_for_export(
    catalog: Catalog,
    tags: list[str] | None = None,
    query_ids: set[str] | None = None,
    identity_ids: set[str] | None = None,
    people_by_photo: dict[str, list[dict]] | None = None,
    tags_by_photo: dict[str, list[str]] | None = None,
) -> list[Photo]:
    photos = catalog.list_photos()
    if query_ids is not None:
        photos = [photo for photo in photos if photo.photo_id in query_ids]
    filters = [catalog.canonical_tag(tag) for tag in (tags or [])]
    if filters:
        tag_index = tags_by_photo if tags_by_photo is not None else _published_tags_by_photo(catalog)
        filtered = []
        for photo in photos:
            pts = set(tag_index.get(photo.photo_id, []))
            if any(tag in pts for tag in filters):
                filtered.append(photo)
        photos = filtered
    if identity_ids:
        index = people_by_photo or {}
        photos = [
            photo for photo in photos
            if any(person["identity_id"] in identity_ids for person in index.get(photo.photo_id, []))
        ]
    return photos


def _tag_frequencies(items: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        for tag in set(item.get("tags", [])):
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _people_frequencies(items: list[dict]) -> list[dict]:
    people: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for item in items:
        distinct = {}
        for person in item.get("people", []):
            identity_id = person.get("identity_id") or f"person:{person['person_id']}"
            distinct[identity_id] = person
        for identity_id, person in distinct.items():
            people[identity_id] = {key: value for key, value in person.items() if key != "count_in_photo"}
            counts[identity_id] = counts.get(identity_id, 0) + 1
    return [
        {**people[identity_id], "count": counts[identity_id]}
        for identity_id in sorted(people, key=lambda iid: (-counts[iid], people[iid]["label"].casefold(), iid))
    ]


def build_gallery_document(title: str, items: list[dict], source_summary: dict) -> dict:
    return {
        "schema": SCHEMA,
        "title": title,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_summary": source_summary,
        "item_count": len(items),
        "items": items,
        "tag_frequencies": _tag_frequencies(items),
        "people_frequencies": _people_frequencies(items),
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
    workspace: Path | None = None,
    collection_root: Path | None = None,
    face_mode: str = "none",
    face_boxes: bool = False,
    people: list[str] | None = None,
    face_groups: list[str] | None = None,
    unknown_faces: bool = False,
) -> GalleryExportResult:
    if face_mode not in FACE_MODES:
        raise ValueError(f"Unsupported gallery face mode: {face_mode!r}")
    if face_boxes and face_mode == "none":
        raise ValueError("Gallery face boxes require a published face mode")

    source_root = (collection_root or catalog.db_path.parent.parent).resolve()

    person_selectors = people or []
    group_selectors = face_groups or []
    selection_index: dict[str, list[dict]] = {}
    published_people_index: dict[str, list[dict]] = {}
    published_face_boxes_index: dict[str, list[dict]] = {}
    if face_mode != "none" or person_selectors or group_selectors or unknown_faces:
        face_index = FaceSidecarIndex.load(workspace or catalog.db_path.parent)
        selection_index = _people_by_photo(catalog, face_index, "detected")
        if face_mode != "none":
            published_people_index = _people_by_photo(catalog, face_index, face_mode)
            if face_boxes:
                published_face_boxes_index = _face_boxes_by_photo(catalog, face_index, face_mode)
    identity_ids = _resolve_person_ids(selection_index, person_selectors)
    identity_ids.update(_resolve_group_ids(selection_index, group_selectors))
    if unknown_faces:
        identity_ids.add("unknown")

    query_ids = None
    if query:
        query_ids = {r.photo_id for r in run_search(catalog, query, top_k=top_k, backend=backend)}
    published_tag_index = _published_tags_by_photo(catalog)
    photos = _photos_for_export(
        catalog,
        tags=tags,
        query_ids=query_ids,
        identity_ids=identity_ids,
        people_by_photo=selection_index,
        tags_by_photo=published_tag_index,
    )

    if output.exists():
        if not overwrite and not _is_legacy_gallery_export(output):
            raise FileExistsError(f"Output directory already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "thumbnails").mkdir()

    items: list[dict] = []
    skipped = 0
    for photo in photos:
        source = _source_path(photo, source_root)
        if source is None:
            relative_candidates = ", ".join(
                str(candidate) for candidate in _relative_source_candidates(source_root, photo.relative_path)
            ) or "(invalid relative path)"
            logger.warning(
                "Skipping gallery photo %s (%s): source not found at relative candidate(s) %s or indexed path %s",
                photo.photo_id,
                photo.relative_path,
                relative_candidates,
                photo.source_path,
            )
            skipped += 1
            continue
        image_ext = ".jpg" if image_max_size else (source.suffix or ".jpg")
        image_rel = f"images/{_safe_name(photo, image_ext)}"
        thumb_rel = f"thumbnails/{photo.photo_id}.jpg"
        try:
            _copy_image(source, output / image_rel, image_max_size=image_max_size)
        except Exception as exc:
            (output / image_rel).unlink(missing_ok=True)
            logger.warning(
                "Skipping gallery photo %s (%s): source export failed with %s: %s",
                photo.photo_id,
                source,
                type(exc).__name__,
                exc,
            )
            skipped += 1
            continue
        try:
            _write_thumbnail(photo, source, output / thumb_rel, catalog)
        except Exception as exc:
            (output / thumb_rel).unlink(missing_ok=True)
            thumb_rel = image_rel
            logger.warning(
                "Gallery thumbnail failed for %s (%s) with %s: %s; using the exported image as its preview",
                photo.photo_id,
                source,
                type(exc).__name__,
                exc,
            )
        analysis = catalog.get_image_analysis(photo.photo_id)
        caption = analysis.caption if analysis else None
        description = analysis.description if analysis else None
        item = {
            "photo_id": photo.photo_id,
            "image_path": image_rel,
            "thumbnail_path": thumb_rel,
            "filename": source.name,
            "relative_path": photo.relative_path,
            "caption": caption,
            "description": description,
            "tags": published_tag_index.get(photo.photo_id, []),
            "datetime_original": photo.exif_datetime_original,
            "dimensions": {"width": photo.width, "height": photo.height},
            "metadata": {"mime_type": photo.mime_type, "gps": {"lat": photo.exif_gps_lat, "lon": photo.exif_gps_lon}},
        }
        if face_mode != "none":
            item["people"] = published_people_index.get(photo.photo_id, [])
        if face_boxes:
            item["face_boxes"] = published_face_boxes_index.get(photo.photo_id, [])
        items.append(item)
    source_summary = {
        "query": query,
        "tags": tags or [],
        "faces": face_mode,
        "selected": len(photos),
        "skipped": skipped,
    }
    if face_mode != "none":
        source_summary["people"] = person_selectors
        source_summary["face_groups"] = group_selectors
        source_summary["unknown_faces"] = unknown_faces
        if face_boxes:
            source_summary["face_boxes"] = True
    else:
        if person_selectors:
            source_summary["person_filter_count"] = len(person_selectors)
        if group_selectors:
            source_summary["face_group_filter_count"] = len(group_selectors)
        if unknown_faces:
            source_summary["unknown_face_filter"] = True
    doc = build_gallery_document(title, items, source_summary)
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
