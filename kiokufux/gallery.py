from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from .catalog import Catalog
from .models import Photo
from .search import search as run_search
from .embeddings import EmbeddingBackend

SCHEMA = "kiokufux.gallery.v1"


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
        if not overwrite:
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
    (output / "index.html").write_text(_index_html(title, min_tag_count, max_cloud_tags), encoding="utf-8")
    return GalleryExportResult(len(photos), len(items), skipped, output)


def _index_html(title: str, min_tag_count: int, max_cloud_tags: int) -> str:
    config = json.dumps({"minTagCount": min_tag_count, "maxCloudTags": max_cloud_tags})
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}}header{{position:sticky;top:0;background:#181818;padding:1rem;z-index:1}}input,button{{font:inherit}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;padding:1rem}}.card{{background:#222;border:0;color:#eee;text-align:left;padding:0;border-radius:10px;overflow:hidden;cursor:pointer}}.card img{{width:100%;aspect-ratio:1;object-fit:cover;display:block}}.card span{{display:block;padding:.5rem}}.cloud button{{margin:.25rem;border-radius:999px;border:1px solid #555;background:#242424;color:#eee;padding:.25rem .55rem}}.active{{outline:2px solid #8fd}}#empty{{padding:2rem;color:#bbb}}dialog{{max-width:min(96vw,1100px);background:#181818;color:#eee;border:1px solid #555;border-radius:12px}}dialog img{{max-width:90vw;max-height:70vh}}.meta{{color:#ccc}}a{{color:#8fd}}</style></head><body><header><h1>{title}</h1><input id="q" type="search" placeholder="Search filename, caption, description, tags" aria-label="Search metadata"> <button id="clear">Clear</button><p id="state"></p><div id="cloud" class="cloud" aria-label="Tag cloud"></div></header><main><div id="grid" class="grid"></div><p id="empty" hidden>No matching photos. Clear the search or selected tag.</p></main><dialog id="box"><button id="close">Close (Esc)</button> <button id="prev">Previous</button> <button id="next">Next</button><figure><img id="full" alt=""><figcaption id="cap"></figcaption></figure><p id="detail" class="meta"></p></dialog><script>const CFG={config};let data,all=[],shown=[],tag='',idx=0;const q=document.querySelector('#q'),grid=document.querySelector('#grid'),state=document.querySelector('#state'),cloud=document.querySelector('#cloud'),empty=document.querySelector('#empty'),box=document.querySelector('#box');
function text(i){{return [i.filename,i.relative_path,i.caption,i.description,...(i.tags||[])].join(' ').toLowerCase().replace(/\\s+/g,' ')}}
function apply(){{const query=q.value.trim().toLowerCase().replace(/\\s+/g,' ');shown=all.filter(i=>(!tag||i.tags.includes(tag))&&(!query||text(i).includes(query)));state.textContent=`${{shown.length}} of ${{all.length}} photos${{query?` · search: "${{query}}"`:''}}${{tag?` · tag: ${{tag}}`:''}}`;empty.hidden=shown.length!==0;grid.innerHTML='';shown.forEach((i,n)=>{{let b=document.createElement('button');b.className='card';b.innerHTML=`<img loading="lazy" src="${{i.thumbnail_path}}" alt="${{alt(i)}}"><span>${{i.caption||i.filename}}</span>`;b.onclick=()=>openBox(n);grid.appendChild(b)}})}}
function alt(i){{return i.caption||i.description||i.filename}}
function tagCloud(){{let ents=Object.entries(data.tag_frequencies).filter(e=>e[1]>=CFG.minTagCount).sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])).slice(0,CFG.maxCloudTags);let max=Math.max(1,...ents.map(e=>e[1]));ents.forEach(([t,c])=>{{let b=document.createElement('button');b.textContent=`${{t}} · ${{c}}`;b.style.fontSize=(.9+Math.log(c+1)/Math.log(max+1)*.9)+'rem';b.onclick=()=>{{tag=tag===t?'':t;[...cloud.children].forEach(x=>x.classList.toggle('active',x===b&&tag));apply()}};cloud.appendChild(b)}})}}
function openBox(n){{idx=n;let i=shown[idx];document.querySelector('#full').src=i.image_path;document.querySelector('#full').alt=alt(i);document.querySelector('#cap').textContent=i.caption||i.filename;document.querySelector('#detail').textContent=[i.description,(i.tags||[]).join(', '),i.datetime_original].filter(Boolean).join(' · ');box.showModal()}}
function move(d){{if(!shown.length)return;openBox((idx+d+shown.length)%shown.length)}}
document.querySelector('#clear').onclick=()=>{{q.value='';tag='';[...cloud.children].forEach(x=>x.classList.remove('active'));apply()}};document.querySelector('#close').onclick=()=>box.close();document.querySelector('#prev').onclick=()=>move(-1);document.querySelector('#next').onclick=()=>move(1);document.addEventListener('keydown',e=>{{if(box.open&&e.key==='ArrowLeft')move(-1);if(box.open&&e.key==='ArrowRight')move(1)}});q.oninput=apply;fetch('gallery.json').then(r=>r.json()).then(j=>{{data=j;all=j.items;tagCloud();apply()}});</script></body></html>'''
