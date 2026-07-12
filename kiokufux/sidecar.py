from __future__ import annotations

import json
from pathlib import Path
from .catalog import Catalog

SCHEMA = "kiokufux.sidecar.v1"


def sidecar_document(photo, embedding_model: str | None = None, tags: list | None = None, tag_proposals: list | None = None, image_analysis=None, proposal_evidence: dict | None = None) -> dict:
    tags = tags or []
    tag_proposals = tag_proposals or []
    proposal_evidence = proposal_evidence or {}
    caption = image_analysis.caption if image_analysis is not None else None
    description = image_analysis.description if image_analysis is not None else None
    vlm = None if image_analysis is None else {"model": image_analysis.model_name, "model_version": image_analysis.model_version, "source": image_analysis.source, "description": image_analysis.description, "scene": image_analysis.scene, "activity": image_analysis.activity, "objects": image_analysis.objects, "aesthetic": image_analysis.aesthetic, "warnings": image_analysis.warnings}
    proposals = []
    for p in tag_proposals:
        item = {"tag": p.tag, "confidence": p.confidence, "source": p.source, "status": p.status}
        evidence = proposal_evidence.get((p.tag, p.source), {})
        if evidence.get("category_hint") is not None:
            item["category_hint"] = evidence["category_hint"]
        if evidence.get("evidence") is not None:
            item["evidence"] = evidence["evidence"]
        proposals.append(item)
    return {
        "schema": SCHEMA, "photo_id": photo.photo_id, "source_path": str(photo.source_path), "file_hash": photo.file_hash,
        "metadata": {"width": photo.width, "height": photo.height, "datetime_original": photo.exif_datetime_original, "gps": {"lat": photo.exif_gps_lat, "lon": photo.exif_gps_lon}},
        "semantic": {"embedding_model": embedding_model, "auto_tags": [t.tag for t in tags if t.source == "auto"], "caption": caption, "description": description, "vlm": vlm, "status": photo.embedding_status},
        "review": {"state": "unreviewed", "tags": [t.tag for t in tags if t.source == "manual"], "tag_proposals": proposals},
    }


def export_sidecars(catalog: Catalog, export_dir: Path | None = None) -> int:
    count = 0
    if export_dir: export_dir.mkdir(parents=True, exist_ok=True)
    for photo in catalog.list_photos():
        target = (export_dir / f"{photo.photo_id}.kiokufux.json") if export_dir else photo.source_path.with_name(photo.source_path.name + ".kiokufux.json")
        target.write_text(json.dumps(sidecar_document(photo, tags=catalog.list_tags(photo.photo_id), tag_proposals=catalog.list_tag_proposals(photo.photo_id), image_analysis=catalog.get_image_analysis(photo.photo_id), proposal_evidence=catalog.tag_proposal_evidence(photo.photo_id)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        count += 1
    return count
