from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .catalog import Catalog
from .faces import friendly_group_name, person_friendly_name
from .hashing import photo_id_for_hash

SCHEMA = "kiokufux.sidecar.v2"


def _empty_faces() -> dict[str, Any]:
    return {"scan_status": "not_scanned", "model_key": None, "occurrences": []}


class FaceSidecarIndex:
    """Read-only adapter from the face workspace to sidecar face blocks."""

    def __init__(self, by_photo: dict[str, dict[str, Any]] | None = None):
        self.by_photo = by_photo or {}

    @classmethod
    def load(cls, workspace: Path) -> "FaceSidecarIndex":
        db_path = workspace / "faces.sqlite"
        if not db_path.exists():
            return cls()
        review = _read_json(workspace / "face-review.json", {})
        people = _read_json(workspace / "people.json", {})
        rejected = set(review.get("rejected_face_ids", []))
        excluded = set(review.get("excluded_face_ids", []))
        person_faces: dict[str, list[str]] = review.get("person_faces", {}) or {}
        people_by_id = {
            p.get("person_id"): {
                "person_id": p.get("person_id"),
                "friendly_name": p.get("friendly_name") or person_friendly_name(str(p.get("person_id", ""))),
                "display_name": p.get("display_name"),
            }
            for p in people.get("people", []) if p.get("person_id")
        }
        face_to_people: dict[str, list[str]] = {}
        for person_id, face_ids in sorted(person_faces.items()):
            for face_id in face_ids:
                face_to_people.setdefault(face_id, []).append(person_id)
        by_photo: dict[str, dict[str, Any]] = {}
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error:
            return cls()
        try:
            scanned_rows = conn.execute("SELECT image_id, model_key FROM scanned_images ORDER BY image_id").fetchall()
            for row in scanned_rows:
                by_photo[str(row["image_id"])] = {"scan_status": "scanned", "model_key": row["model_key"], "occurrences": []}
            group_rows = conn.execute("""
                SELECT m.face_id,g.group_id,g.cluster_run_id,g.review_state,g.conflict
                FROM face_group_members m JOIN face_groups g USING(group_id)
                ORDER BY m.face_id,g.group_id
            """).fetchall()
            group_by_face = {str(r["face_id"]): dict(r) for r in group_rows}
            rows = conn.execute("""
                SELECT face_id,image_id,x1,y1,x2,y2,confidence,quality,
                  backend_id,model_id,model_version,preprocessing_version,embedding_dimensions
                FROM face_occurrences ORDER BY image_id,y1,x1,face_id
            """).fetchall()
        except sqlite3.Error:
            return cls(by_photo)
        finally:
            conn.close()
        for row in rows:
            face_id = str(row["face_id"])
            photo = by_photo.setdefault(str(row["image_id"]), {"scan_status": "scanned", "model_key": _row_model_key(row), "occurrences": []})
            if photo.get("model_key") is None:
                photo["model_key"] = _row_model_key(row)
            occurrence = {
                "face_id": face_id,
                "box": {key: _clamp01(float(row[key])) for key in ("x1", "y1", "x2", "y2")},
                "detection_confidence": row["confidence"],
                "quality": row["quality"],
                "identity": _identity(face_id, rejected, excluded, face_to_people, people_by_id, group_by_face),
            }
            photo["occurrences"].append(occurrence)
        for photo in by_photo.values():
            photo["occurrences"].sort(key=lambda o: (o["box"]["y1"], o["box"]["x1"], o["face_id"]))
        return cls(by_photo)

    def for_photo(self, photo_id: str) -> dict[str, Any]:
        value = self.by_photo.get(photo_id)
        if value is None:
            # Older face indexes used the full SHA-256 as image_id while the catalog stores
            # photo_id_for_hash(file_hash). Cache a bridge map so export-sidecars stays O(n).
            bridge = getattr(self, "_legacy_bridge", None)
            if bridge is None:
                bridge = {photo_id_for_hash(image_id): faces for image_id, faces in self.by_photo.items()}
                setattr(self, "_legacy_bridge", bridge)
            value = bridge.get(photo_id)
        return json.loads(json.dumps(value if value is not None else _empty_faces(), sort_keys=True))


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _row_model_key(row: sqlite3.Row) -> str:
    return ":".join(str(row[k]) for k in ("backend_id", "model_id", "model_version", "preprocessing_version", "embedding_dimensions"))


def _identity(face_id: str, rejected: set[str], excluded: set[str], face_to_people: dict[str, list[str]], people: dict[str, dict[str, Any]], groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    person_ids = sorted(set(face_to_people.get(face_id, [])))
    if len(person_ids) > 1:
        return {"status": "conflict", "person_ids": person_ids}
    if face_id in rejected:
        return {"status": "rejected"}
    if face_id in excluded:
        return {"status": "excluded"}
    if person_ids:
        person = people.get(person_ids[0], {"person_id": person_ids[0], "friendly_name": person_friendly_name(person_ids[0]), "display_name": None})
        return {"status": "confirmed", "person_id": person["person_id"], "friendly_name": person["friendly_name"], "display_name": person.get("display_name")}
    group = groups.get(face_id)
    if group:
        return {"status": "provisional", "group_id": group["group_id"], "friendly_name": friendly_group_name(group["group_id"]), "cluster_run_id": group["cluster_run_id"], "group_review_state": group["review_state"], "group_conflict": bool(group["conflict"])}
    return {"status": "ungrouped"}


def sidecar_document(photo, embedding_model: str | None = None, tags: list | None = None, tag_proposals: list | None = None, image_analysis=None, proposal_evidence: dict | None = None, faces: dict[str, Any] | None = None) -> dict:
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
        "faces": faces if faces is not None else _empty_faces(),
    }


def export_sidecars(catalog: Catalog, export_dir: Path | None = None, workspace: Path | None = None) -> int:
    count = 0
    if export_dir: export_dir.mkdir(parents=True, exist_ok=True)
    face_index = FaceSidecarIndex.load(workspace or catalog.db_path.parent)
    for photo in catalog.list_photos():
        target = (export_dir / f"{photo.photo_id}.kiokufux.json") if export_dir else photo.source_path.with_name(photo.source_path.name + ".kiokufux.json")
        target.write_text(json.dumps(sidecar_document(photo, tags=catalog.list_tags(photo.photo_id), tag_proposals=catalog.list_tag_proposals(photo.photo_id), image_analysis=catalog.get_image_analysis(photo.photo_id), proposal_evidence=catalog.tag_proposal_evidence(photo.photo_id), faces=face_index.for_photo(photo.photo_id)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        count += 1
    return count
