"""Local, collection-scoped face discovery and durable human review state.

Model objects are deliberately contained behind :class:`FaceBackend`; importing this
module never imports torch or downloads weights.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image, ImageOps

from .config import FacesConfig, SUPPORTED_EXTENSIONS, WORKSPACE_NAME
from .hashing import photo_id_for_hash

_GROUP_ADJECTIVES = ("amber", "brave", "bright", "calm", "clever", "dancing", "gentle", "golden",
                     "happy", "hidden", "kind", "lively", "lucky", "misty", "quiet", "running")
_GROUP_ANIMALS = ("badger", "bear", "deer", "dolphin", "eagle", "falcon", "fox", "hare",
                  "heron", "lynx", "otter", "owl", "panda", "raven", "tiger", "wolf")


def _friendly_name_from_seed(seed: str) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return f"{_GROUP_ADJECTIVES[digest[0] % len(_GROUP_ADJECTIVES)]}_{_GROUP_ANIMALS[digest[1] % len(_GROUP_ANIMALS)]}"


def friendly_group_name(group_id: str) -> str:
    """Return a deterministic, non-identifying label for a provisional group."""
    return _friendly_name_from_seed(group_id)


def person_friendly_name(person_id: str) -> str:
    """Return a deterministic, non-identifying migration label for a stable person."""
    return _friendly_name_from_seed(person_id)


@dataclass(frozen=True, slots=True)
class FaceDetection:
    box: tuple[float, float, float, float]  # pixels in the supplied, oriented image
    confidence: float
    aligned_face: Image.Image
    landmarks: tuple[tuple[float, float], ...] = ()
    quality: float | None = None


class FaceBackend(Protocol):
    backend_id: str
    model_id: str
    model_version: str
    preprocessing_version: str
    embedding_dimensions: int

    def detect(self, image: Image.Image) -> list[FaceDetection]: ...
    def embed(self, aligned_faces: Sequence[Image.Image]) -> np.ndarray: ...


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-length face embedding cannot be normalized")
    return values / norms


def embedding_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(normalize_embeddings(left)[0] - normalize_embeddings(right)[0]))


def model_key(backend: FaceBackend) -> str:
    return ":".join((backend.backend_id, backend.model_id, backend.model_version,
                     backend.preprocessing_version, str(backend.embedding_dimensions)))


def boxes_iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


class FaceStore:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.path = workspace / "faces.sqlite"
        workspace.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.init_schema()

    def close(self) -> None: self.db.close()
    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    def init_schema(self) -> None:
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS face_occurrences(
          face_id TEXT PRIMARY KEY, image_id TEXT NOT NULL, image_path TEXT NOT NULL,
          x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
          landmarks TEXT NOT NULL DEFAULT '[]', confidence REAL NOT NULL, quality REAL,
          backend_id TEXT NOT NULL, model_id TEXT NOT NULL, model_version TEXT NOT NULL,
          preprocessing_version TEXT NOT NULL, embedding_dimensions INTEGER NOT NULL,
          l2_normalized INTEGER NOT NULL, embedding BLOB NOT NULL,
          content_fingerprint TEXT NOT NULL, scanned_at TEXT NOT NULL, excluded INTEGER NOT NULL DEFAULT 0);
        CREATE INDEX IF NOT EXISTS face_image_idx ON face_occurrences(image_id);
        CREATE TABLE IF NOT EXISTS scanned_images(image_id TEXT PRIMARY KEY, image_path TEXT NOT NULL,
          content_fingerprint TEXT NOT NULL, model_key TEXT NOT NULL, scanned_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS cluster_runs(cluster_run_id TEXT PRIMARY KEY, model_key TEXT NOT NULL,
          parameters TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS face_groups(group_id TEXT PRIMARY KEY, cluster_run_id TEXT NOT NULL REFERENCES cluster_runs(cluster_run_id) ON DELETE CASCADE,
          representative_face_id TEXT NOT NULL REFERENCES face_occurrences(face_id), review_state TEXT NOT NULL DEFAULT 'unreviewed', conflict INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS face_group_members(group_id TEXT NOT NULL REFERENCES face_groups(group_id) ON DELETE CASCADE, face_id TEXT NOT NULL REFERENCES face_occurrences(face_id) ON DELETE CASCADE,
          membership_score REAL, PRIMARY KEY(group_id,face_id), UNIQUE(face_id));
        """)
        self.db.commit()

    def groups(self) -> list[dict[str, Any]]:
        rows = self.db.execute("""SELECT g.*, COUNT(m.face_id) face_count,
          COUNT(DISTINCT f.image_id) photo_count FROM face_groups g
          JOIN face_group_members m USING(group_id) JOIN face_occurrences f USING(face_id)
          WHERE NOT EXISTS (SELECT 1 FROM face_group_members mx WHERE mx.group_id=g.group_id AND mx.face_id IN (SELECT value FROM json_each(?)))
          GROUP BY g.group_id ORDER BY face_count DESC""", (json.dumps(self._confirmed_face_ids()),)).fetchall()
        return [{**dict(row), "friendly_id": friendly_group_name(row["group_id"]), "friendly_name": friendly_group_name(row["group_id"])} for row in rows]

    def group(self, group_id: str) -> dict[str, Any] | None:
        group = self.db.execute("SELECT * FROM face_groups WHERE group_id=?", (group_id,)).fetchone()
        if group is None:
            return None
        faces = self.db.execute("""SELECT f.face_id,f.image_id,f.confidence,f.quality,
          f.x1,f.y1,f.x2,f.y2 FROM face_group_members m
          JOIN face_occurrences f USING(face_id) WHERE m.group_id=? ORDER BY f.face_id""", (group_id,)).fetchall()
        result = dict(group)
        result["friendly_id"] = friendly_group_name(group_id)
        result["friendly_name"] = friendly_group_name(group_id)
        result["faces"] = [dict(face) for face in faces]
        return result

    def ungrouped(self) -> list[dict[str, Any]]:
        rows = self.db.execute("""SELECT f.face_id,f.image_id,f.confidence,f.quality
          FROM face_occurrences f LEFT JOIN face_group_members m USING(face_id)
          WHERE m.face_id IS NULL AND f.excluded=0 ORDER BY f.face_id""").fetchall()
        return [dict(row) for row in rows]


    def _confirmed_face_ids(self) -> list[str]:
        data = self.workspace / "face-review.json"
        if not data.exists():
            return []
        try:
            review = json.loads(data.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return sorted({face_id for face_ids in (review.get("person_faces") or {}).values() for face_id in face_ids})


def _fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def scan_faces(root: Path, store: FaceStore, backend: FaceBackend, *, working_resolution: int = 1600,
               confidence_threshold: float = .95, minimum_face_size: int = 40,
               thumbnail_size: int = FacesConfig().thumbnail_size) -> dict[str, int]:
    stats = {"scanned": 0, "skipped": 0, "failed": 0, "detected": 0, "embedded": 0, "rejected": 0}
    cache = store.workspace / "cache" / "face-thumbnails"; cache.mkdir(parents=True, exist_ok=True)
    key = model_key(backend)
    for path in (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and WORKSPACE_NAME not in p.parts):
        try:
            fingerprint = _fingerprint(path); image_id = photo_id_for_hash(fingerprint)
            old = store.db.execute("SELECT * FROM scanned_images WHERE image_id=?", (image_id,)).fetchone()
            if old and old["content_fingerprint"] == fingerprint and old["model_key"] == key:
                stats["skipped"] += 1; continue
            with Image.open(path) as source: oriented = ImageOps.exif_transpose(source).convert("RGB")
            ow, oh = oriented.size; working = oriented.copy(); working.thumbnail((working_resolution, working_resolution))
            sx, sy = ow / working.width, oh / working.height
            detections = backend.detect(working)
            usable = [d for d in detections if d.confidence >= confidence_threshold and min(d.box[2]-d.box[0], d.box[3]-d.box[1]) >= minimum_face_size / max(sx, sy)]
            stats["detected"] += len(detections); stats["rejected"] += len(detections)-len(usable)
            vectors = normalize_embeddings(backend.embed([d.aligned_face for d in usable])) if usable else np.empty((0, backend.embedding_dimensions), np.float32)
            if vectors.shape != (len(usable), backend.embedding_dimensions): raise ValueError("backend returned incompatible embedding shape")
            previous = store.db.execute("SELECT face_id,x1,y1,x2,y2 FROM face_occurrences WHERE image_id=?", (image_id,)).fetchall()
            used: set[str] = set(); now = datetime.now(timezone.utc).isoformat()
            store.db.execute("DELETE FROM face_occurrences WHERE image_id=?", (image_id,))
            for detection, vector in zip(usable, vectors):
                box = (detection.box[0]*sx/ow, detection.box[1]*sy/oh, detection.box[2]*sx/ow, detection.box[3]*sy/oh)
                matches = [(boxes_iou(box, (r['x1'],r['y1'],r['x2'],r['y2'])), r['face_id']) for r in previous if r['face_id'] not in used]
                best = max(matches, default=(0, "")); face_id = best[1] if best[0] >= .65 else str(uuid.uuid4()); used.add(face_id)
                landmarks = [[x*sx/ow, y*sy/oh] for x,y in detection.landmarks]
                store.db.execute("""INSERT INTO face_occurrences
                    (face_id,image_id,image_path,x1,y1,x2,y2,landmarks,confidence,quality,
                     backend_id,model_id,model_version,preprocessing_version,embedding_dimensions,
                     l2_normalized,embedding,content_fingerprint,scanned_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (face_id,image_id,str(path.resolve().relative_to(root.resolve())),*box,json.dumps(landmarks),detection.confidence,detection.quality,
                     backend.backend_id,backend.model_id,backend.model_version,backend.preprocessing_version,
                     backend.embedding_dimensions,1,vector.astype('<f4').tobytes(),fingerprint,now))
                crop = oriented.crop((box[0]*ow,box[1]*oh,box[2]*ow,box[3]*oh)); crop.thumbnail((thumbnail_size,thumbnail_size)); crop.save(cache/f"{face_id}.jpg", "JPEG")
            store.db.execute("INSERT OR REPLACE INTO scanned_images VALUES(?,?,?,?,?)", (image_id,str(path.resolve().relative_to(root.resolve())),fingerprint,key,now))
            store.db.commit(); stats["scanned"] += 1; stats["embedded"] += len(usable)
        except Exception:
            store.db.rollback(); stats["failed"] += 1
    return stats


def cluster_faces(store: FaceStore, *, min_cluster_size: int = 2, min_samples: int = 2) -> dict[str, int]:
    review = ReviewState.load_existing(store.workspace)
    blocked = set(review.get("rejected_face_ids", [])) | set(review.get("excluded_face_ids", [])) | {face_id for face_ids in (review.get("person_faces") or {}).values() for face_id in face_ids}
    rows = store.db.execute("SELECT * FROM face_occurrences WHERE excluded=0 ORDER BY face_id").fetchall()
    rows = [row for row in rows if row["face_id"] not in blocked]
    by_key: dict[tuple[Any,...], list[Any]] = {}
    for row in rows:
        key = tuple(row[k] for k in ("backend_id","model_id","model_version","preprocessing_version","embedding_dimensions"))
        by_key.setdefault(key, []).append(row)
    store.db.execute("DELETE FROM face_group_members"); store.db.execute("DELETE FROM face_groups")
    groups = ungrouped = 0
    for key, compatible in by_key.items():
        vectors = np.stack([np.frombuffer(r["embedding"], dtype="<f4") for r in compatible])
        try:
            try:
                from hdbscan import HDBSCAN  # type: ignore
            except ImportError:
                from sklearn.cluster import HDBSCAN  # type: ignore[attr-defined]
            labels = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
            ).fit_predict(vectors)
        except (ImportError, AttributeError):
            # Conservative dependency-light fallback: connected components at a tight L2 threshold.
            labels = np.full(len(vectors), -1); next_label = 0
            for i in range(len(vectors)):
                peers = np.flatnonzero(np.linalg.norm(vectors-vectors[i], axis=1) <= .65)
                if len(peers) >= min_cluster_size:
                    existing = labels[peers][labels[peers]>=0]; label = int(existing[0]) if len(existing) else next_label
                    labels[peers] = label
                    if not len(existing): next_label += 1
        run_id = str(uuid.uuid4()); params = {"min_cluster_size":min_cluster_size,"min_samples":min_samples,"metric":"euclidean"}
        store.db.execute("INSERT INTO cluster_runs VALUES(?,?,?,?)",(run_id,":".join(map(str,key)),json.dumps(params),datetime.now(timezone.utc).isoformat()))
        for label in sorted(set(labels)-{-1}):
            indices = np.flatnonzero(labels==label); centroid = vectors[indices].mean(axis=0)
            rep = int(indices[np.argmin(np.linalg.norm(vectors[indices]-centroid,axis=1))]); group_id = str(uuid.uuid4())
            image_ids=[compatible[i]["image_id"] for i in indices]; conflict=len(image_ids)!=len(set(image_ids))
            store.db.execute("INSERT INTO face_groups VALUES(?,?,?,?,?)",(group_id,run_id,compatible[rep]["face_id"],"unreviewed",int(conflict)))
            for i in indices: store.db.execute("INSERT INTO face_group_members VALUES(?,?,?)",(group_id,compatible[i]["face_id"],None))
            groups += 1
        ungrouped += int(np.sum(labels == -1))
    store.db.commit(); return {"groups":groups,"ungrouped":ungrouped}


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


class ReviewState:
    @staticmethod
    def load_existing(workspace: Path) -> dict[str, Any]:
        path = workspace / "face-review.json"
        if not path.exists():
            return {"schema_version": 2, "actions": [], "rejected_face_ids": [], "excluded_face_ids": [], "person_faces": {}, "face_refs": {}}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"schema_version": 2, "actions": [], "rejected_face_ids": [], "excluded_face_ids": [], "person_faces": {}, "face_refs": {}}

    def __init__(self, workspace: Path):
        self.workspace=workspace; self.review_path=workspace/"face-review.json"; self.people_path=workspace/"people.json"
        self.review=self._load(self.review_path,{"schema_version":2,"collection_id":str(uuid.uuid4()),"actions":[],"rejected_face_ids":[],"excluded_face_ids":[],"person_faces":{},"face_refs":{}})
        self.people=self._load(self.people_path,{"schema_version":2,"people":[]})
        self._migrate()
        self.save()
    @staticmethod
    def _load(path: Path, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default
    def _migrate(self) -> None:
        self.review.setdefault("collection_id", str(uuid.uuid4()))
        self.review.setdefault("actions", [])
        self.review.setdefault("rejected_face_ids", [])
        self.review.setdefault("excluded_face_ids", [])
        self.review.setdefault("person_faces", {})
        self.review.setdefault("face_refs", {})
        self.review["schema_version"] = 2
        self.people.setdefault("people", [])
        for person in self.people["people"]:
            person.setdefault("friendly_name", person_friendly_name(person.get("person_id", "")))
            person.setdefault("display_name", None)
        self.people["schema_version"] = 2

    def save(self): atomic_write_json(self.review_path,self.review); atomic_write_json(self.people_path,self.people)

    def create_person(self, face_ids: Sequence[str], display_name: str|None=None, friendly_name: str|None=None)->dict[str,Any]:
        unique = list(dict.fromkeys(face_ids))
        existing = {face_id: person_id for person_id, ids in self.review["person_faces"].items() for face_id in ids}
        conflicts = sorted({existing[face_id] for face_id in unique if face_id in existing})
        if conflicts:
            if len(conflicts) == 1 and set(unique) <= set(self.review["person_faces"].get(conflicts[0], [])):
                return next(p for p in self.people["people"] if p["person_id"] == conflicts[0])
            raise ValueError("face already belongs to a confirmed person")
        person_id = str(uuid.uuid4())
        person={"person_id":person_id,"friendly_name":friendly_name or person_friendly_name(person_id),"display_name":display_name or None}; self.people["people"].append(person)
        self.review["person_faces"][person["person_id"]]=unique; self.save(); return person
    def rename_person(self, person_id:str, display_name:str|None)->dict[str,Any]:
        person=next((p for p in self.people["people"] if p["person_id"]==person_id),None)
        if not person: raise KeyError(person_id)
        person.setdefault("friendly_name", person_friendly_name(person_id))
        person["display_name"]=display_name or None; self.save(); return person
    def record_action(self, action: str, face_ids: Sequence[str], **details: Any) -> dict[str, Any]:
        entry={"action":action,"face_ids":list(dict.fromkeys(face_ids)),"details":details,
               "created_at":datetime.now(timezone.utc).isoformat()}
        self.review["actions"].append(entry)
        if action == "reject-face": self.review["rejected_face_ids"] = sorted(set(self.review["rejected_face_ids"]) | set(face_ids))
        if action == "exclude-from-clustering": self.review["excluded_face_ids"] = sorted(set(self.review["excluded_face_ids"]) | set(face_ids))
        self.save(); return self.review
    def undo(self) -> dict[str, Any]:
        if not self.review["actions"]: return self.review
        removed=self.review["actions"].pop(); faces=set(removed["face_ids"])
        if removed["action"] == "reject-face": self.review["rejected_face_ids"]=[x for x in self.review["rejected_face_ids"] if x not in faces]
        if removed["action"] == "exclude-from-clustering": self.review["excluded_face_ids"]=[x for x in self.review["excluded_face_ids"] if x not in faces]
        self.save(); return self.review


def reset_derived(workspace: Path) -> None:
    (workspace/"faces.sqlite").unlink(missing_ok=True); shutil.rmtree(workspace/"cache"/"face-thumbnails",ignore_errors=True)


def remove_all(workspace: Path) -> None:
    reset_derived(workspace); (workspace/"face-review.json").unlink(missing_ok=True); (workspace/"people.json").unlink(missing_ok=True)


class FacenetBackend:
    backend_id="facenet-pytorch"; model_id="inception-resnet-v1-vggface2"; model_version="facenet-pytorch"; preprocessing_version="mtcnn-160-v1"; embedding_dimensions=512
    def __init__(self, device: str="auto"):
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1, MTCNN
        except ImportError as exc: raise RuntimeError("install kiokufux[faces] to use face inference") from exc
        self.device=("cuda" if torch.cuda.is_available() else "cpu") if device=="auto" else device
        if self.device=="cuda" and not torch.cuda.is_available(): self.device="cpu"
        self._mtcnn=MTCNN(image_size=160,margin=0,keep_all=True,device=self.device)
        self._resnet=InceptionResnetV1(pretrained="vggface2").eval().to(self.device); self._torch=torch
    def detect(self,image:Image.Image)->list[FaceDetection]:
        boxes,probs,landmarks=self._mtcnn.detect(image,landmarks=True); result=[]
        if boxes is None:return result
        for box,prob,marks in zip(boxes,probs,landmarks):
            crop=image.crop(tuple(box)); crop=crop.resize((160,160)); result.append(FaceDetection(tuple(map(float,box)),float(prob),crop,tuple(map(tuple,marks))))
        return result
    def embed(self,faces:Sequence[Image.Image])->np.ndarray:
        if not faces:return np.empty((0,512),np.float32)
        arrays=[]
        for face in faces:
            a=np.asarray(face.resize((160,160)),dtype=np.float32); arrays.append((a/127.5-1).transpose(2,0,1))
        with self._torch.no_grad(): return self._resnet(self._torch.tensor(np.stack(arrays),device=self.device)).cpu().numpy()
