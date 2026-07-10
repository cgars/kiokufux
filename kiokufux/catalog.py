from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import Embedding, Photo, PhotoTag, TagProposal, TagProposalSummary, TagVocabularyEntry
from .vlm import ImageAnalysis


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Catalog:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Catalog":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS photos (
              photo_id TEXT PRIMARY KEY, source_path TEXT NOT NULL, relative_path TEXT NOT NULL,
              file_hash TEXT NOT NULL, file_size INTEGER, mime_type TEXT, width INTEGER, height INTEGER,
              created_at_file TEXT, modified_at_file TEXT, exif_datetime_original TEXT,
              exif_gps_lat REAL, exif_gps_lon REAL, thumbnail_path TEXT,
              embedding_status TEXT NOT NULL DEFAULT 'pending', indexed_at TEXT, updated_at TEXT,
              missing INTEGER NOT NULL DEFAULT 0, error TEXT
            );
            CREATE TABLE IF NOT EXISTS embeddings (
              photo_id TEXT NOT NULL, model_name TEXT NOT NULL, model_version TEXT NOT NULL,
              vector_dimension INTEGER NOT NULL, embedding_path TEXT NOT NULL, created_at TEXT NOT NULL,
              PRIMARY KEY (photo_id, model_name, model_version),
              FOREIGN KEY (photo_id) REFERENCES photos(photo_id)
            );
            CREATE INDEX IF NOT EXISTS idx_photos_photo_id ON photos(photo_id);
            CREATE INDEX IF NOT EXISTS idx_photos_relative_path ON photos(relative_path);
            CREATE INDEX IF NOT EXISTS idx_photos_file_hash ON photos(file_hash);
            CREATE INDEX IF NOT EXISTS idx_photos_embedding_status ON photos(embedding_status);
            CREATE TABLE IF NOT EXISTS photo_tags (
              photo_id TEXT NOT NULL, tag TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'manual',
              created_at TEXT NOT NULL,
              PRIMARY KEY (photo_id, tag, source),
              FOREIGN KEY (photo_id) REFERENCES photos(photo_id)
            );
            CREATE INDEX IF NOT EXISTS idx_photo_tags_photo_id ON photo_tags(photo_id);
            CREATE INDEX IF NOT EXISTS idx_photo_tags_tag ON photo_tags(tag);
            CREATE TABLE IF NOT EXISTS tag_proposals (
              photo_id TEXT NOT NULL, tag TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'ai-zero-shot',
              confidence REAL NOT NULL, status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL,
              PRIMARY KEY (photo_id, tag, source),
              FOREIGN KEY (photo_id) REFERENCES photos(photo_id)
            );
            CREATE INDEX IF NOT EXISTS idx_tag_proposals_photo_id ON tag_proposals(photo_id);
            CREATE INDEX IF NOT EXISTS idx_tag_proposals_status ON tag_proposals(status);
            CREATE TABLE IF NOT EXISTS tag_proposal_evidence (
              photo_id TEXT NOT NULL, tag TEXT NOT NULL, source TEXT NOT NULL,
              category_hint TEXT, evidence TEXT, created_at TEXT NOT NULL,
              PRIMARY KEY (photo_id, tag, source),
              FOREIGN KEY (photo_id) REFERENCES photos(photo_id)
            );
            CREATE TABLE IF NOT EXISTS image_analyses (
              photo_id TEXT NOT NULL, source TEXT NOT NULL, model_name TEXT NOT NULL, model_version TEXT NOT NULL,
              caption TEXT, scene TEXT, activity TEXT, objects_json TEXT NOT NULL DEFAULT '[]',
              aesthetic_json TEXT NOT NULL DEFAULT '[]', warnings_json TEXT NOT NULL DEFAULT '[]',
              raw_response_json TEXT, created_at TEXT NOT NULL,
              PRIMARY KEY (photo_id, source, model_name, model_version),
              FOREIGN KEY (photo_id) REFERENCES photos(photo_id)
            );
            CREATE INDEX IF NOT EXISTS idx_image_analyses_photo_id ON image_analyses(photo_id);
            CREATE TABLE IF NOT EXISTS tag_vocabulary (
              tag TEXT PRIMARY KEY, category TEXT NOT NULL DEFAULT 'uncategorized',
              scope TEXT NOT NULL DEFAULT 'optional', status TEXT NOT NULL DEFAULT 'proposed',
              parent TEXT, notes TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tag_vocabulary_status ON tag_vocabulary(status);
            CREATE INDEX IF NOT EXISTS idx_tag_vocabulary_category ON tag_vocabulary(category);
            CREATE TABLE IF NOT EXISTS tag_aliases (
              alias TEXT PRIMARY KEY, tag TEXT NOT NULL, created_at TEXT NOT NULL,
              FOREIGN KEY (tag) REFERENCES tag_vocabulary(tag)
            );
            CREATE INDEX IF NOT EXISTS idx_tag_aliases_tag ON tag_aliases(tag);
            """
        )
        self.conn.commit()

    def upsert_image_analysis(self, analysis: ImageAnalysis) -> None:
        created_at = analysis.created_at or now_iso()
        self.conn.execute(
            """
            INSERT INTO image_analyses VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(photo_id, source, model_name, model_version) DO UPDATE SET
              caption=excluded.caption, scene=excluded.scene, activity=excluded.activity,
              objects_json=excluded.objects_json, aesthetic_json=excluded.aesthetic_json,
              warnings_json=excluded.warnings_json, raw_response_json=excluded.raw_response_json,
              created_at=excluded.created_at
            """,
            (
                analysis.photo_id, analysis.source, analysis.model_name, analysis.model_version,
                analysis.caption, analysis.scene, analysis.activity,
                json.dumps(analysis.objects, sort_keys=True),
                json.dumps(analysis.aesthetic, sort_keys=True),
                json.dumps(analysis.warnings, sort_keys=True),
                json.dumps(analysis.raw_response, sort_keys=True) if analysis.raw_response is not None else None,
                created_at,
            ),
        )
        for tag in analysis.candidate_tags:
            self.propose_tag(
                analysis.photo_id, tag.tag, tag.confidence, source=analysis.source,
                category_hint=tag.category_hint, evidence=tag.evidence, commit=False,
            )
        for tag in analysis.uncertain_tags:
            self.propose_tag(
                analysis.photo_id, tag.tag, tag.confidence, source=analysis.source,
                category_hint=tag.category_hint, evidence=tag.evidence, commit=False,
            )
        self.conn.commit()

    def get_image_analysis(self, photo_id: str) -> ImageAnalysis | None:
        row = self.conn.execute(
            "SELECT * FROM image_analyses WHERE photo_id=? ORDER BY created_at DESC LIMIT 1",
            (photo_id,),
        ).fetchone()
        if row is None:
            return None
        return ImageAnalysis(
            photo_id=str(row["photo_id"]), source=str(row["source"]),
            model_name=str(row["model_name"]), model_version=str(row["model_version"]),
            caption=row["caption"], scene=row["scene"], activity=row["activity"],
            objects=json.loads(row["objects_json"]), aesthetic=json.loads(row["aesthetic_json"]),
            warnings=json.loads(row["warnings_json"]),
            raw_response=json.loads(row["raw_response_json"]) if row["raw_response_json"] else None,
            created_at=str(row["created_at"]),
        )

    def tag_proposal_evidence(self, photo_id: str) -> dict[tuple[str, str], dict[str, str | None]]:
        rows = self.conn.execute(
            "SELECT tag, source, category_hint, evidence FROM tag_proposal_evidence WHERE photo_id=?",
            (photo_id,),
        ).fetchall()
        return {
            (str(row["tag"]), str(row["source"])): {"category_hint": row["category_hint"], "evidence": row["evidence"]}
            for row in rows
        }

    def upsert_photo(self, photo: Photo) -> None:
        existing = self.get_photo(photo.photo_id)
        indexed_at = existing.indexed_at if existing else now_iso()
        updated_at = now_iso()
        self.conn.execute(
            """
            INSERT INTO photos VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(photo_id) DO UPDATE SET
              source_path=excluded.source_path, relative_path=excluded.relative_path, file_hash=excluded.file_hash,
              file_size=excluded.file_size, mime_type=excluded.mime_type, width=excluded.width, height=excluded.height,
              created_at_file=excluded.created_at_file, modified_at_file=excluded.modified_at_file,
              exif_datetime_original=excluded.exif_datetime_original, exif_gps_lat=excluded.exif_gps_lat,
              exif_gps_lon=excluded.exif_gps_lon, updated_at=excluded.updated_at, missing=0, error=excluded.error
            """,
            (photo.photo_id, str(photo.source_path), photo.relative_path, photo.file_hash, photo.file_size,
             photo.mime_type, photo.width, photo.height, photo.created_at_file, photo.modified_at_file,
             photo.exif_datetime_original, photo.exif_gps_lat, photo.exif_gps_lon, photo.thumbnail_path,
             photo.embedding_status, indexed_at, updated_at, int(photo.missing), photo.error),
        )
        self.conn.commit()

    def record_error(self, photo_id: str, source_path: Path, relative_path: str, file_hash: str, error: str) -> None:
        self.upsert_photo(Photo(photo_id, source_path, relative_path, file_hash, error=error))

    def get_photo(self, photo_id: str) -> Photo | None:
        row = self.conn.execute("SELECT * FROM photos WHERE photo_id=?", (photo_id,)).fetchone()
        return self._photo(row) if row else None

    def resolve_photo_id(self, photo_id_or_prefix: str) -> str:
        if len(photo_id_or_prefix) < 7:
            raise ValueError("Photo ID prefix must be at least 7 characters")
        rows = self.conn.execute(
            "SELECT photo_id FROM photos WHERE photo_id LIKE ? ORDER BY photo_id",
            (f"{photo_id_or_prefix}%",),
        ).fetchall()
        if not rows:
            raise ValueError(f"No photo found for ID prefix: {photo_id_or_prefix}")
        if len(rows) > 1:
            matches = ", ".join(row["photo_id"][:7] for row in rows[:5])
            raise ValueError(f"Ambiguous photo ID prefix {photo_id_or_prefix!r}; matches: {matches}")
        return str(rows[0]["photo_id"])

    def list_photos(self, include_missing: bool = False) -> list[Photo]:
        sql = "SELECT * FROM photos" + ("" if include_missing else " WHERE missing=0") + " ORDER BY relative_path"
        return [self._photo(r) for r in self.conn.execute(sql)]

    def mark_missing_except(self, seen: Iterable[str]) -> None:
        seen_set = set(seen)
        rows = self.conn.execute("SELECT photo_id FROM photos WHERE missing=0").fetchall()
        for row in rows:
            if row["photo_id"] not in seen_set:
                self.conn.execute("UPDATE photos SET missing=1, updated_at=? WHERE photo_id=?", (now_iso(), row["photo_id"]))
        self.conn.commit()

    def set_thumbnail(self, photo_id: str, path: Path) -> None:
        self.conn.execute("UPDATE photos SET thumbnail_path=?, updated_at=? WHERE photo_id=?", (str(path), now_iso(), photo_id)); self.conn.commit()

    def stored_artifact_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.db_path.parent))
        except ValueError:
            return str(path)

    def artifact_path(self, stored_path: str) -> Path:
        path = Path(stored_path)
        return path if path.is_absolute() else self.db_path.parent / path

    def upsert_embedding(self, embedding: Embedding) -> None:
        self.conn.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?)", (embedding.photo_id, embedding.model_name, embedding.model_version, embedding.vector_dimension, embedding.embedding_path, embedding.created_at))
        self.conn.execute("UPDATE photos SET embedding_status='indexed', updated_at=? WHERE photo_id=?", (now_iso(), embedding.photo_id)); self.conn.commit()

    def list_embeddings(self, model_name: str, model_version: str) -> list[Embedding]:
        rows = self.conn.execute("SELECT * FROM embeddings WHERE model_name=? AND model_version=?", (model_name, model_version)).fetchall()
        return [Embedding(**dict(r)) for r in rows]



    def upsert_vocabulary_tag(
        self,
        tag: str,
        category: str = "uncategorized",
        scope: str = "optional",
        status: str = "proposed",
        parent: str | None = None,
        aliases: Iterable[str] = (),
        notes: str | None = None,
    ) -> None:
        normalized = normalize_tag(tag)
        if not normalized:
            raise ValueError("Vocabulary tag cannot be empty")
        created_at = now_iso()
        parent_tag = normalize_tag(parent) if parent else None
        self.conn.execute(
            """
            INSERT INTO tag_vocabulary VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(tag) DO UPDATE SET
              category=excluded.category, scope=excluded.scope, status=excluded.status,
              parent=excluded.parent, notes=excluded.notes, updated_at=excluded.updated_at
            """,
            (normalized, normalize_tag(category) or "uncategorized", normalize_tag(scope) or "optional",
             normalize_tag(status) or "proposed", parent_tag, notes, created_at, created_at),
        )
        for alias in aliases:
            self.add_tag_alias(alias, normalized)
        self.conn.commit()

    def add_tag_alias(self, alias: str, tag: str) -> None:
        normalized_alias = normalize_tag(alias)
        normalized_tag = normalize_tag(tag)
        if not normalized_alias or not normalized_tag:
            raise ValueError("Alias and tag cannot be empty")
        if normalized_alias == normalized_tag:
            return
        self.conn.execute(
            "INSERT OR IGNORE INTO tag_aliases VALUES (?,?,?)",
            (normalized_alias, normalized_tag, now_iso()),
        )

    def list_vocabulary(self, status: str | None = None) -> list[TagVocabularyEntry]:
        clauses: list[str] = []
        params: list[str] = []
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM tag_vocabulary{where} ORDER BY category, tag",
            params,
        ).fetchall()
        aliases = self.conn.execute("SELECT alias, tag FROM tag_aliases ORDER BY alias").fetchall()
        by_tag: dict[str, list[str]] = {}
        for row in aliases:
            by_tag.setdefault(str(row["tag"]), []).append(str(row["alias"]))
        return [
            TagVocabularyEntry(
                tag=str(row["tag"]), category=str(row["category"]), scope=str(row["scope"]),
                status=str(row["status"]), parent=row["parent"], aliases=by_tag.get(str(row["tag"]), []),
                notes=row["notes"], created_at=str(row["created_at"]), updated_at=str(row["updated_at"]),
            )
            for row in rows
        ]

    def get_vocabulary_tag(self, tag: str) -> TagVocabularyEntry | None:
        normalized = self.canonical_tag(tag)
        return next((entry for entry in self.list_vocabulary(status=None) if entry.tag == normalized), None)

    def canonical_tag(self, tag: str) -> str:
        normalized = normalize_tag(tag)
        row = self.conn.execute("SELECT tag FROM tag_aliases WHERE alias=?", (normalized,)).fetchone()
        return str(row["tag"]) if row else normalized

    def propose_vocabulary_from_tag_proposals(self, min_photos: int = 1, status: str = "pending") -> int:
        created = 0
        for summary in self.summarize_tag_proposals(status=status):
            if summary.photo_count < min_photos:
                continue
            canonical = self.canonical_tag(summary.tag)
            existing = self.conn.execute("SELECT status FROM tag_vocabulary WHERE tag=?", (canonical,)).fetchone()
            if existing is None:
                self.upsert_vocabulary_tag(
                    canonical,
                    status="proposed",
                    notes=f"Proposed from {summary.photo_count} photos and {summary.proposal_count} tag proposals.",
                )
                created += 1
        return created

    def merge_vocabulary_tag(self, alias: str, canonical: str) -> None:
        normalized_alias = normalize_tag(alias)
        normalized_canonical = normalize_tag(canonical)
        if not normalized_alias or not normalized_canonical:
            raise ValueError("Alias and canonical tag cannot be empty")
        existing = self.conn.execute("SELECT tag FROM tag_vocabulary WHERE tag=?", (normalized_canonical,)).fetchone()
        if existing is None:
            self.upsert_vocabulary_tag(normalized_canonical, status="accepted")
        self.add_tag_alias(normalized_alias, normalized_canonical)
        if normalized_alias != normalized_canonical:
            self.conn.execute(
                "UPDATE tag_vocabulary SET status='rejected', notes=?, updated_at=? WHERE tag=?",
                (f"Merged into {normalized_canonical}.", now_iso(), normalized_alias),
            )
        self.conn.commit()

    def apply_vocabulary_to_tag_proposals(self, source: str = "ai-zero-shot") -> int:
        proposals = self.list_tag_proposals(status="pending")
        changed = 0
        for proposal in proposals:
            if proposal.source != source:
                continue
            canonical = self.canonical_tag(proposal.tag)
            row = self.conn.execute("SELECT status FROM tag_vocabulary WHERE tag=?", (canonical,)).fetchone()
            if row is None:
                continue
            vocab_status = str(row["status"])
            if vocab_status == "accepted":
                self.add_tag(proposal.photo_id, canonical, source="auto")
                self.set_tag_proposal_status(proposal.photo_id, proposal.tag, "accepted", source=proposal.source)
                changed += 1
            elif vocab_status == "rejected":
                self.set_tag_proposal_status(proposal.photo_id, proposal.tag, "rejected", source=proposal.source)
                changed += 1
        return changed

    def add_tag(self, photo_id: str, tag: str, source: str = "manual") -> None:
        normalized = normalize_tag(tag)
        if not normalized:
            raise ValueError("Tag cannot be empty")
        self.conn.execute(
            "INSERT OR IGNORE INTO photo_tags VALUES (?,?,?,?)",
            (photo_id, normalized, source, now_iso()),
        )
        self.conn.commit()

    def remove_tag(self, photo_id: str, tag: str, source: str | None = None) -> None:
        normalized = normalize_tag(tag)
        if source is None:
            self.conn.execute("DELETE FROM photo_tags WHERE photo_id=? AND tag=?", (photo_id, normalized))
        else:
            self.conn.execute("DELETE FROM photo_tags WHERE photo_id=? AND tag=? AND source=?", (photo_id, normalized, source))
        self.conn.commit()

    def list_tags(self, photo_id: str) -> list[PhotoTag]:
        rows = self.conn.execute(
            "SELECT * FROM photo_tags WHERE photo_id=? ORDER BY tag, source",
            (photo_id,),
        ).fetchall()
        return [PhotoTag(**dict(row)) for row in rows]

    def list_all_tags(self) -> list[PhotoTag]:
        rows = self.conn.execute("SELECT * FROM photo_tags ORDER BY tag, photo_id, source").fetchall()
        return [PhotoTag(**dict(row)) for row in rows]


    def propose_tag(
        self,
        photo_id: str,
        tag: str,
        confidence: float,
        source: str = "ai-zero-shot",
        category_hint: str | None = None,
        evidence: str | None = None,
        commit: bool = True,
    ) -> None:
        normalized = normalize_tag(tag)
        if not normalized:
            raise ValueError("Tag proposal cannot be empty")
        created_at = now_iso()
        self.conn.execute(
            """
            INSERT INTO tag_proposals VALUES (?,?,?,?,?,?)
            ON CONFLICT(photo_id, tag, source) DO UPDATE SET
              confidence=excluded.confidence, status=excluded.status, created_at=excluded.created_at
            """,
            (photo_id, normalized, source, float(confidence), "pending", created_at),
        )
        if category_hint is not None or evidence is not None:
            self.conn.execute(
                """
                INSERT INTO tag_proposal_evidence VALUES (?,?,?,?,?,?)
                ON CONFLICT(photo_id, tag, source) DO UPDATE SET
                  category_hint=excluded.category_hint, evidence=excluded.evidence, created_at=excluded.created_at
                """,
                (photo_id, normalized, source, normalize_tag(category_hint) if category_hint else None, evidence, created_at),
            )
        if commit:
            self.conn.commit()

    def list_tag_proposals(self, photo_id: str | None = None, status: str | None = None) -> list[TagProposal]:
        clauses: list[str] = []
        params: list[str] = []
        if photo_id is not None:
            clauses.append("photo_id=?")
            params.append(photo_id)
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM tag_proposals{where} ORDER BY photo_id, confidence DESC, tag",
            params,
        ).fetchall()
        return [TagProposal(**dict(row)) for row in rows]


    def summarize_tag_proposals(self, status: str | None = "pending") -> list[TagProposalSummary]:
        clauses: list[str] = []
        params: list[str] = []
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.conn.execute(
            f"""
            SELECT tag, source, status, COUNT(*) AS proposal_count, COUNT(DISTINCT photo_id) AS photo_count,
                   AVG(confidence) AS avg_confidence, MAX(confidence) AS max_confidence
            FROM tag_proposals{where}
            GROUP BY tag, source, status
            ORDER BY photo_count DESC, avg_confidence DESC, tag
            """,
            params,
        ).fetchall()
        return [
            TagProposalSummary(
                tag=str(row["tag"]),
                source=str(row["source"]),
                status=str(row["status"]),
                proposal_count=int(row["proposal_count"]),
                photo_count=int(row["photo_count"]),
                avg_confidence=float(row["avg_confidence"]),
                max_confidence=float(row["max_confidence"]),
            )
            for row in rows
        ]

    def set_tag_proposal_status(self, photo_id: str, tag: str, status: str, source: str = "ai-zero-shot") -> None:
        normalized = normalize_tag(tag)
        self.conn.execute(
            "UPDATE tag_proposals SET status=? WHERE photo_id=? AND tag=? AND source=?",
            (status, photo_id, normalized, source),
        )
        self.conn.commit()

    def accept_tag_proposal(self, photo_id: str, tag: str, source: str = "ai-zero-shot") -> None:
        normalized = normalize_tag(tag)
        self.add_tag(photo_id, normalized, source="auto")
        self.set_tag_proposal_status(photo_id, normalized, "accepted", source=source)

    def accept_tag_proposals(self, photo_id: str | None = None, source: str = "ai-zero-shot") -> int:
        proposals = self.list_tag_proposals(photo_id, status="pending")
        accepted = 0
        for proposal in proposals:
            if proposal.source != source:
                continue
            self.add_tag(proposal.photo_id, proposal.tag, source="auto")
            self.set_tag_proposal_status(proposal.photo_id, proposal.tag, "accepted", source=source)
            accepted += 1
        return accepted

    def reject_tag_proposal(self, photo_id: str, tag: str, source: str = "ai-zero-shot") -> None:
        self.set_tag_proposal_status(photo_id, tag, "rejected", source=source)

    def _photo(self, row: sqlite3.Row) -> Photo:
        d = dict(row); d["source_path"] = Path(d["source_path"]); d["missing"] = bool(d["missing"]); return Photo(**d)


def normalize_tag(tag: str) -> str:
    return " ".join(tag.strip().lower().split())
