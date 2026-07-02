from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import Embedding, Photo, PhotoTag, TagProposal


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
              photo_id TEXT NOT NULL, tag TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'local-ai',
              confidence REAL NOT NULL, status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL,
              PRIMARY KEY (photo_id, tag, source),
              FOREIGN KEY (photo_id) REFERENCES photos(photo_id)
            );
            CREATE INDEX IF NOT EXISTS idx_tag_proposals_photo_id ON tag_proposals(photo_id);
            CREATE INDEX IF NOT EXISTS idx_tag_proposals_status ON tag_proposals(status);
            """
        )
        self.conn.commit()

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

    def upsert_embedding(self, embedding: Embedding) -> None:
        self.conn.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?)", (embedding.photo_id, embedding.model_name, embedding.model_version, embedding.vector_dimension, embedding.embedding_path, embedding.created_at))
        self.conn.execute("UPDATE photos SET embedding_status='indexed', updated_at=? WHERE photo_id=?", (now_iso(), embedding.photo_id)); self.conn.commit()

    def list_embeddings(self, model_name: str, model_version: str) -> list[Embedding]:
        rows = self.conn.execute("SELECT * FROM embeddings WHERE model_name=? AND model_version=?", (model_name, model_version)).fetchall()
        return [Embedding(**dict(r)) for r in rows]


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


    def propose_tag(self, photo_id: str, tag: str, confidence: float, source: str = "local-ai") -> None:
        normalized = normalize_tag(tag)
        if not normalized:
            raise ValueError("Tag proposal cannot be empty")
        self.conn.execute(
            """
            INSERT INTO tag_proposals VALUES (?,?,?,?,?,?)
            ON CONFLICT(photo_id, tag, source) DO UPDATE SET
              confidence=excluded.confidence, status=excluded.status, created_at=excluded.created_at
            """,
            (photo_id, normalized, source, float(confidence), "pending", now_iso()),
        )
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

    def set_tag_proposal_status(self, photo_id: str, tag: str, status: str, source: str = "local-ai") -> None:
        normalized = normalize_tag(tag)
        self.conn.execute(
            "UPDATE tag_proposals SET status=? WHERE photo_id=? AND tag=? AND source=?",
            (status, photo_id, normalized, source),
        )
        self.conn.commit()

    def accept_tag_proposal(self, photo_id: str, tag: str, source: str = "local-ai") -> None:
        normalized = normalize_tag(tag)
        self.add_tag(photo_id, normalized, source="auto")
        self.set_tag_proposal_status(photo_id, normalized, "accepted", source=source)

    def reject_tag_proposal(self, photo_id: str, tag: str, source: str = "local-ai") -> None:
        self.set_tag_proposal_status(photo_id, tag, "rejected", source=source)

    def _photo(self, row: sqlite3.Row) -> Photo:
        d = dict(row); d["source_path"] = Path(d["source_path"]); d["missing"] = bool(d["missing"]); return Photo(**d)


def normalize_tag(tag: str) -> str:
    return " ".join(tag.strip().lower().split())
