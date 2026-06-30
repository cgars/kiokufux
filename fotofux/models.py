from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Photo:
    photo_id: str
    source_path: Path
    relative_path: str
    file_hash: str
    file_size: int | None = None
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    created_at_file: str | None = None
    modified_at_file: str | None = None
    exif_datetime_original: str | None = None
    exif_gps_lat: float | None = None
    exif_gps_lon: float | None = None
    thumbnail_path: str | None = None
    embedding_status: str = "pending"
    indexed_at: str | None = None
    updated_at: str | None = None
    missing: bool = False
    error: str | None = None


@dataclass(slots=True)
class Embedding:
    photo_id: str
    model_name: str
    model_version: str
    vector_dimension: int
    embedding_path: str
    created_at: str


@dataclass(slots=True)
class SearchResult:
    photo_id: str
    score: float
    source_path: Path
    thumbnail_path: str | None
    metadata_summary: dict[str, Any]
