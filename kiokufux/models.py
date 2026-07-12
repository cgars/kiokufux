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
class PhotoTag:
    photo_id: str
    tag: str
    source: str
    created_at: str


@dataclass(slots=True)
class TagProposal:
    photo_id: str
    tag: str
    source: str
    confidence: float
    status: str
    created_at: str


@dataclass(slots=True)
class TagVocabularyEntry:
    tag: str
    category: str
    scope: str
    status: str
    parent: str | None
    aliases: list[str]
    notes: str | None
    created_at: str
    updated_at: str


@dataclass(slots=True)
class TagProposalSummary:
    tag: str
    source: str
    status: str
    proposal_count: int
    photo_count: int
    avg_confidence: float
    max_confidence: float


@dataclass(slots=True)
class SearchResult:
    photo_id: str
    score: float
    source_path: Path
    thumbnail_path: str | None
    metadata_summary: dict[str, Any]
    rank: int | None = None
    total_ranked: int | None = None
    top_percent: float | None = None
    normalized_score: float | None = None
    robust_z_score: float | None = None
    confidence_gate_passed: bool | None = None
    match_label: str | None = None
