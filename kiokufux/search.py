from __future__ import annotations

from ._np import np
from .catalog import Catalog
from .embeddings import EmbeddingBackend, default_backend
from .models import SearchResult


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def match_label_for_normalized_score(normalized_score: float) -> str:
    """Return query-relative wording, not a probability interpretation."""
    if normalized_score >= 0.85:
        return "very good match"
    if normalized_score >= 0.60:
        return "good match"
    if normalized_score >= 0.35:
        return "possible match"
    return "weak match"


def normalize_search_results(results: list[SearchResult]) -> list[SearchResult]:
    """Add rank and relative interpretation values while preserving raw scores.

    ``SearchResult.score`` remains the raw cosine similarity. The normalized
    score is only relative to the result set for this query and must not be
    displayed as a probability or percentage match.
    """
    if not results:
        return []

    ranked = sorted(results, key=lambda r: r.score, reverse=True)
    scores = [r.score for r in ranked]
    best = max(scores)
    worst = min(scores)
    span = best - worst
    total = len(ranked)

    for index, result in enumerate(ranked, 1):
        if span == 0:
            normalized = 1.0 if total == 1 else 0.5
        else:
            normalized = (result.score - worst) / span
        result.rank = index
        result.total_ranked = total
        result.top_percent = index / total * 100.0
        result.normalized_score = normalized
        result.match_label = match_label_for_normalized_score(normalized)
    return ranked


def search(catalog: Catalog, query: str, top_k: int = 10, backend: EmbeddingBackend | None = None) -> list[SearchResult]:
    backend = backend or default_backend()
    q = backend.embed_text(query)
    results = []
    for emb in catalog.list_embeddings(backend.model_name, backend.model_version):
        photo = catalog.get_photo(emb.photo_id)
        if not photo or photo.missing:
            continue
        vec = np.load(emb.embedding_path)
        results.append(
            SearchResult(
                photo.photo_id,
                cosine(q, vec),
                photo.source_path,
                photo.thumbnail_path,
                {
                    "relative_path": photo.relative_path,
                    "width": photo.width,
                    "height": photo.height,
                    "datetime_original": photo.exif_datetime_original,
                    "gps": {"lat": photo.exif_gps_lat, "lon": photo.exif_gps_lon},
                },
            )
        )
    return normalize_search_results(results)[:top_k]
