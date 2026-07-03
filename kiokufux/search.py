from __future__ import annotations

from statistics import median

from ._np import np
from .catalog import Catalog
from .embeddings import EmbeddingBackend, default_backend
from .models import SearchResult

DEFAULT_MIN_RAW_SCORE = 0.20
DEFAULT_MIN_ROBUST_Z = 1.0
ROBUST_Z_SCALE = 1.4826


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance ** 0.5


def robust_z_scores(scores: list[float]) -> list[float]:
    """Return robust z-scores based on median absolute deviation.

    A robust z-score asks whether a raw similarity score is unusual compared
    with the rest of this specific result set. It is not a probability.
    """
    if not scores:
        return []
    center = median(scores)
    deviations = [abs(score - center) for score in scores]
    mad = median(deviations)
    if mad > 0:
        return [(score - center) / (ROBUST_Z_SCALE * mad) for score in scores]

    std = _population_std(scores)
    if std > 0:
        mean = sum(scores) / len(scores)
        return [(score - mean) / std for score in scores]
    return [0.0 for _ in scores]


def match_label_for_result(normalized_score: float, confidence_gate_passed: bool) -> str:
    """Return honest match wording, never a probability interpretation."""
    if not confidence_gate_passed:
        return "low confidence"
    if normalized_score >= 0.85:
        return "very good match"
    if normalized_score >= 0.60:
        return "good match"
    if normalized_score >= 0.35:
        return "possible match"
    return "low confidence"


def normalize_search_results(
    results: list[SearchResult],
    min_raw_score: float = DEFAULT_MIN_RAW_SCORE,
    min_robust_z: float = DEFAULT_MIN_ROBUST_Z,
) -> list[SearchResult]:
    """Add rank, robust z-score, and gated interpretation values.

    ``SearchResult.score`` remains the raw cosine similarity. Rank decides
    ordering. The confidence gate decides whether the system should trust the
    best match instead of calling it merely the closest available result.
    """
    if not results:
        return []

    ranked = sorted(results, key=lambda r: r.score, reverse=True)
    scores = [r.score for r in ranked]
    z_scores = robust_z_scores(scores)
    best = max(scores)
    worst = min(scores)
    span = best - worst
    total = len(ranked)
    best_z = z_scores[0]
    best_is_confident = best >= min_raw_score and best_z >= min_robust_z

    for index, (result, z_score) in enumerate(zip(ranked, z_scores), 1):
        if span == 0:
            normalized = 1.0 if total == 1 else 0.5
        else:
            normalized = (result.score - worst) / span
        result.rank = index
        result.total_ranked = total
        result.top_percent = index / total * 100.0
        result.normalized_score = normalized
        result.robust_z_score = z_score
        result.confidence_gate_passed = best_is_confident and result.score >= min_raw_score and z_score >= min_robust_z
        if not best_is_confident and index == 1:
            result.match_label = "closest available · low confidence"
        else:
            result.match_label = match_label_for_result(normalized, bool(result.confidence_gate_passed))
    return ranked


def search(
    catalog: Catalog,
    query: str,
    top_k: int = 10,
    backend: EmbeddingBackend | None = None,
    min_raw_score: float = DEFAULT_MIN_RAW_SCORE,
    min_robust_z: float = DEFAULT_MIN_ROBUST_Z,
) -> list[SearchResult]:
    backend = backend or default_backend()
    q = backend.embed_text(query)
    results = []
    for emb in catalog.list_embeddings(backend.model_name, backend.model_version):
        photo = catalog.get_photo(emb.photo_id)
        if not photo or photo.missing:
            continue
        vec = np.load(catalog.artifact_path(emb.embedding_path))
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
    return normalize_search_results(results, min_raw_score=min_raw_score, min_robust_z=min_robust_z)[:top_k]
