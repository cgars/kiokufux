from __future__ import annotations

from ._np import np
from .catalog import Catalog
from .embeddings import EmbeddingBackend, default_backend
from .models import SearchResult


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def search(catalog: Catalog, query: str, top_k: int = 10, backend: EmbeddingBackend | None = None) -> list[SearchResult]:
    backend = backend or default_backend(); q = backend.embed_text(query); results = []
    for emb in catalog.list_embeddings(backend.model_name, backend.model_version):
        photo = catalog.get_photo(emb.photo_id)
        if not photo or photo.missing: continue
        vec = np.load(emb.embedding_path)
        results.append(SearchResult(photo.photo_id, cosine(q, vec), photo.source_path, photo.thumbnail_path, {"relative_path": photo.relative_path, "width": photo.width, "height": photo.height, "datetime_original": photo.exif_datetime_original, "gps": {"lat": photo.exif_gps_lat, "lon": photo.exif_gps_lon}}))
    return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]
