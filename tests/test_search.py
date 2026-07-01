import pytest

from kiokufux._np import np
from kiokufux.catalog import Catalog, now_iso
from kiokufux.embeddings import FakeEmbeddingBackend
from kiokufux.models import Embedding, Photo
from kiokufux.search import search


def test_search_ranking_with_fake_backend(tmp_path):
    c = Catalog(tmp_path / "catalog.sqlite"); c.init_schema(); b = FakeEmbeddingBackend()
    img1 = tmp_path / "church.jpg"; img2 = tmp_path / "dog.jpg"; img1.write_text("a"); img2.write_text("b")
    for img in (img1, img2):
        pid = img.stem; c.upsert_photo(Photo(pid, img, img.name, pid, embedding_status="indexed"))
        vec = b.embed_text("church" if pid == "church" else "dog")
        ep = tmp_path / f"{pid}.npy"; np.save(ep, vec)
        c.upsert_embedding(Embedding(pid, b.model_name, b.model_version, b.dimension, str(ep), now_iso()))
    results = search(c, "church", backend=b)
    assert results[0].photo_id == "church"


def test_missing_file_record_practical(tmp_path):
    c = Catalog(tmp_path / "catalog.sqlite"); c.init_schema()
    c.upsert_photo(Photo("id", tmp_path / "gone.jpg", "gone.jpg", "hash"))
    c.mark_missing_except([])
    assert c.get_photo("id").missing


def test_search_results_include_query_relative_normalization(tmp_path):
    c = Catalog(tmp_path / "catalog.sqlite"); c.init_schema(); b = FakeEmbeddingBackend()
    query = b.embed_text("church")
    vectors = {
        "best": query,
        "middle": b.embed_text("chapel"),
        "last": b.embed_text("dog"),
    }
    for pid, vec in vectors.items():
        img = tmp_path / f"{pid}.jpg"; img.write_text(pid)
        c.upsert_photo(Photo(pid, img, img.name, pid, embedding_status="indexed"))
        ep = tmp_path / f"{pid}.npy"; np.save(ep, vec)
        c.upsert_embedding(Embedding(pid, b.model_name, b.model_version, b.dimension, str(ep), now_iso()))

    results = search(c, "church", backend=b)

    assert [r.rank for r in results] == [1, 2, 3]
    assert all(r.score is not None for r in results)
    assert results[0].normalized_score == 1.0
    assert results[0].match_label == "very good match"
    assert results[0].top_percent == pytest.approx(100 / 3)
    assert results[-1].normalized_score == 0.0
    assert results[-1].match_label == "weak match"
