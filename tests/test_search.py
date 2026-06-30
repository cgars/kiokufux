from fotofux._np import np
from fotofux.catalog import Catalog, now_iso
from fotofux.embeddings import FakeEmbeddingBackend
from fotofux.models import Embedding, Photo
from fotofux.search import search


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
