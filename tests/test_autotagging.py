from pathlib import Path

from kiokufux._np import np
from kiokufux.autotagging import EmbeddingAutoTagger, normalize_candidate_tags, propose_tags
from kiokufux.catalog import Catalog
from kiokufux.models import Photo


class TinyTagBackend:
    model_name = "tiny"
    model_version = "test"
    dimension = 2

    def embed_image(self, image_path: Path):
        return np.array([1.0, 0.0])

    def embed_text(self, text: str):
        if text == "cow":
            return np.array([1.0, 0.0])
        return np.array([0.0, 1.0])


def test_embedding_auto_tagger_proposes_zero_shot_tags(tmp_path):
    photo_path = tmp_path / "anything.jpg"
    photo_path.write_text("not an image")
    photo = Photo("id", photo_path, photo_path.name, "hash")

    proposals = EmbeddingAutoTagger(
        backend=TinyTagBackend(),
        candidate_tags=["cow", "dog"],
        top_k=2,
        min_score=0.5,
    ).propose(photo)

    assert [p.tag for p in proposals] == ["cow"]
    assert proposals[0].reason == "zero-shot image/text similarity"


def test_propose_accept_and_reject_tag_workflow(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite"); catalog.init_schema()
    photo_path = tmp_path / "photo.jpg"; photo_path.write_text("x")
    catalog.upsert_photo(Photo("id", photo_path, photo_path.name, "hash"))
    tagger = EmbeddingAutoTagger(backend=TinyTagBackend(), candidate_tags=["cow", "dog"], min_score=0.5)

    assert propose_tags(catalog, tagger) == 1
    proposals = catalog.list_tag_proposals("id")
    assert [p.tag for p in proposals] == ["cow"]
    assert proposals[0].source == "ai-zero-shot"

    catalog.accept_tag_proposal("id", "cow", source="ai-zero-shot")
    catalog.reject_tag_proposal("id", "dog", source="ai-zero-shot")

    assert catalog.list_tags("id")[0].tag == "cow"
    statuses = {p.tag: p.status for p in catalog.list_tag_proposals("id", status=None)}
    assert statuses == {"cow": "accepted"}


def test_normalize_candidate_tags_deduplicates_and_strips():
    assert normalize_candidate_tags([" Cow ", "cow", "family party"]) == ["cow", "family party"]


def test_accept_all_tag_proposals_can_apply_globally_or_by_photo(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite"); catalog.init_schema()
    for photo_id in ["id1", "id2"]:
        photo_path = tmp_path / f"{photo_id}.jpg"; photo_path.write_text("x")
        catalog.upsert_photo(Photo(photo_id, photo_path, photo_path.name, f"hash-{photo_id}"))
        catalog.propose_tag(photo_id, "cow", 0.9)

    assert catalog.accept_tag_proposals("id1") == 1
    assert [tag.tag for tag in catalog.list_tags("id1")] == ["cow"]
    assert catalog.list_tags("id2") == []

    assert catalog.accept_tag_proposals() == 1
    assert [tag.tag for tag in catalog.list_tags("id2")] == ["cow"]
