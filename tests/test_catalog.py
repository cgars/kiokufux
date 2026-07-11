from kiokufux.catalog import Catalog
from kiokufux.models import Photo


def test_catalog_insert_update_and_missing(tmp_path):
    c = Catalog(tmp_path / "catalog.sqlite"); c.init_schema()
    p = Photo("id1", tmp_path / "a.jpg", "a.jpg", "hash1", width=10)
    c.upsert_photo(p)
    assert c.get_photo("id1").width == 10
    c.upsert_photo(Photo("id1", tmp_path / "a.jpg", "a.jpg", "hash1", width=20))
    assert c.get_photo("id1").width == 20
    c.mark_missing_except([])
    assert c.get_photo("id1").missing is True


def test_catalog_photo_tags_are_normalized_and_removable(tmp_path):
    c = Catalog(tmp_path / "catalog.sqlite"); c.init_schema()
    c.upsert_photo(Photo("id1", tmp_path / "a.jpg", "a.jpg", "hash1"))

    c.add_tag("id1", "  Family  Party ")
    c.add_tag("id1", "family party")

    tags = c.list_tags("id1")
    assert len(tags) == 1
    assert tags[0].tag == "family party"
    assert tags[0].source == "manual"

    c.remove_tag("id1", "family party")
    assert c.list_tags("id1") == []


def test_catalog_resolves_full_or_seven_character_photo_ids(tmp_path):
    c = Catalog(tmp_path / "c.sqlite"); c.init_schema()
    photo_path = tmp_path / "p.jpg"; photo_path.write_text("x")
    c.upsert_photo(Photo("abcdef123456", photo_path, photo_path.name, "hash"))

    assert c.resolve_photo_id("abcdef1") == "abcdef123456"
    assert c.resolve_photo_id("abcdef123456") == "abcdef123456"


def test_catalog_summarizes_tag_proposals_by_tag_status_and_source(tmp_path):
    c = Catalog(tmp_path / "catalog.sqlite"); c.init_schema()
    c.upsert_photo(Photo("id1", tmp_path / "a.jpg", "a.jpg", "hash1"))
    c.upsert_photo(Photo("id2", tmp_path / "b.jpg", "b.jpg", "hash2"))
    c.propose_tag("id1", "Dog", 0.8)
    c.propose_tag("id2", "dog", 0.6)
    c.propose_tag("id2", "cat", 0.9)
    c.reject_tag_proposal("id2", "cat")

    pending = c.summarize_tag_proposals()

    assert len(pending) == 1
    assert pending[0].tag == "dog"
    assert pending[0].photo_count == 2
    assert pending[0].proposal_count == 2
    assert pending[0].avg_confidence == 0.7
    assert pending[0].max_confidence == 0.8
    assert pending[0].status == "pending"

    all_rows = c.summarize_tag_proposals(status=None)
    assert [(row.tag, row.status) for row in all_rows] == [("dog", "pending"), ("cat", "rejected")]


def test_catalog_vocabulary_propose_accept_merge_and_apply(tmp_path):
    c = Catalog(tmp_path / "catalog.sqlite"); c.init_schema()
    for photo_id in ["id1", "id2"]:
        c.upsert_photo(Photo(photo_id, tmp_path / f"{photo_id}.jpg", f"{photo_id}.jpg", f"hash-{photo_id}"))
    c.propose_tag("id1", "backyard", 0.8)
    c.propose_tag("id2", "garden", 0.9)

    assert c.propose_vocabulary_from_tag_proposals() == 2
    proposed = {entry.tag: entry for entry in c.list_vocabulary(status="proposed")}
    assert set(proposed) == {"backyard", "garden"}

    c.upsert_vocabulary_tag("garden", category="place", scope="core", status="accepted", aliases=["yard"])
    c.merge_vocabulary_tag("backyard", "garden")

    assert c.canonical_tag("yard") == "garden"
    assert c.canonical_tag("backyard") == "garden"
    assert c.apply_vocabulary_to_tag_proposals() == 2
    assert [tag.tag for tag in c.list_tags("id1")] == ["garden"]
    assert [tag.tag for tag in c.list_tags("id2")] == ["garden"]
    assert {p.tag: p.status for p in c.list_tag_proposals(status=None)} == {"backyard": "accepted", "garden": "accepted"}


def test_catalog_rejected_vocabulary_rejects_matching_pending_proposals(tmp_path):
    c = Catalog(tmp_path / "catalog.sqlite"); c.init_schema()
    c.upsert_photo(Photo("id1", tmp_path / "a.jpg", "a.jpg", "hash1"))
    c.propose_tag("id1", "nice", 0.7)
    c.upsert_vocabulary_tag("nice", status="rejected")

    assert c.apply_vocabulary_to_tag_proposals() == 1
    assert c.list_tag_proposals("id1", status=None)[0].status == "rejected"


def test_catalog_stores_vlm_analysis_and_evidence(tmp_path):
    from kiokufux.vlm import ImageAnalysis, ImageAnalysisTag

    c = Catalog(tmp_path / "catalog.sqlite"); c.init_schema()
    c.upsert_photo(Photo("id1", tmp_path / "garden.jpg", "garden.jpg", "hash1"))
    c.upsert_image_analysis(ImageAnalysis(
        photo_id="id1",
        source="vlm-test",
        model_name="fake",
        model_version="test",
        caption="A garden photo.",
        description="A complete garden description.",
        objects=["table"],
        scene="garden",
        candidate_tags=[ImageAnalysisTag("garden", 0.88, "place", "green plants visible")],
    ))

    stored = c.get_image_analysis("id1")
    assert stored is not None
    assert stored.caption == "A garden photo."
    assert stored.description == "A complete garden description."
    assert stored.objects == ["table"]
    proposals = c.list_tag_proposals("id1")
    assert proposals[0].tag == "garden"
    assert proposals[0].source == "vlm-test"
    assert c.tag_proposal_evidence("id1")[("garden", "vlm-test")] == {"category_hint": "place", "evidence": "green plants visible"}
