import json
from kiokufux.models import Photo
from kiokufux.sidecar import SCHEMA, sidecar_document, export_sidecars
from kiokufux.catalog import Catalog


def test_sidecar_document_schema(tmp_path):
    photo = Photo("id", tmp_path / "x.jpg", "x.jpg", "hash", width=1, height=2)
    doc = sidecar_document(photo)
    assert doc["schema"] == SCHEMA
    assert doc["metadata"]["width"] == 1


def test_export_sidecars(tmp_path):
    db = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite"); db.init_schema()
    img = tmp_path / "x.jpg"; img.write_text("not image")
    db.upsert_photo(Photo("id", img, "x.jpg", "hash"))
    db.add_tag("id", "family")
    db.add_tag("id", "dog", source="auto")
    db.propose_tag("id", "cat", 0.51)
    assert export_sidecars(db) == 1
    doc = json.loads((tmp_path / "x.jpg.kiokufux.json").read_text())
    assert doc["photo_id"] == "id"
    assert doc["review"]["tags"] == ["family"]
    assert doc["semantic"]["auto_tags"] == ["dog"]
    assert doc["review"]["tag_proposals"] == [{"confidence": 0.51, "source": "ai-zero-shot", "status": "pending", "tag": "cat"}]


def test_export_sidecars_includes_vlm_analysis_and_proposal_evidence(tmp_path):
    from kiokufux.vlm import ImageAnalysis, ImageAnalysisTag

    db = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite"); db.init_schema()
    img = tmp_path / "garden.jpg"; img.write_text("not image")
    db.upsert_photo(Photo("id", img, "garden.jpg", "hash"))
    db.upsert_image_analysis(ImageAnalysis(
        photo_id="id",
        source="vlm-test",
        model_name="fake",
        model_version="test",
        caption="A garden photo.",
        description="A complete garden description.",
        objects=["table"],
        scene="garden",
        candidate_tags=[ImageAnalysisTag("garden", 0.88, "place", "green plants visible")],
    ))

    assert export_sidecars(db) == 1
    doc = json.loads((tmp_path / "garden.jpg.kiokufux.json").read_text())

    assert doc["semantic"]["caption"] == "A garden photo."
    assert doc["semantic"]["description"] == "A complete garden description."
    assert doc["semantic"]["vlm"]["description"] == "A complete garden description."
    assert doc["semantic"]["vlm"]["objects"] == ["table"]
    assert doc["review"]["tag_proposals"] == [{
        "category_hint": "place",
        "confidence": 0.88,
        "evidence": "green plants visible",
        "source": "vlm-test",
        "status": "pending",
        "tag": "garden",
    }]
