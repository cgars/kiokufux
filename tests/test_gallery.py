import json

from PIL import Image

from kiokufux.catalog import Catalog
from kiokufux.gallery import SCHEMA, build_gallery_document, export_gallery, published_tags
from kiokufux.models import Photo
from kiokufux.vlm import ImageAnalysis


def _image(path):
    Image.new("RGB", (8, 6), "red").save(path)


def test_published_tags_canonicalizes_and_excludes_proposals(tmp_path):
    db = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite"); db.init_schema()
    img = tmp_path / "x.jpg"; _image(img)
    db.upsert_photo(Photo("id", img, "x.jpg", "hash"))
    db.upsert_vocabulary_tag("beach", status="accepted", aliases=["seaside"])
    db.add_tag("id", "seaside")
    db.add_tag("id", "family", source="auto")
    db.propose_tag("id", "draft", 0.9)

    assert published_tags(db, "id") == ["beach", "family"]


def test_build_gallery_document_counts_distinct_tags_per_image():
    doc = build_gallery_document("T", [{"tags": ["beach", "beach", "family"]}, {"tags": ["beach"]}], {})

    assert doc["schema"] == SCHEMA
    assert doc["item_count"] == 2
    assert doc["tag_frequencies"] == {"beach": 2, "family": 1}


def test_export_gallery_writes_portable_files_and_filters_by_tag(tmp_path):
    db = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite"); db.init_schema()
    beach = tmp_path / "beach.jpg"; cat = tmp_path / "cat.jpg"
    _image(beach); _image(cat)
    db.upsert_photo(Photo("id1", beach, "beach.jpg", "h1", width=8, height=6))
    db.upsert_photo(Photo("id2", cat, "cat.jpg", "h2"))
    db.add_tag("id1", "beach")
    db.add_tag("id2", "cat")
    db.upsert_image_analysis(ImageAnalysis(photo_id="id1", source="test", model_name="m", model_version="v", caption="Sunny beach", description="A shoreline"))

    result = export_gallery(db, tmp_path / "out", tags=["beach"], min_tag_count=1)

    assert result.selected == 1
    assert result.exported == 1
    doc = json.loads((tmp_path / "out" / "gallery.json").read_text())
    assert doc["items"][0]["caption"] == "Sunny beach"
    assert doc["items"][0]["image_path"] == "images/id1.jpg"
    assert (tmp_path / "out" / "index.html").exists()
    assert (tmp_path / "out" / "style.css").exists()
    assert (tmp_path / "out" / "gallery.js").exists()
    assert (tmp_path / "out" / "images" / "id1.jpg").exists()
    assert (tmp_path / "out" / "thumbnails" / "id1.jpg").exists()
