import json
import logging

import numpy as np
import pytest
from PIL import Image

import kiokufux.gallery as gallery_module
from kiokufux.catalog import Catalog
from kiokufux.faces import FaceDetection, FaceStore, ReviewState, friendly_group_name, scan_faces
from kiokufux.gallery import EXPORT_FORMAT, SCHEMA, build_gallery_document, export_gallery, published_tags
from kiokufux.models import Photo
from kiokufux.scanner import scan as scan_catalog
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
    anna = {"person_id": "person-anna", "label": "Anna", "display_name": "Anna", "friendly_name": "quiet_fox"}
    doc = build_gallery_document("T", [
        {"tags": ["beach", "beach", "family"], "people": [anna, anna]},
        {"tags": ["beach"], "people": [anna]},
    ], {})

    assert doc["schema"] == SCHEMA
    assert doc["item_count"] == 2
    assert doc["tag_frequencies"] == {"beach": 2, "family": 1}
    assert doc["people_frequencies"] == [{**anna, "count": 2}]


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

    index = (tmp_path / "out" / "index.html").read_text()
    script = (tmp_path / "out" / "gallery.js").read_text()
    assert 'id="gallery-data" type="application/json"' in index
    assert f'<meta name="generator" content="{EXPORT_FORMAT}">' in index
    assert '"caption": "Sunny beach"' in index
    assert '<link rel="stylesheet" href="style.css">' not in index
    assert '<script src="gallery.js"></script>' not in index
    assert 'data = JSON.parse(document.querySelector("#gallery-data").textContent);' in index
    assert "fetch(" not in script
    assert "fetch(" not in index


def test_export_gallery_uses_relative_path_after_collection_moves(tmp_path):
    collection = tmp_path / "moved-collection"
    image = collection / "nested" / "photo.jpg"
    image.parent.mkdir(parents=True)
    _image(image)
    db = Catalog(collection / ".kiokufux" / "catalog.sqlite"); db.init_schema()
    stale_source = tmp_path / "old-location" / "nested" / "photo.jpg"
    db.upsert_photo(Photo("id", stale_source, "nested/photo.jpg", "hash", width=8, height=6))

    result = export_gallery(db, tmp_path / "out", collection_root=collection)

    assert result.exported == 1
    assert result.skipped == 0
    assert (tmp_path / "out" / "images" / "id.jpg").read_bytes() == image.read_bytes()


def test_export_gallery_falls_back_to_indexed_absolute_path(tmp_path):
    collection = tmp_path / "collection"
    collection.mkdir()
    image = tmp_path / "legacy-location.jpg"
    _image(image)
    db = Catalog(collection / ".kiokufux" / "catalog.sqlite"); db.init_schema()
    db.upsert_photo(Photo("id", image.resolve(), "missing/photo.jpg", "hash"))

    result = export_gallery(db, tmp_path / "out", collection_root=collection)

    assert result.exported == 1
    assert result.skipped == 0


def test_export_gallery_rejects_relative_paths_outside_collection(tmp_path, caplog):
    collection = tmp_path / "collection"
    collection.mkdir()
    outside = tmp_path / "outside.jpg"
    _image(outside)
    db = Catalog(collection / ".kiokufux" / "catalog.sqlite"); db.init_schema()
    db.upsert_photo(Photo("id", tmp_path / "missing.jpg", "../outside.jpg", "hash"))

    with caplog.at_level(logging.WARNING, logger="kiokufux.gallery"):
        result = export_gallery(db, tmp_path / "out", collection_root=collection)

    assert result.exported == 0
    assert result.skipped == 1
    assert "source not found" in caplog.text
    assert "../outside.jpg" in caplog.text


def test_export_gallery_uses_exported_image_when_thumbnail_creation_fails(tmp_path, monkeypatch, caplog):
    db = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite"); db.init_schema()
    image = tmp_path / "photo.jpg"
    _image(image)
    db.upsert_photo(Photo("id", image, "photo.jpg", "hash"))

    def fail_thumbnail(*_args, **_kwargs):
        raise OSError("thumbnail decoder failed")

    monkeypatch.setattr(gallery_module, "_write_thumbnail", fail_thumbnail)
    with caplog.at_level(logging.WARNING, logger="kiokufux.gallery"):
        result = export_gallery(db, tmp_path / "out")

    doc = json.loads((tmp_path / "out" / "gallery.json").read_text())
    assert result.exported == 1
    assert result.skipped == 0
    assert doc["items"][0]["thumbnail_path"] == doc["items"][0]["image_path"]
    assert "using the exported image as its preview" in caplog.text


def test_export_gallery_logs_source_export_failure(tmp_path, monkeypatch, caplog):
    db = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite"); db.init_schema()
    image = tmp_path / "photo.jpg"
    _image(image)
    db.upsert_photo(Photo("id", image, "photo.jpg", "hash"))

    def fail_copy(*_args, **_kwargs):
        raise PermissionError("destination blocked")

    monkeypatch.setattr(gallery_module, "_copy_image", fail_copy)
    with caplog.at_level(logging.WARNING, logger="kiokufux.gallery"):
        result = export_gallery(db, tmp_path / "out")

    assert result.exported == 0
    assert result.skipped == 1
    assert "PermissionError: destination blocked" in caplog.text


def test_export_gallery_safely_embeds_markup_in_metadata(tmp_path):
    db = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite"); db.init_schema()
    image = tmp_path / "x.jpg"; _image(image)
    db.upsert_photo(Photo("id", image, "x.jpg", "hash"))
    db.upsert_image_analysis(ImageAnalysis(
        photo_id="id",
        source="test",
        model_name="m",
        model_version="v",
        caption="</script><script>alert(1)</script>",
    ))

    export_gallery(db, tmp_path / "out")

    index = (tmp_path / "out" / "index.html").read_text()
    assert "</script><script>alert(1)</script>" not in index
    assert r"\u003c/script>\u003cscript>alert(1)\u003c/script>" in index


def test_export_gallery_replaces_legacy_file_url_export_without_overwrite(tmp_path):
    db = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite"); db.init_schema()
    output = tmp_path / "out"
    output.mkdir()
    (output / "index.html").write_text('<script src="gallery.js"></script>')
    (output / "gallery.json").write_text("{}")
    (output / "gallery.js").write_text('fetch("gallery.json")')

    export_gallery(db, output)

    index = (output / "index.html").read_text()
    assert f'<meta name="generator" content="{EXPORT_FORMAT}">' in index
    assert "fetch(" not in index


class GalleryFaceBackend:
    backend_id = "fake"
    model_id = "gallery"
    model_version = "1"
    preprocessing_version = "1"
    embedding_dimensions = 3

    def detect(self, image):
        return [FaceDetection((10, 10, 50, 60), .99, image.crop((10, 10, 50, 60)))]

    def embed(self, faces):
        return np.tile(np.array([[1, 0, 0]], dtype=np.float32), (len(faces), 1))


def _catalog_with_confirmed_people(root):
    workspace = root / ".kiokufux"
    db = Catalog(workspace / "catalog.sqlite"); db.init_schema()
    for name, color in (("anna.jpg", "red"), ("bert.jpg", "blue")):
        Image.new("RGB", (80, 80), color).save(root / name)
    scan_catalog(root, db, logging.getLogger("test-gallery-people"))
    with FaceStore(workspace) as store:
        scan_faces(root, store, GalleryFaceBackend(), minimum_face_size=1)
        rows = store.db.execute("SELECT face_id,image_path FROM face_occurrences ORDER BY image_path").fetchall()
    faces = {row["image_path"]: row["face_id"] for row in rows}
    state = ReviewState(workspace)
    state.create_person([next(face_id for path, face_id in faces.items() if path.endswith("anna.jpg"))], "Anna", friendly_name="quiet_fox")
    state.create_person([next(face_id for path, face_id in faces.items() if path.endswith("bert.jpg"))], "Bert", friendly_name="calm_badger")
    return db, workspace


def _catalog_with_face_modes(root):
    workspace = root / ".kiokufux"
    db = Catalog(workspace / "catalog.sqlite"); db.init_schema()
    colors = {
        "anna.jpg": "red",
        "grouped.jpg": "blue",
        "unknown.jpg": "green",
        "rejected.jpg": "yellow",
        "excluded.jpg": "purple",
        "conflict.jpg": "orange",
    }
    for name, color in colors.items():
        Image.new("RGB", (80, 80), color).save(root / name)
    scan_catalog(root, db, logging.getLogger("test-gallery-face-modes"))
    with FaceStore(workspace) as store:
        scan_faces(root, store, GalleryFaceBackend(), minimum_face_size=1)
        rows = store.db.execute("SELECT face_id,image_path FROM face_occurrences ORDER BY image_path").fetchall()
        faces = {row["image_path"]: row["face_id"] for row in rows}
        run_id = "gallery-test-run"
        store.db.execute("INSERT INTO cluster_runs VALUES(?,?,?,?)", (run_id, "fake:gallery:1:1:3", "{}", "2026-01-01T00:00:00Z"))
        for group_id, filename, conflict in (
            ("recurring-group", "grouped.jpg", 0),
            ("conflicting-group", "conflict.jpg", 1),
        ):
            face_id = next(face_id for path, face_id in faces.items() if path.endswith(filename))
            store.db.execute("INSERT INTO face_groups VALUES(?,?,?,?,?)", (group_id, run_id, face_id, "unreviewed", conflict))
            store.db.execute("INSERT INTO face_group_members VALUES(?,?,?)", (group_id, face_id, None))
        store.db.commit()

    face_id_for = lambda filename: next(face_id for path, face_id in faces.items() if path.endswith(filename))
    state = ReviewState(workspace)
    state.create_person([face_id_for("anna.jpg")], "Anna", friendly_name="quiet_fox")
    state.record_action("reject-face", [face_id_for("rejected.jpg")])
    state.record_action("exclude-from-clustering", [face_id_for("excluded.jpg")])
    return db, workspace


def test_export_gallery_can_include_confirmed_people_without_biometric_data(tmp_path):
    db, workspace = _catalog_with_confirmed_people(tmp_path)

    result = export_gallery(db, tmp_path / "out", workspace=workspace, face_mode="confirmed")

    assert result.exported == 2
    doc = json.loads((tmp_path / "out" / "gallery.json").read_text())
    assert [(person["label"], person["count"]) for person in doc["people_frequencies"]] == [("Anna", 1), ("Bert", 1)]
    assert {item["people"][0]["label"] for item in doc["items"]} == {"Anna", "Bert"}
    index = (tmp_path / "out" / "index.html").read_text()
    assert 'id="people-cloud"' in index
    assert 'id="detail-people"' in index
    assert "buildPeopleCloud();" in index
    serialized = json.dumps(doc)
    for forbidden in ("face_id", "box", "detection_confidence", "embedding", "model_key"):
        assert forbidden not in serialized


def test_export_gallery_filters_by_confirmed_person_name_or_friendly_name(tmp_path):
    db, workspace = _catalog_with_confirmed_people(tmp_path)

    named = export_gallery(db, tmp_path / "named", workspace=workspace, face_mode="confirmed", people=["anna"])
    private = export_gallery(db, tmp_path / "private", workspace=workspace, people=["calm_badger"])

    assert named.exported == 1
    named_doc = json.loads((tmp_path / "named" / "gallery.json").read_text())
    assert named_doc["items"][0]["filename"] == "anna.jpg"
    assert named_doc["items"][0]["people"][0]["display_name"] == "Anna"
    private_doc = json.loads((tmp_path / "private" / "gallery.json").read_text())
    assert private_doc["items"][0]["filename"] == "bert.jpg"
    assert "people" not in private_doc["items"][0]
    assert private_doc["people_frequencies"] == []
    assert private_doc["source_summary"]["person_filter_count"] == 1
    assert "calm_badger" not in json.dumps(private_doc)


def test_unknown_person_does_not_replace_existing_gallery(tmp_path):
    db, workspace = _catalog_with_confirmed_people(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")

    with pytest.raises(ValueError, match="No confirmed person"):
        export_gallery(db, output, workspace=workspace, people=["Missing"], overwrite=True)

    assert marker.read_text() == "keep"


def test_grouped_face_mode_adds_non_conflicting_anonymous_groups(tmp_path):
    db, workspace = _catalog_with_face_modes(tmp_path)

    export_gallery(db, tmp_path / "out", workspace=workspace, face_mode="grouped")

    doc = json.loads((tmp_path / "out" / "gallery.json").read_text())
    people = doc["people_frequencies"]
    assert [(person["label"], person["status"]) for person in people] == [
        ("Anna", "confirmed"),
        (friendly_group_name("recurring-group"), "provisional"),
    ]
    provisional = next(person for person in people if person["status"] == "provisional")
    assert provisional["review_state"] == "unreviewed"
    assert provisional["identity_id"] == "group:recurring-group"
    assert "Unknown people" not in json.dumps(doc)
    assert friendly_group_name("conflicting-group") not in json.dumps(doc)


def test_detected_face_mode_adds_aggregate_unknown_filter_but_not_private_face_data(tmp_path):
    db, workspace = _catalog_with_face_modes(tmp_path)

    export_gallery(db, tmp_path / "out", workspace=workspace, face_mode="detected")

    doc = json.loads((tmp_path / "out" / "gallery.json").read_text())
    unknown = next(person for person in doc["people_frequencies"] if person["identity_id"] == "unknown")
    assert unknown == {"identity_id": "unknown", "label": "Unknown people", "status": "ungrouped", "count": 1}
    unknown_item = next(item for item in doc["items"] if item["filename"] == "unknown.jpg")
    assert unknown_item["people"] == [{
        "identity_id": "unknown",
        "label": "Unknown people",
        "status": "ungrouped",
        "count_in_photo": 1,
    }]
    for filename in ("rejected.jpg", "excluded.jpg", "conflict.jpg"):
        assert next(item for item in doc["items"] if item["filename"] == filename)["people"] == []
    serialized = json.dumps(doc)
    for forbidden in ("face_id", "box", "detection_confidence", "embedding", "model_key"):
        assert forbidden not in serialized


def test_face_group_and_unknown_filters_work_without_publishing_identity_metadata(tmp_path):
    db, workspace = _catalog_with_face_modes(tmp_path)
    group_name = friendly_group_name("recurring-group")

    result = export_gallery(
        db,
        tmp_path / "out",
        workspace=workspace,
        face_groups=[group_name],
        unknown_faces=True,
    )

    assert result.exported == 2
    doc = json.loads((tmp_path / "out" / "gallery.json").read_text())
    assert {item["filename"] for item in doc["items"]} == {"grouped.jpg", "unknown.jpg"}
    assert all("people" not in item for item in doc["items"])
    assert doc["people_frequencies"] == []
    assert doc["source_summary"]["face_group_filter_count"] == 1
    assert doc["source_summary"]["unknown_face_filter"] is True
    assert group_name not in json.dumps(doc)
