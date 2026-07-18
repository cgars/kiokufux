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

from PIL import Image
import numpy as np
from kiokufux.faces import FaceDetection, FaceStore, ReviewState, cluster_faces, friendly_group_name, scan_faces


class SidecarFakeBackend:
    backend_id = "fake"
    model_id = "sidecar"
    model_version = "1"
    preprocessing_version = "1"
    embedding_dimensions = 3

    def detect(self, image):
        return [FaceDetection((10, 10, 50, 60), .992, image.crop((10, 10, 50, 60)))]

    def embed(self, faces):
        return np.tile(np.array([[1, 0, 0]], dtype=np.float32), (len(faces), 1))


class NoFaceBackend(SidecarFakeBackend):
    def detect(self, image):
        return []


def _catalog_with_image(root, name="face.jpg"):
    db = Catalog(root / ".kiokufux" / "catalog.sqlite"); db.init_schema()
    img = root / name
    Image.new("RGB", (80, 80), "white").save(img)
    from kiokufux.hashing import file_sha256
    photo_id = file_sha256(img)
    db.upsert_photo(Photo(photo_id, img, name, photo_id, width=80, height=80))
    return db, img, photo_id


def test_export_without_face_files_is_not_scanned_and_read_only(tmp_path):
    db, img, _ = _catalog_with_image(tmp_path)
    assert export_sidecars(db, workspace=tmp_path / ".kiokufux") == 1
    doc = json.loads((tmp_path / "face.jpg.kiokufux.json").read_text())
    assert doc["schema"] == "kiokufux.sidecar.v2"
    assert doc["faces"] == {"model_key": None, "occurrences": [], "scan_status": "not_scanned"}
    assert not (tmp_path / ".kiokufux" / "face-review.json").exists()
    assert not (tmp_path / ".kiokufux" / "people.json").exists()


def test_export_scanned_image_with_no_faces(tmp_path):
    db, img, photo_id = _catalog_with_image(tmp_path)
    workspace = tmp_path / ".kiokufux"
    with FaceStore(workspace) as store:
        scan_faces(tmp_path, store, NoFaceBackend(), minimum_face_size=1)
    export_sidecars(db, workspace=workspace)
    faces = json.loads((tmp_path / "face.jpg.kiokufux.json").read_text())["faces"]
    assert faces["scan_status"] == "scanned"
    assert faces["model_key"].startswith("fake:sidecar")
    assert faces["occurrences"] == []


def test_export_provisional_confirmed_rejected_and_excluded_faces(tmp_path):
    db = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite"); db.init_schema()
    workspace = tmp_path / ".kiokufux"
    photo_ids = []
    for name, color in [("a.jpg", "red"), ("b.jpg", "blue")]:
        img = tmp_path / name
        Image.new("RGB", (80, 80), color).save(img)
        from kiokufux.hashing import file_sha256
        photo_id = file_sha256(img)
        photo_ids.append(photo_id)
        db.upsert_photo(Photo(photo_id, img, name, photo_id, width=80, height=80))
    with FaceStore(workspace) as store:
        scan_faces(tmp_path, store, SidecarFakeBackend(), minimum_face_size=1)
        cluster_faces(store)
        group = store.groups()[0]
        face_ids = [f["face_id"] for f in store.group(group["group_id"])["faces"]]
    state = ReviewState(workspace)
    state.record_action("reject-face", [face_ids[0]])
    state.record_action("exclude-from-clustering", [face_ids[1]])
    export_sidecars(db, workspace=workspace)
    docs = [json.loads((tmp_path / f"{name}.kiokufux.json").read_text()) for name in ("a.jpg", "b.jpg")]
    statuses = sorted(d["faces"]["occurrences"][0]["identity"]["status"] for d in docs)
    assert statuses == ["excluded", "rejected"]
    serialized = json.dumps(docs)
    assert "embedding BLOB" not in serialized
    assert "face-thumbnails" not in serialized


def test_export_confirmed_person_has_stable_friendly_and_display_name(tmp_path):
    db, img, photo_id = _catalog_with_image(tmp_path)
    workspace = tmp_path / ".kiokufux"
    with FaceStore(workspace) as store:
        scan_faces(tmp_path, store, SidecarFakeBackend(), minimum_face_size=1)
        face_id = store.db.execute("SELECT face_id FROM face_occurrences").fetchone()[0]
    state = ReviewState(workspace)
    person = state.create_person([face_id], "Anna", friendly_name="quiet_fox")
    state.rename_person(person["person_id"], None)
    state.rename_person(person["person_id"], "Anna")
    export_sidecars(db, workspace=workspace)
    identity = json.loads((tmp_path / "face.jpg.kiokufux.json").read_text())["faces"]["occurrences"][0]["identity"]
    assert identity == {"display_name": "Anna", "friendly_name": "quiet_fox", "person_id": person["person_id"], "status": "confirmed"}


def test_people_schema_v1_migrates_to_stable_friendly_name(tmp_path):
    workspace = tmp_path / ".kiokufux"
    workspace.mkdir()
    (workspace / "people.json").write_text(json.dumps({"schema_version": 1, "people": [{"person_id": "person-1", "display_name": "Anna"}]}))
    state = ReviewState(workspace)
    first = state.people["people"][0]["friendly_name"]
    state = ReviewState(workspace)
    assert state.people["schema_version"] == 2
    assert state.people["people"][0]["friendly_name"] == first


def test_export_provisional_group_identity(tmp_path):
    db = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite"); db.init_schema()
    workspace = tmp_path / ".kiokufux"
    for name, color in [("a.jpg", "red"), ("b.jpg", "blue")]:
        img = tmp_path / name
        Image.new("RGB", (80, 80), color).save(img)
        from kiokufux.hashing import file_sha256
        photo_id = file_sha256(img)
        db.upsert_photo(Photo(photo_id, img, name, photo_id, width=80, height=80))
    with FaceStore(workspace) as store:
        scan_faces(tmp_path, store, SidecarFakeBackend(), minimum_face_size=1)
        cluster_faces(store)
        group = store.groups()[0]
    export_sidecars(db, workspace=workspace)
    identity = json.loads((tmp_path / "a.jpg.kiokufux.json").read_text())["faces"]["occurrences"][0]["identity"]
    assert identity["status"] == "provisional"
    assert identity["group_id"] == group["group_id"]
    assert identity["friendly_name"] == friendly_group_name(group["group_id"])
    assert identity["cluster_run_id"] == group["cluster_run_id"]
    assert identity["group_review_state"] == "unreviewed"
