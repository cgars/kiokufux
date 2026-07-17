import json
from pathlib import Path

import numpy as np
from PIL import Image

from kiokufux.face_review import safe_collection_path
from kiokufux.faces import (FaceDetection, FaceStore, ReviewState, boxes_iou,
    cluster_faces, normalize_embeddings, scan_faces)


class FakeBackend:
    backend_id = "fake"
    model_id = "fixture"
    model_version = "1"
    preprocessing_version = "1"
    embedding_dimensions = 3

    def detect(self, image):
        return [FaceDetection((5, 5, 55, 55), .99, image.crop((5, 5, 55, 55)))]

    def embed(self, faces):
        return np.tile(np.array([[1, 0, 0]], dtype=np.float32), (len(faces), 1))


def test_normalization_and_zero_rejection():
    assert np.allclose(normalize_embeddings(np.array([[3, 4]])), [[.6, .8]])
    try:
        normalize_embeddings(np.array([[0, 0]]))
    except ValueError:
        pass
    else:
        raise AssertionError("zero embedding accepted")


def test_scan_is_idempotent_and_cluster_is_anonymous(tmp_path):
    Image.new("RGB", (100, 100), "red").save(tmp_path / "one.jpg")
    Image.new("RGB", (100, 100), "blue").save(tmp_path / "two.jpg")
    with FaceStore(tmp_path / ".kiokufux") as store:
        first = scan_faces(tmp_path, store, FakeBackend(), minimum_face_size=1)
        ids = [r[0] for r in store.db.execute("SELECT face_id FROM face_occurrences")]
        second = scan_faces(tmp_path, store, FakeBackend(), minimum_face_size=1)
        result = cluster_faces(store)
        assert first["embedded"] == 2 and second["skipped"] == 2
        assert ids == [r[0] for r in store.db.execute("SELECT face_id FROM face_occurrences")]
        assert result == {"groups": 1, "ungrouped": 0}
        assert store.groups()[0]["conflict"] == 0


def test_person_rename_preserves_id_and_atomic_json(tmp_path):
    state = ReviewState(tmp_path)
    person = state.create_person(["face-a"], None)
    renamed = state.rename_person(person["person_id"], "Anna")
    unnamed = state.rename_person(person["person_id"], None)
    assert renamed["person_id"] == unnamed["person_id"] == person["person_id"]
    assert json.loads((tmp_path / "people.json").read_text())["people"][0]["display_name"] is None
    assert not list(tmp_path.glob("*.tmp"))


def test_path_traversal_and_iou(tmp_path):
    assert safe_collection_path(tmp_path, "photo.jpg") == tmp_path / "photo.jpg"
    try:
        safe_collection_path(tmp_path, "../secret")
    except ValueError:
        pass
    else:
        raise AssertionError("traversal accepted")
    assert boxes_iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1
