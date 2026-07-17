import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from kiokufux.face_review import make_server, safe_collection_path
from kiokufux.faces import (FaceDetection, FaceStore, ReviewState, boxes_iou,
    cluster_faces, friendly_group_name, normalize_embeddings, scan_faces)


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


def test_friendly_group_names_are_stable_and_non_identifying():
    name = friendly_group_name("49bb24b6-b709-4c5c-b107-19257ce9e02f")
    assert name == friendly_group_name("49bb24b6-b709-4c5c-b107-19257ce9e02f")
    assert name.count("_") == 1
    assert "49bb" not in name


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


def test_threaded_review_server_uses_request_local_sqlite_connections(tmp_path):
    server = make_server(tmp_path, tmp_path / ".kiokufux")
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/api/groups"
        responses = []

        def fetch_groups():
            with urllib.request.urlopen(url, timeout=2) as response:
                responses.append(json.load(response))

        clients = [threading.Thread(target=fetch_groups) for _ in range(4)]
        for client in clients:
            client.start()
        for client in clients:
            client.join()
        assert responses == [[], [], [], []]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_review_api_exposes_group_detail_and_source_context(tmp_path):
    Image.new("RGB", (100, 100), "red").save(tmp_path / "one.jpg")
    Image.new("RGB", (100, 100), "blue").save(tmp_path / "two.jpg")
    workspace = tmp_path / ".kiokufux"
    with FaceStore(workspace) as store:
        scan_faces(tmp_path, store, FakeBackend(), minimum_face_size=1)
        cluster_faces(store)
        group_id = store.groups()[0]["group_id"]
    server = make_server(tmp_path, workspace)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(base + f"/api/groups/{group_id}") as response:
            group = json.load(response)
        assert len(group["faces"]) == 2
        with urllib.request.urlopen(base + f"/api/images/{group['faces'][0]['image_id']}/thumbnail") as response:
            assert response.headers["Content-Type"] == "image/jpeg"
            assert response.read().startswith(b"\xff\xd8")
        with urllib.request.urlopen(base + f"/api/images/{group['faces'][0]['image_id']}/faces") as response:
            detections = json.load(response)
        assert detections[0]["face_id"] == group["faces"][0]["face_id"]
        assert all(0 <= detections[0][key] <= 1 for key in ("x1", "y1", "x2", "y2"))
        with urllib.request.urlopen(base + "/") as response:
            page = response.read().decode()
        assert "View in photograph" in page
        assert "Compare selected" in page
        assert "Marked photograph" in page
        assert "_" in group["friendly_id"]
        assert "g.friendly_id" in page
        assert "Split selected" in page
        assert "Confirm as person" in page
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
