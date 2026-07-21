import json
import threading
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from kiokufux.face_review import HTML, _send_response_body, make_server, safe_collection_path
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
    assert safe_collection_path(tmp_path, r"nested\photo.jpg") == tmp_path / "nested" / "photo.jpg"
    try:
        safe_collection_path(tmp_path, "../secret")
    except ValueError:
        pass
    else:
        raise AssertionError("traversal accepted")
    assert boxes_iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1

def test_review_comparison_sources_fit_and_can_scroll_side_by_side():
    assert ".compare-items.side-by-side" in HTML
    assert "compareItems.classList.toggle('side-by-side',items.length>1)" in HTML
    assert "function fitPhoto(item)" in HTML
    assert "Math.min(viewport.clientWidth/img.naturalWidth,viewport.clientHeight/img.naturalHeight)" in HTML


def test_review_response_ignores_client_disconnects():
    class BrokenWriter:
        def write(self, _data):
            raise BrokenPipeError("client closed")

    class Handler:
        wfile = BrokenWriter()

        def __init__(self):
            self.calls = []

        def send_response(self, status):
            self.calls.append(("status", status))

        def send_header(self, name, value):
            self.calls.append(("header", name, value))

        def end_headers(self):
            self.calls.append(("end",))

    handler = Handler()

    _send_response_body(handler, 200, "image/jpeg", b"jpeg")

    assert ("status", 200) in handler.calls
    assert ("header", "Content-Length", "4") in handler.calls


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


def test_review_source_image_route_accepts_windows_relative_paths(tmp_path):
    image_path = tmp_path / "nested" / "photo.jpg"
    image_path.parent.mkdir()
    Image.new("RGB", (100, 100), "red").save(image_path)
    workspace = tmp_path / ".kiokufux"
    with FaceStore(workspace) as store:
        scan_faces(tmp_path, store, FakeBackend(), minimum_face_size=1)
        row = store.db.execute("SELECT image_id FROM face_occurrences LIMIT 1").fetchone()
        store.db.execute("UPDATE face_occurrences SET image_path=? WHERE image_id=?", (r"nested\photo.jpg", row["image_id"]))
        store.db.commit()

    server = make_server(tmp_path, workspace)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/api/images/{row['image_id']}/thumbnail"
        with urllib.request.urlopen(url, timeout=2) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/jpeg"
            assert response.read().startswith(b"\xff\xd8")
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
        assert "Open photograph" in page
        assert "Compare selected" in page
        assert "Source photograph" in page
        assert "Zoom in" in page
        assert "zoom-viewport" in page
        assert "wheelZoom" in page
        assert "KiokuFux · People" in page
        assert "actions-panel" in page
        assert "More actions" in page
        assert "Clear crop" in page
        assert "checkmark" in page
        assert "Source photograph" in page
        assert "_" in group["friendly_id"]
        assert "g.friendly_id" in page
        assert "Split selected" in page
        assert "Confirm person" in page
        assert 'name="viewport"' in page
        assert 'name="theme-color" content="#21342b"' in page
        assert "--pine-950: #17241f" in page
        assert "Faulmann gallery visual system" in page
        assert "setActionMode('ungrouped')" in page
        assert "Known with care" in page
        assert 'role="button" tabindex="0"' in page
        assert "aria-pressed" in page
        assert 'id="confirmDialog"' in page
        assert "submitPerson" in page
        assert "prompt(" not in page
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_review_api_creates_group_from_ungrouped_faces_and_ui_feedback(tmp_path):
    Image.new("RGB", (100, 100), "red").save(tmp_path / "one.jpg")
    Image.new("RGB", (100, 100), "blue").save(tmp_path / "two.jpg")
    workspace = tmp_path / ".kiokufux"
    with FaceStore(workspace) as store:
        scan_faces(tmp_path, store, FakeBackend(), minimum_face_size=1)
        face_ids = [row[0] for row in store.db.execute("SELECT face_id FROM face_occurrences ORDER BY face_id")]
        assert store.groups() == []
    server = make_server(tmp_path, workspace)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(base + "/api/status") as response:
            collection_id = json.load(response)["collection_id"]
        payload = json.dumps({"collection_id": collection_id, "face_ids": face_ids}).encode()
        request = urllib.request.Request(base + "/api/review/create-group", data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request) as response:
            group = json.load(response)
        assert response.status == 201
        assert group["review_state"] == "needs_review"
        assert {face["face_id"] for face in group["faces"]} == set(face_ids)
        with FaceStore(workspace) as store:
            groups = store.groups()
            assert len(groups) == 1
            assert groups[0]["group_id"] == group["group_id"]
        review = json.loads((workspace / "face-review.json").read_text())
        assert review["actions"][-1]["action"] == "must-link"
        assert review["actions"][-1]["details"]["source"] == "create-group"
        with urllib.request.urlopen(base + "/") as response:
            page = response.read().decode()
        assert "Create group from selected" in page
        assert "toast" in page
        assert "saved." in page
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_confirmed_tab_lists_confirmed_people(tmp_path):
    Image.new("RGB", (100, 100), "red").save(tmp_path / "one.jpg")
    workspace = tmp_path / ".kiokufux"
    with FaceStore(workspace) as store:
        scan_faces(tmp_path, store, FakeBackend(), minimum_face_size=1)
        face_id = store.db.execute("SELECT face_id FROM face_occurrences").fetchone()[0]
    state = ReviewState(workspace)
    person = state.create_person([face_id], "Anna", friendly_name="quiet_fox")
    server = make_server(tmp_path, workspace)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(base + "/api/people") as response:
            people = json.load(response)
        assert people == [{
            "display_name": "Anna",
            "face_count": 1,
            "friendly_name": "quiet_fox",
            "person_id": person["person_id"],
            "photo_count": 1,
            "representative_face_id": face_id,
        }]
        with urllib.request.urlopen(base + "/") as response:
            page = response.read().decode()
        assert "showConfirmed" in page
        assert "/api/people" in page
        assert "Open a confirmed person to inspect their photographs or merge duplicates." in page
        assert "No confirmed people yet" in page
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_confirmed_people_detail_opens_photos_and_merges_people(tmp_path):
    Image.new("RGB", (100, 100), "red").save(tmp_path / "one.jpg")
    Image.new("RGB", (100, 100), "blue").save(tmp_path / "two.jpg")
    workspace = tmp_path / ".kiokufux"
    with FaceStore(workspace) as store:
        scan_faces(tmp_path, store, FakeBackend(), minimum_face_size=1)
        face_ids = [row[0] for row in store.db.execute("SELECT face_id FROM face_occurrences ORDER BY face_id")]
    state = ReviewState(workspace)
    source = state.create_person([face_ids[0]], "Anna", friendly_name="quiet_fox")
    target = state.create_person([face_ids[1]], None, friendly_name="calm_otter")
    server = make_server(tmp_path, workspace)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(base + f"/api/people/{source['person_id']}") as response:
            detail = json.load(response)
        assert detail["faces"][0]["face_id"] == face_ids[0]
        assert all(key in detail["faces"][0] for key in ("image_id", "x1", "y1", "x2", "y2"))
        with urllib.request.urlopen(base + "/api/status") as response:
            collection_id = json.load(response)["collection_id"]
        payload = json.dumps({"collection_id": collection_id, "source_person_id": source["person_id"], "target_person_id": target["person_id"]}).encode()
        request = urllib.request.Request(base + "/api/people/merge", data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request) as response:
            merged = json.load(response)
        assert merged["person_id"] == target["person_id"]
        assert {face["face_id"] for face in merged["faces"]} == set(face_ids)
        with urllib.request.urlopen(base + "/api/people") as response:
            people = json.load(response)
        assert [person["person_id"] for person in people] == [target["person_id"]]
        review = json.loads((workspace / "face-review.json").read_text())
        assert set(review["person_faces"][target["person_id"]]) == set(face_ids)
        assert source["person_id"] not in review["person_faces"]
        assert review["actions"][-1]["action"] == "merge-people"
        with urllib.request.urlopen(base + "/") as response:
            page = response.read().decode()
        assert "openPerson" in page
        assert "Merge confirmed people" in page
        assert "personMergeTarget" in page
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _served(workspace_root, workspace):
    server = make_server(workspace_root, workspace)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}"


def test_face_context_endpoint_levels_cache_invalid_and_edge_crop(tmp_path):
    Image.new("RGB", (320, 180), "white").save(tmp_path / "edge.jpg")
    workspace = tmp_path / ".kiokufux"
    with FaceStore(workspace) as store:
        scan_faces(tmp_path, store, FakeBackend(), minimum_face_size=1)
        face_id = store.db.execute("SELECT face_id FROM face_occurrences").fetchone()[0]
    server, thread, base = _served(tmp_path, workspace)
    try:
        sizes = {}
        for level in ("face", "person", "scene"):
            with urllib.request.urlopen(base + f"/api/faces/{face_id}/context?level={level}") as response:
                data = response.read()
            assert data.startswith(b"\xff\xd8")
            sizes[level] = Image.open(__import__('io').BytesIO(data)).size
        assert sizes["face"][0] <= 360 and sizes["person"][0] <= 900 and max(sizes["scene"]) <= 1400
        assert list((workspace / "cache" / "face-context").glob(f"*-{face_id}-face.jpg"))
        try:
            urllib.request.urlopen(base + "/api/faces/not-a-face/context?level=face")
        except urllib.error.HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError("unknown face_id accepted")
        try:
            urllib.request.urlopen(base + f"/api/faces/{face_id}/context?level=../../scene")
        except urllib.error.HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("invalid context level accepted")
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_face_context_endpoint_honors_exif_orientation(tmp_path):
    image = Image.new("RGB", (40, 120), "red")
    exif = image.getexif(); exif[274] = 6
    image.save(tmp_path / "oriented.jpg", exif=exif)
    workspace = tmp_path / ".kiokufux"
    with FaceStore(workspace) as store:
        scan_faces(tmp_path, store, FakeBackend(), minimum_face_size=1)
        face_id = store.db.execute("SELECT face_id FROM face_occurrences").fetchone()[0]
    server, thread, base = _served(tmp_path, workspace)
    try:
        with urllib.request.urlopen(base + f"/api/faces/{face_id}/context?level=scene") as response:
            size = Image.open(__import__('io').BytesIO(response.read())).size
        assert size[0] > size[1]
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_comparison_workbench_html_wiring_accessibility_and_temporary_decisions():
    assert "review-modes" in HTML
    assert "Matrix" in HTML and "Kontext" in HTML and "1:1" in HTML
    assert "Angeheftete Referenz" in HTML and "Als Referenz" in HTML
    assert "Gesicht" in HTML and "Person" in HTML and "Szene" in HTML
    assert "aria-pressed" in HTML and "aria-label=\"Comparison modes\"" in HTML
    assert "loading=\"lazy\"" in HTML
    assert "Nicht zugehörig" in HTML and "tempDecisions" in HTML
    assert "mutate('/api/review/" in HTML
    assert "markTemp" in HTML and "toggleFace(card)" in HTML
    subprocess.run(["node", "--check"], input=HTML.split("<script>", 1)[1].split("</script>", 1)[0], text=True, check=True)


def test_groups_of_two_three_and_many_have_pinned_matrix_context_and_pair_controls(tmp_path):
    for count in (2, 3, 10):
        faces = [{"face_id": f"f{i}", "image_id": f"i{i}", "confidence": .99, "quality": i / 10, "image_path": f"p{i}.jpg"} for i in range(count)]
        assert "pinnedFaceId=currentGroup.representative_face_id||bestFace(currentGroup.faces).face_id" in HTML
        assert "compareMode==='pair'" in HTML
        assert "others.slice(0,visibleLimit)" in HTML
        assert len(faces) == count
