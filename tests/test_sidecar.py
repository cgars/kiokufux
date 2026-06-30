import json
from fotofux.models import Photo
from fotofux.sidecar import SCHEMA, sidecar_document, export_sidecars
from fotofux.catalog import Catalog


def test_sidecar_document_schema(tmp_path):
    photo = Photo("id", tmp_path / "x.jpg", "x.jpg", "hash", width=1, height=2)
    doc = sidecar_document(photo)
    assert doc["schema"] == SCHEMA
    assert doc["metadata"]["width"] == 1


def test_export_sidecars(tmp_path):
    db = Catalog(tmp_path / ".fotofux" / "catalog.sqlite"); db.init_schema()
    img = tmp_path / "x.jpg"; img.write_text("not image")
    db.upsert_photo(Photo("id", img, "x.jpg", "hash"))
    assert export_sidecars(db) == 1
    assert json.loads((tmp_path / "x.jpg.fotofux.json").read_text())["photo_id"] == "id"
