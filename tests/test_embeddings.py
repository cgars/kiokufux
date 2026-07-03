import sys
import types
from pathlib import Path

from kiokufux.catalog import Catalog
from kiokufux.embeddings import FakeEmbeddingBackend, OpenCLIPBackend, backend_from_options, generate_embeddings
from kiokufux.models import Photo


class _FakeProjection:
    shape = (1, 16)


class _FakeModel:
    text_projection = _FakeProjection()

    def eval(self):
        return None


def test_openclip_backend_uses_configured_model(monkeypatch):
    calls = {}

    def create_model_and_transforms(model, pretrained):
        calls["create"] = (model, pretrained)
        return _FakeModel(), None, None

    def get_tokenizer(model):
        calls["tokenizer"] = model
        return lambda text: text

    fake_open_clip = types.SimpleNamespace(
        create_model_and_transforms=create_model_and_transforms,
        get_tokenizer=get_tokenizer,
    )
    fake_torch = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    backend = OpenCLIPBackend(model="ViT-L-14", pretrained="datacomp_xl_s13b_b90k")

    assert calls["create"] == ("ViT-L-14", "datacomp_xl_s13b_b90k")
    assert calls["tokenizer"] == "ViT-L-14"
    assert backend.model_version == "ViT-L-14/datacomp_xl_s13b_b90k"


def test_backend_from_options_can_force_simple():
    backend = backend_from_options("simple", openclip_model="ignored", openclip_pretrained="ignored")
    assert backend.model_name == "simple-local"


def test_generate_embeddings_stores_workspace_relative_paths(tmp_path):
    workspace = tmp_path / ".kiokufux"
    catalog = Catalog(workspace / "catalog.sqlite"); catalog.init_schema()
    image_path = tmp_path / "image.jpg"; image_path.write_text("x")
    catalog.upsert_photo(Photo("abcdef123", image_path, image_path.name, "hash"))
    backend = FakeEmbeddingBackend()

    assert generate_embeddings(catalog, workspace, backend) == 1

    [embedding] = catalog.list_embeddings(backend.model_name, backend.model_version)
    assert embedding.embedding_path == "embeddings/abcdef123.fake.npy"
    assert not Path(embedding.embedding_path).is_absolute()
    assert catalog.artifact_path(embedding.embedding_path).exists()
