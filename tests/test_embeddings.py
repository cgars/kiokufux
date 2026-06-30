import sys
import types

from kiokufux.embeddings import OpenCLIPBackend, backend_from_options


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
