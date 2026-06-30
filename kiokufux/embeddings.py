from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from ._np import np
from .catalog import Catalog, now_iso
from .models import Embedding


class EmbeddingBackend(ABC):
    model_name = "base"
    model_version = "v1"
    dimension = 64
    @abstractmethod
    def embed_image(self, image_path: Path) -> np.ndarray: ...
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray: ...


def _norm(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32); n = float(np.linalg.norm(v)); return v / n if n else v


class FakeEmbeddingBackend(EmbeddingBackend):
    model_name = "fake"
    model_version = "test"
    dimension = 8
    def embed_image(self, image_path: Path) -> np.ndarray:
        h = hashlib.sha256(str(image_path).encode()).digest()[: self.dimension]
        return _norm(np.frombuffer(h, dtype=np.uint8).astype(np.float32))
    def embed_text(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()[: self.dimension]
        return _norm(np.frombuffer(h, dtype=np.uint8).astype(np.float32))


class SimpleLocalEmbeddingBackend(EmbeddingBackend):
    """Dependency-light local backend using color statistics plus token hashes.

    This is not CLIP quality, but keeps MVP commands local and runnable when
    open_clip_torch model weights are unavailable.
    """
    model_name = "simple-local"
    model_version = "v1"
    dimension = 64
    def embed_image(self, image_path: Path) -> np.ndarray:
        from PIL import Image, ImageOps
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB").resize((8, 8))
            arr = np.asarray(img, dtype=np.float32).reshape(-1, 3) / 255.0
        vec = np.zeros(self.dimension, dtype=np.float32)
        vec[:3] = arr.mean(axis=0); vec[3:6] = arr.std(axis=0)
        name = image_path.stem.lower()
        for token in name.replace("_", " ").replace("-", " ").split():
            vec[int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimension] += 0.25
        return _norm(vec)
    def embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        colors = {"red":0, "green":1, "blue":2, "dark":3, "bright":0, "snow":0, "lake":2, "garden":1}
        for token in text.lower().split():
            if token in colors: vec[colors[token]] += 0.5
            vec[int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimension] += 1.0
        return _norm(vec)


class OpenCLIPBackend(EmbeddingBackend):
    model_name = "openclip"
    model_version = "ViT-B-32/laion2b_s34b_b79k"
    def __init__(self) -> None:
        import open_clip, torch
        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model.eval(); self.dimension = self.model.text_projection.shape[1]
    def embed_image(self, image_path: Path) -> np.ndarray:
        from PIL import Image
        with Image.open(image_path) as image, self.torch.no_grad():
            tensor = self.preprocess(image.convert("RGB")).unsqueeze(0)
            return _norm(self.model.encode_image(tensor).cpu().numpy()[0])
    def embed_text(self, text: str) -> np.ndarray:
        with self.torch.no_grad():
            return _norm(self.model.encode_text(self.tokenizer([text])).cpu().numpy()[0])


def default_backend() -> EmbeddingBackend:
    try:
        return OpenCLIPBackend()
    except Exception:
        return SimpleLocalEmbeddingBackend()


def generate_embeddings(catalog: Catalog, workspace: Path, backend: EmbeddingBackend | None = None) -> int:
    backend = backend or default_backend(); out = workspace / "embeddings"; out.mkdir(parents=True, exist_ok=True); count = 0
    for photo in catalog.list_photos():
        target = out / f"{photo.photo_id}.{backend.model_name}.npy"
        if target.exists() and photo.embedding_status == "indexed": continue
        try:
            vec = backend.embed_image(photo.source_path); np.save(target, vec)
            catalog.upsert_embedding(Embedding(photo.photo_id, backend.model_name, backend.model_version, int(vec.shape[0]), str(target), now_iso())); count += 1
        except Exception as exc:
            catalog.conn.execute("UPDATE photos SET embedding_status='error', error=? WHERE photo_id=?", (f"embedding: {exc}", photo.photo_id)); catalog.conn.commit()
    return count
