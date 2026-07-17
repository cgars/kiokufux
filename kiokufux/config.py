from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .default_tags import default_candidate_tags_text

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
WORKSPACE_NAME = ".kiokufux"
CATALOG_NAME = "catalog.sqlite"
CONFIG_NAME = "config.toml"


@dataclass(slots=True)
class WorkspaceConfig:
    directory: str = WORKSPACE_NAME


@dataclass(slots=True)
class ThumbnailConfig:
    max_size: int = 512


@dataclass(slots=True)
class EmbeddingConfig:
    backend: str = "auto"
    openclip_model: str | None = None
    openclip_pretrained: str | None = None


@dataclass(slots=True)
class AutoTaggingConfig:
    candidate_tags: str = ""
    top_k: int = 5
    min_score: float = 0.20


@dataclass(slots=True)
class SearchConfig:
    top_k: int = 10
    min_raw_score: float = 0.20
    min_robust_z: float = 1.0


@dataclass(slots=True)
class LoggingConfig:
    verbose: int = 0

@dataclass(slots=True)
class FacesConfig:
    device: str = "auto"
    backend: str = "facenet-pytorch"
    detection_confidence: float = 0.95
    minimum_face_size: int = 40
    working_resolution: int = 1600
    min_cluster_size: int = 2
    min_samples: int = 2
    thumbnail_size: int = 256


@dataclass(slots=True)
class KiokuFuxConfig:
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    thumbnails: ThumbnailConfig = field(default_factory=ThumbnailConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    autotagging: AutoTaggingConfig = field(default_factory=AutoTaggingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    faces: FacesConfig = field(default_factory=FacesConfig)


def workspace_for(root: Path, output_dir: Path | None = None, config: KiokuFuxConfig | None = None) -> Path:
    workspace_name = (config or KiokuFuxConfig()).workspace.directory
    return (output_dir or root / workspace_name).expanduser().resolve()


def ensure_workspace(root: Path, output_dir: Path | None = None, config: KiokuFuxConfig | None = None) -> Path:
    ws = workspace_for(root, output_dir, config)
    for name in ("thumbnails", "embeddings", "indexes", "logs"):
        (ws / name).mkdir(parents=True, exist_ok=True)
    return ws


def catalog_path(root: Path, output_dir: Path | None = None, config: KiokuFuxConfig | None = None) -> Path:
    return workspace_for(root, output_dir, config) / CATALOG_NAME


def config_path(root: Path, output_dir: Path | None = None, config: KiokuFuxConfig | None = None) -> Path:
    return workspace_for(root, output_dir, config) / CONFIG_NAME


def default_config_text() -> str:
    candidate_tags = default_candidate_tags_text()
    return f"""# KiokuFux configuration
# Values here are local-only defaults. CLI flags override these values for one run.

[workspace]
directory = ".kiokufux"

[thumbnails]
max_size = 512

[embeddings]
backend = "auto" # auto, openclip, or simple
openclip_model = "ViT-B-32"
openclip_pretrained = "laion2b_s34b_b79k"

[search]
top_k = 10
min_raw_score = 0.20
min_robust_z = 1.0

[autotagging]
candidate_tags = "{candidate_tags}"
top_k = 5
min_score = 0.20

[logging]
verbose = 0

[faces]
device = "auto" # auto, cuda, or cpu
backend = "facenet-pytorch"
detection_confidence = 0.95
minimum_face_size = 40
working_resolution = 1600
min_cluster_size = 2
min_samples = 2
thumbnail_size = 256
"""


def write_default_config(root: Path, overwrite: bool = False) -> Path:
    path = config_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not path.exists():
        path.write_text(default_config_text(), encoding="utf-8")
    return path


def load_config(root: Path) -> KiokuFuxConfig:
    path = config_path(root)
    if not path.exists():
        return KiokuFuxConfig()
    data = _load_toml(path)
    return config_from_mapping(data)


def config_from_mapping(data: dict[str, Any]) -> KiokuFuxConfig:
    cfg = KiokuFuxConfig()
    workspace = data.get("workspace", {})
    thumbnails = data.get("thumbnails", {})
    embeddings = data.get("embeddings", {})
    search = data.get("search", {})
    autotagging = data.get("autotagging", {})
    logging = data.get("logging", {})
    faces = data.get("faces", {})
    if "directory" in workspace:
        cfg.workspace.directory = str(workspace["directory"])
    if "max_size" in thumbnails:
        cfg.thumbnails.max_size = int(thumbnails["max_size"])
    if "backend" in embeddings:
        cfg.embeddings.backend = str(embeddings["backend"])
    if "openclip_model" in embeddings:
        cfg.embeddings.openclip_model = _optional_str(embeddings["openclip_model"])
    if "openclip_pretrained" in embeddings:
        cfg.embeddings.openclip_pretrained = _optional_str(embeddings["openclip_pretrained"])
    if "top_k" in search:
        cfg.search.top_k = int(search["top_k"])
    if "min_raw_score" in search:
        cfg.search.min_raw_score = float(search["min_raw_score"])
    if "min_robust_z" in search:
        cfg.search.min_robust_z = float(search["min_robust_z"])
    if "candidate_tags" in autotagging:
        value = autotagging["candidate_tags"]
        if isinstance(value, list):
            cfg.autotagging.candidate_tags = "{candidate_tags}".join(str(item) for item in value)
        else:
            cfg.autotagging.candidate_tags = str(value)
    if "top_k" in autotagging:
        cfg.autotagging.top_k = int(autotagging["top_k"])
    if "min_score" in autotagging:
        cfg.autotagging.min_score = float(autotagging["min_score"])
    if "verbose" in logging:
        cfg.logging.verbose = int(logging["verbose"])
    for name, converter in (("device",str),("backend",str),("detection_confidence",float),
        ("minimum_face_size",int),("working_resolution",int),("min_cluster_size",int),
        ("min_samples",int),("thumbnail_size",int)):
        if name in faces: setattr(cfg.faces, name, converter(faces[name]))
    return cfg


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        return _parse_simple_toml(path.read_text(encoding="utf-8"))


def _parse_simple_toml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    section: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = data.setdefault(line[1:-1].strip(), {})
            continue
        if "=" not in line or section is None:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        section[key] = _parse_scalar(value)
    return data


def _parse_scalar(value: str) -> Any:
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value
