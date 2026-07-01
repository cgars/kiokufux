from kiokufux.config import config_from_mapping, load_config, write_default_config


def test_write_and_load_default_config(tmp_path):
    path = write_default_config(tmp_path)

    assert path.name == "config.toml"
    assert path.exists()
    cfg = load_config(tmp_path)
    assert cfg.workspace.directory == ".kiokufux"
    assert cfg.thumbnails.max_size == 512
    assert cfg.embeddings.backend == "auto"
    assert cfg.search.top_k == 10


def test_config_from_mapping_parses_aktenfuchs_style_sections():
    cfg = config_from_mapping(
        {
            "thumbnails": {"max_size": 256},
            "embeddings": {"backend": "simple", "openclip_model": "ViT-L-14", "openclip_pretrained": "weights"},
            "search": {"top_k": 5, "min_raw_score": 0.4, "min_robust_z": 2.0},
            "logging": {"verbose": 2},
        }
    )

    assert cfg.thumbnails.max_size == 256
    assert cfg.embeddings.backend == "simple"
    assert cfg.embeddings.openclip_model == "ViT-L-14"
    assert cfg.search.top_k == 5
    assert cfg.search.min_raw_score == 0.4
    assert cfg.search.min_robust_z == 2.0
    assert cfg.logging.verbose == 2
