from pathlib import Path

from kiokufux.cli import _format_search_result
from kiokufux.models import SearchResult


def test_format_search_result_summary_only_includes_stats_and_name():
    result = SearchResult(
        photo_id="id",
        score=0.42,
        source_path=Path("/archive/family/church.jpg"),
        thumbnail_path="/archive/.kiokufux/thumbnails/id.jpg",
        metadata_summary={"width": 100},
        rank=1,
        total_ranked=4,
        top_percent=25.0,
        normalized_score=1.0,
        robust_z_score=2.5,
        confidence_gate_passed=True,
        match_label="very good match",
    )

    line = _format_search_result(1, result, summary=True)

    assert "raw_score=0.4200" in line
    assert "match=very good match" in line
    assert "top_percent=25.0" in line
    assert "robust_z=2.50" in line
    assert "normalized_relative=1.0000" in line
    assert "name=church.jpg" in line
    assert "path=" not in line
    assert "thumb=" not in line
    assert "metadata=" not in line


def test_format_search_result_full_includes_path_thumb_and_metadata():
    result = SearchResult(
        photo_id="id",
        score=0.42,
        source_path=Path("/archive/family/church.jpg"),
        thumbnail_path="/archive/.kiokufux/thumbnails/id.jpg",
        metadata_summary={"width": 100},
        top_percent=25.0,
        normalized_score=1.0,
        robust_z_score=2.5,
        confidence_gate_passed=True,
        match_label="very good match",
    )

    line = _format_search_result(1, result)

    assert "path=/archive/family/church.jpg" in line
    assert "thumb=/archive/.kiokufux/thumbnails/id.jpg" in line
    assert "metadata={'width': 100}" in line


def test_privacy_notice_for_local_only_command():
    from argparse import Namespace
    from kiokufux.cli import PRIVACY_LOCAL_NOTICE, _privacy_notice

    assert _privacy_notice(Namespace(cmd="scan")) == PRIVACY_LOCAL_NOTICE


def test_privacy_notice_for_openclip_backend_mentions_weight_download_only():
    from argparse import Namespace
    from kiokufux.cli import OPENCLIP_DOWNLOAD_NOTICE, _privacy_notice

    notice = _privacy_notice(Namespace(cmd="embed", embedding_backend="openclip"))

    assert notice == OPENCLIP_DOWNLOAD_NOTICE
    assert "no photo, metadata, or query data will be sent" in notice
    assert "download model weights" in notice


def test_parser_accepts_verbose_flag_before_command(tmp_path):
    from kiokufux.cli import _build_parser

    args = _build_parser().parse_args(["-v", "scan", str(tmp_path)])

    assert args.verbose == 1
    assert args.cmd == "scan"


def test_extract_verbose_args_accepts_verbose_after_subcommand(tmp_path):
    from kiokufux.cli import _build_parser, _extract_verbose_args

    cleaned, verbose = _extract_verbose_args(["scan", str(tmp_path), "-v"])
    args = _build_parser().parse_args(cleaned)

    assert verbose == 1
    assert args.cmd == "scan"


def test_extract_verbose_args_accepts_compact_debug_flag_after_options(tmp_path):
    from kiokufux.cli import _build_parser, _extract_verbose_args

    cleaned, verbose = _extract_verbose_args(["search", str(tmp_path), "church", "--summary", "-vv"])
    args = _build_parser().parse_args(cleaned)

    assert verbose == 2
    assert args.cmd == "search"
    assert args.summary is True


def test_privacy_notice_uses_configured_embedding_backend():
    from argparse import Namespace
    from kiokufux.cli import OPENCLIP_DOWNLOAD_NOTICE, PRIVACY_LOCAL_NOTICE, _privacy_notice
    from kiokufux.config import KiokuFuxConfig

    cfg = KiokuFuxConfig()
    cfg.embeddings.backend = "auto"
    assert _privacy_notice(Namespace(cmd="init", embedding_backend=None), cfg) == PRIVACY_LOCAL_NOTICE
    assert _privacy_notice(Namespace(cmd="embed", embedding_backend=None), cfg) == OPENCLIP_DOWNLOAD_NOTICE
    cfg.embeddings.backend = "simple"
    assert _privacy_notice(Namespace(cmd="embed", embedding_backend=None), cfg) == PRIVACY_LOCAL_NOTICE
