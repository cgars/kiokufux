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
        match_label="very good match",
    )

    line = _format_search_result(1, result, summary=True)

    assert "raw_score=0.4200" in line
    assert "match=very good match" in line
    assert "top_percent=25.0" in line
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
        match_label="very good match",
    )

    line = _format_search_result(1, result)

    assert "path=/archive/family/church.jpg" in line
    assert "thumb=/archive/.kiokufux/thumbnails/id.jpg" in line
    assert "metadata={'width': 100}" in line
