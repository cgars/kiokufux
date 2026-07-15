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


def test_tag_review_alias_and_accept_all_parser(tmp_path):
    from kiokufux.cli import _build_parser

    review_args = _build_parser().parse_args(["tag-review", str(tmp_path)])
    assert review_args.cmd == "tag-review"
    assert review_args.photo_id is None

    accept_args = _build_parser().parse_args(["accept-tag", str(tmp_path), "--all"])
    assert accept_args.all is True
    assert accept_args.photo_id is None
    assert accept_args.tag is None

    accept_photo_args = _build_parser().parse_args(["accept-tag", str(tmp_path), "photo-id", "--all"])
    assert accept_photo_args.all is True
    assert accept_photo_args.photo_id == "photo-id"
    assert accept_photo_args.tag is None


def test_print_tag_proposals_uses_aligned_table(capsys):
    from pathlib import Path

    from kiokufux.cli import _print_tag_proposals
    from kiokufux.models import Photo, TagProposal

    rows = [
        TagProposal("abcdef123", "cat", "ai-zero-shot", 0.91, "pending", "now"),
        TagProposal("abcdef123", "pet", "ai-zero-shot", 0.81, "pending", "now"),
        TagProposal("123456789", "dog", "ai-zero-shot", 0.71, "pending", "now"),
    ]
    photos = {
        "abcdef123": Photo("abcdef123", Path("/photos/cat.jpg"), "cat.jpg", "hash1"),
        "123456789": Photo("123456789", Path("/photos/dog.jpg"), "nested/dog.jpg", "hash2"),
    }

    _print_tag_proposals(rows, photos)

    output = capsys.readouterr().out.splitlines()

    assert output[0].startswith("photo")
    assert "file" in output[0]
    assert "conf" in output[0]
    assert any("abcdef1" in line and "cat.jpg" in line and "cat" in line and "0.91" in line for line in output)
    assert any("1234567" in line and "dog.jpg" in line and "dog" in line and "0.71" in line for line in output)


def test_print_tag_proposal_summary(capsys):
    from kiokufux.cli import _print_tag_proposal_summary
    from kiokufux.models import TagProposalSummary

    _print_tag_proposal_summary([
        TagProposalSummary("garden", "ai-zero-shot", "pending", 3, 2, 0.756, 0.91),
    ])

    output = capsys.readouterr().out.splitlines()

    assert output[0].startswith("tag")
    assert "photos" in output[0]
    assert any("garden" in line and "2" in line and "3" in line and "0.76" in line for line in output)


def test_parser_accepts_tag_summary_command(tmp_path):
    from kiokufux.cli import _build_parser

    args = _build_parser().parse_args(["tag-summary", str(tmp_path), "--status", "all"])

    assert args.cmd == "tag-summary"
    assert args.status == "all"


def test_print_vocabulary(capsys):
    from kiokufux.cli import _print_vocabulary
    from kiokufux.models import TagVocabularyEntry

    _print_vocabulary([
        TagVocabularyEntry("garden", "place", "core", "accepted", "outdoor", ["yard"], "common scene", "created", "updated"),
    ])

    output = capsys.readouterr().out.splitlines()

    assert output[0].startswith("tag")
    assert "category" in output[0]
    assert any("garden" in line and "place" in line and "accepted" in line and "yard" in line for line in output)


def test_parser_accepts_vocabulary_commands(tmp_path):
    from kiokufux.cli import _build_parser

    parser = _build_parser()
    assert parser.parse_args(["vocab", str(tmp_path), "--status", "accepted"]).cmd == "vocab"
    assert parser.parse_args(["vocab-propose", str(tmp_path), "--min-photos", "2"]).min_photos == 2
    accept = parser.parse_args(["vocab-accept", str(tmp_path), "garden", "--category", "place", "--scope", "core", "--alias", "yard"])
    assert accept.cmd == "vocab-accept"
    assert accept.alias == ["yard"]
    assert parser.parse_args(["vocab-reject", str(tmp_path), "nice"]).cmd == "vocab-reject"
    merge = parser.parse_args(["vocab-merge", str(tmp_path), "backyard", "garden"])
    assert merge.alias == "backyard"
    assert merge.canonical == "garden"
    assert parser.parse_args(["vocab-apply", str(tmp_path)]).cmd == "vocab-apply"
    assert parser.parse_args(["descriptions", str(tmp_path)]).cmd == "descriptions"
    assert parser.parse_args(["vlm-descriptions", str(tmp_path), "abcdef1"]).photo_id == "abcdef1"


def test_parser_accepts_vlm_analyze_command(tmp_path):
    from kiokufux.cli import _build_parser

    args = _build_parser().parse_args(["vlm-analyze", str(tmp_path), "--vlm-backend", "ollama", "--ollama-url", "http://gaming-pc:11434", "--ollama-model", "llava:latest", "--vlm-timeout", "3", "--limit", "3", "--force"])

    assert args.cmd == "vlm-analyze"
    assert args.vlm_backend == "ollama"
    assert args.ollama_url == "http://gaming-pc:11434"
    assert args.ollama_model == "llava:latest"
    assert args.vlm_timeout == 3
    assert args.limit == 3
    assert args.force is True


def test_privacy_notice_for_vlm_analyze_is_local_only():
    from argparse import Namespace
    from kiokufux.cli import PRIVACY_LOCAL_NOTICE, _privacy_notice
    from kiokufux.config import KiokuFuxConfig

    cfg = KiokuFuxConfig()
    cfg.embeddings.backend = "auto"

    assert _privacy_notice(Namespace(cmd="vlm-analyze", embedding_backend=None), cfg) == PRIVACY_LOCAL_NOTICE


def test_print_descriptions_uses_aligned_table(capsys, tmp_path):
    from kiokufux.cli import _print_descriptions
    from kiokufux.models import Photo
    from kiokufux.vlm import ImageAnalysis

    photo = Photo("abcdef123", tmp_path / "garden.jpg", "garden.jpg", "hash")
    analysis = ImageAnalysis("abcdef123", "vlm-test", "fake", "test", caption="A garden.", description="A longer garden description with plants and a table.")

    _print_descriptions([(photo, analysis)])

    output = capsys.readouterr().out.splitlines()
    assert output[0].startswith("photo")
    assert "description" in output[0]
    assert any("abcdef1" in line and "garden.jpg" in line and "A garden." in line for line in output)


def test_privacy_notice_for_remote_ollama_warns_photos_are_sent():
    from argparse import Namespace
    from kiokufux.cli import VLM_REMOTE_NOTICE, _privacy_notice

    notice = _privacy_notice(Namespace(cmd="vlm-analyze", vlm_backend="ollama", ollama_url="http://gaming-pc:11434"))

    assert notice == VLM_REMOTE_NOTICE
    assert "photos will be sent" in notice


def test_privacy_notice_for_local_ollama_remains_local_only():
    from argparse import Namespace
    from kiokufux.cli import PRIVACY_LOCAL_NOTICE, _privacy_notice

    assert _privacy_notice(Namespace(cmd="vlm-analyze", vlm_backend="ollama", ollama_url="http://localhost:11434")) == PRIVACY_LOCAL_NOTICE


def test_parser_accepts_review_source_options(tmp_path):
    from kiokufux.cli import _build_parser

    parser = _build_parser()
    assert parser.parse_args(["vocab-apply", str(tmp_path), "--source", "vlm-fake"]).source == "vlm-fake"
    assert parser.parse_args(["accept-tag", str(tmp_path), "abcdef1", "garden", "--source", "vlm-fake"]).source == "vlm-fake"
    assert parser.parse_args(["reject-tag", str(tmp_path), "abcdef1", "garden", "--source", "vlm-fake"]).source == "vlm-fake"



def test_parser_accepts_rotate_command(tmp_path):
    from kiokufux.cli import _build_parser

    args = _build_parser().parse_args(["rotate", str(tmp_path), "abcdef1", "--degrees", "90"])
    auto_args = _build_parser().parse_args(["rotate", str(tmp_path), "abcdef1", "--auto"])
    batch_auto_args = _build_parser().parse_args(["rotate", str(tmp_path), "--auto"])

    assert args.cmd == "rotate"
    assert args.photo_id == "abcdef1"
    assert args.degrees == 90
    assert args.auto is False
    assert auto_args.auto is True
    assert auto_args.degrees is None
    assert batch_auto_args.photo_id is None
    assert batch_auto_args.auto is True
    assert batch_auto_args.vlm_fallback is False
    vlm_args = _build_parser().parse_args(["rotate", str(tmp_path), "--auto", "--vlm-fallback", "--vlm-backend", "ollama"])
    vlm_only_args = _build_parser().parse_args(["rotate", str(tmp_path), "--auto", "--vlm-only", "--vlm-verify", "--vlm-compare"])
    assert vlm_args.vlm_fallback is True
    assert vlm_args.vlm_backend == "ollama"
    assert vlm_only_args.vlm_only is True
    assert vlm_only_args.vlm_verify is True
    assert vlm_only_args.vlm_compare is True
    assert args.no_backup is False


def test_rotate_command_rotates_image_and_invalidates_catalog(tmp_path):
    from kiokufux.catalog import Catalog
    from kiokufux.cli import main
    from kiokufux.hashing import file_sha256, photo_id_for_hash
    from kiokufux.metadata import extract_metadata
    from kiokufux.models import Embedding, Photo
    from PIL import Image

    image_path = tmp_path / "wide.jpg"
    Image.new("RGB", (8, 4), "red").save(image_path)
    file_hash = file_sha256(image_path)
    photo_id = photo_id_for_hash(file_hash)
    catalog = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite")
    catalog.init_schema()
    catalog.upsert_photo(Photo(photo_id=photo_id, source_path=image_path, relative_path="wide.jpg", file_hash=file_hash, **extract_metadata(image_path)))
    catalog.set_thumbnail(photo_id, tmp_path / ".kiokufux" / "thumbnails" / f"{photo_id}.jpg")
    catalog.upsert_embedding(Embedding(photo_id, "model", "version", 2, "embeddings/x.npy", "now"))
    catalog.close()

    assert main(["rotate", str(tmp_path), photo_id[:7], "--degrees", "90"]) == 0

    with Image.open(image_path) as rotated:
        assert rotated.size == (4, 8)
    assert (tmp_path / "wide.jpg.bak").exists()

    catalog = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite")
    catalog.init_schema()
    photo = catalog.get_photo(photo_id)
    assert photo is not None
    assert photo.width == 4
    assert photo.height == 8
    assert photo.thumbnail_path is None
    assert photo.embedding_status == "pending"
    assert catalog.list_embeddings("model", "version") == []
    catalog.close()


def test_auto_rotate_command_uses_non_exif_textline_detection(tmp_path):
    from kiokufux.catalog import Catalog
    from kiokufux.cli import main
    from kiokufux.hashing import file_sha256, photo_id_for_hash
    from kiokufux.metadata import extract_metadata
    from kiokufux.models import Photo
    from PIL import Image, ImageDraw

    upright = Image.new("RGB", (120, 80), "white")
    draw = ImageDraw.Draw(upright)
    for y in (20, 32, 44, 56):
        draw.rectangle((16, y, 104, y + 3), fill="black")
    image_path = tmp_path / "text.jpg"
    upright.rotate(90, expand=True).save(image_path)
    file_hash = file_sha256(image_path)
    photo_id = photo_id_for_hash(file_hash)
    catalog = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite")
    catalog.init_schema()
    catalog.upsert_photo(Photo(photo_id=photo_id, source_path=image_path, relative_path="text.jpg", file_hash=file_hash, **extract_metadata(image_path)))
    catalog.close()

    assert main(["rotate", str(tmp_path), photo_id[:7], "--auto", "--no-backup"]) == 0

    with Image.open(image_path) as rotated:
        assert rotated.size == (120, 80)


def test_auto_rotate_command_can_use_existing_vlm_description(tmp_path, capsys):
    from kiokufux.catalog import Catalog
    from kiokufux.cli import main
    from kiokufux.hashing import file_sha256, photo_id_for_hash
    from kiokufux.metadata import extract_metadata
    from kiokufux.models import Photo
    from kiokufux.vlm import ImageAnalysis
    from PIL import Image

    image_path = tmp_path / "sideways.jpg"
    Image.new("RGB", (6, 10), "blue").save(image_path)
    file_hash = file_sha256(image_path)
    photo_id = photo_id_for_hash(file_hash)
    catalog = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite")
    catalog.init_schema()
    catalog.upsert_photo(Photo(photo_id=photo_id, source_path=image_path, relative_path="sideways.jpg", file_hash=file_hash, **extract_metadata(image_path)))
    catalog.upsert_image_analysis(ImageAnalysis(
        photo_id=photo_id,
        source="vlm-test",
        model_name="fake",
        model_version="test",
        caption="A sideways photo.",
        description="The image appears rotated counterclockwise and should be turned upright.",
    ))
    catalog.close()

    assert main(["rotate", str(tmp_path), photo_id[:7], "--auto", "--no-backup"]) == 0

    output = capsys.readouterr().out
    assert "basis=stored-vlm-description" in output
    with Image.open(image_path) as rotated:
        assert rotated.size == (10, 6)


def test_vlm_description_rotation_detection_maps_image_orientation_to_correction():
    from kiokufux.rotation import detect_clockwise_rotation_from_description

    left = detect_clockwise_rotation_from_description("The image appears rotated counterclockwise.")
    right = detect_clockwise_rotation_from_description("The image appears rotated clockwise.")
    upside_down = detect_clockwise_rotation_from_description("The photo is upside down.")

    assert left.degrees == 90
    assert right.degrees == 270
    assert upside_down.degrees == 180


def test_batch_auto_rotate_processes_all_indexed_images(tmp_path, capsys):
    from kiokufux.catalog import Catalog
    from kiokufux.cli import main
    from kiokufux.hashing import file_sha256, photo_id_for_hash
    from kiokufux.metadata import extract_metadata
    from kiokufux.models import Photo
    from kiokufux.vlm import ImageAnalysis
    from PIL import Image

    catalog = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite")
    catalog.init_schema()
    for name, description, color in [
        ("first.jpg", "The image appears rotated counterclockwise.", "green"),
        ("nested/second.jpg", "The image appears rotated clockwise.", "purple"),
    ]:
        image_path = tmp_path / name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (6, 10), color).save(image_path)
        file_hash = file_sha256(image_path)
        photo_id = photo_id_for_hash(file_hash)
        catalog.upsert_photo(Photo(
            photo_id=photo_id,
            source_path=image_path,
            relative_path=name,
            file_hash=file_hash,
            **extract_metadata(image_path),
        ))
        catalog.upsert_image_analysis(ImageAnalysis(
            photo_id=photo_id,
            source="vlm-test",
            model_name="fake",
            model_version="test",
            description=description,
        ))
    catalog.close()

    assert main(["rotate", str(tmp_path), "--auto", "--no-backup"]) == 0

    output = capsys.readouterr().out
    assert "first.jpg: decision=rotate 90° clockwise; basis=stored-vlm-description" in output
    assert "nested/second.jpg: decision=rotate 270° clockwise; basis=stored-vlm-description" in output
    assert "Auto-rotation complete: 2 rotated, 0 skipped" in output
    with Image.open(tmp_path / "first.jpg") as first, Image.open(tmp_path / "nested/second.jpg") as second:
        assert first.size == (10, 6)
        assert second.size == (10, 6)


def test_auto_rotate_vlm_fallback_runs_when_other_detection_is_uncertain(tmp_path, capsys):
    from kiokufux.catalog import Catalog
    from kiokufux.cli import main
    from kiokufux.hashing import file_sha256, photo_id_for_hash
    from kiokufux.metadata import extract_metadata
    from kiokufux.models import Photo
    from PIL import Image

    image_path = tmp_path / "rotated-counterclockwise.jpg"
    Image.new("RGB", (6, 10), "orange").save(image_path)
    file_hash = file_sha256(image_path)
    photo_id = photo_id_for_hash(file_hash)
    catalog = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite")
    catalog.init_schema()
    catalog.upsert_photo(Photo(photo_id=photo_id, source_path=image_path, relative_path=image_path.name, file_hash=file_hash, **extract_metadata(image_path)))
    catalog.close()

    assert main(["rotate", str(tmp_path), photo_id[:7], "--auto", "--vlm-fallback", "--no-backup"]) == 0

    output = capsys.readouterr().out
    assert "basis=fresh-vlm-fallback" in output
    with Image.open(image_path) as rotated:
        assert rotated.size == (10, 6)


def test_privacy_notice_for_rotate_remote_vlm_fallback_warns_photos_are_sent():
    from argparse import Namespace
    from kiokufux.cli import VLM_REMOTE_NOTICE, _privacy_notice

    notice = _privacy_notice(Namespace(cmd="rotate", vlm_fallback=True, vlm_backend="ollama", ollama_url="http://gaming-pc:11434"))

    assert notice == VLM_REMOTE_NOTICE


def test_scan_prints_progress_to_stderr(tmp_path, capsys):
    from kiokufux.cli import main
    from PIL import Image

    Image.new("RGB", (4, 4), "red").save(tmp_path / "photo.jpg")

    assert main(["scan", str(tmp_path)]) == 0

    captured = capsys.readouterr()
    assert "Scanning" in captured.err
    assert "Scanned 1 images" in captured.err
    assert "current=photo.jpg" in captured.err
    assert "Scan complete: 1 indexed/updated, 0 errors" in captured.out



def test_scan_progress_still_marks_deleted_files_missing(tmp_path):
    from kiokufux.catalog import Catalog
    from kiokufux.cli import main
    from kiokufux.config import catalog_path
    from PIL import Image

    image_path = tmp_path / "deleted.jpg"
    Image.new("RGB", (4, 4), "blue").save(image_path)

    assert main(["scan", str(tmp_path)]) == 0
    image_path.unlink()
    assert main(["scan", str(tmp_path)]) == 0

    catalog = Catalog(catalog_path(tmp_path))
    catalog.init_schema()
    photos = catalog.list_photos(include_missing=True)
    catalog.close()

    assert len(photos) == 1
    assert photos[0].relative_path == "deleted.jpg"
    assert photos[0].missing is True


def test_auto_rotate_vlm_only_skips_local_textline_heuristic(tmp_path, capsys):
    from kiokufux.catalog import Catalog
    from kiokufux.cli import main
    from kiokufux.hashing import file_sha256, photo_id_for_hash
    from kiokufux.metadata import extract_metadata
    from kiokufux.models import Photo
    from PIL import Image, ImageDraw

    upright = Image.new("RGB", (120, 80), "white")
    draw = ImageDraw.Draw(upright)
    for y in (20, 32, 44, 56):
        draw.rectangle((16, y, 104, y + 3), fill="black")
    image_path = tmp_path / "text.jpg"
    upright.rotate(90, expand=True).save(image_path)
    file_hash = file_sha256(image_path)
    photo_id = photo_id_for_hash(file_hash)
    catalog = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite")
    catalog.init_schema()
    catalog.upsert_photo(Photo(photo_id=photo_id, source_path=image_path, relative_path="text.jpg", file_hash=file_hash, **extract_metadata(image_path)))
    catalog.close()

    assert main(["rotate", str(tmp_path), photo_id[:7], "--auto", "--vlm-only", "--no-backup"]) == 0

    output = capsys.readouterr().out
    assert "basis=fresh-vlm-only" in output
    assert "No image changes made" in output
    with Image.open(image_path) as rotated:
        assert rotated.size == (80, 120)


def test_auto_rotate_vlm_only_can_rotate_from_fresh_vlm_description(tmp_path, capsys):
    from kiokufux.catalog import Catalog
    from kiokufux.cli import main
    from kiokufux.hashing import file_sha256, photo_id_for_hash
    from kiokufux.metadata import extract_metadata
    from kiokufux.models import Photo
    from PIL import Image

    image_path = tmp_path / "rotated-counterclockwise.jpg"
    Image.new("RGB", (6, 10), "yellow").save(image_path)
    file_hash = file_sha256(image_path)
    photo_id = photo_id_for_hash(file_hash)
    catalog = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite")
    catalog.init_schema()
    catalog.upsert_photo(Photo(photo_id=photo_id, source_path=image_path, relative_path=image_path.name, file_hash=file_hash, **extract_metadata(image_path)))
    catalog.close()

    assert main(["rotate", str(tmp_path), photo_id[:7], "--auto", "--vlm-only", "--no-backup"]) == 0

    output = capsys.readouterr().out
    assert "basis=fresh-vlm-only" in output
    with Image.open(image_path) as rotated:
        assert rotated.size == (10, 6)


def test_privacy_notice_for_rotate_remote_vlm_only_warns_photos_are_sent():
    from argparse import Namespace
    from kiokufux.cli import VLM_REMOTE_NOTICE, _privacy_notice

    notice = _privacy_notice(Namespace(cmd="rotate", vlm_fallback=False, vlm_only=True, vlm_backend="ollama", ollama_url="http://gaming-pc:11434"))

    assert notice == VLM_REMOTE_NOTICE


def test_vlm_only_ignores_existing_vlm_description(tmp_path, capsys):
    from kiokufux.catalog import Catalog
    from kiokufux.cli import main
    from kiokufux.hashing import file_sha256, photo_id_for_hash
    from kiokufux.metadata import extract_metadata
    from kiokufux.models import Photo
    from kiokufux.vlm import ImageAnalysis
    from PIL import Image

    image_path = tmp_path / "plain.jpg"
    Image.new("RGB", (6, 10), "cyan").save(image_path)
    file_hash = file_sha256(image_path)
    photo_id = photo_id_for_hash(file_hash)
    catalog = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite")
    catalog.init_schema()
    catalog.upsert_photo(Photo(photo_id=photo_id, source_path=image_path, relative_path=image_path.name, file_hash=file_hash, **extract_metadata(image_path)))
    catalog.upsert_image_analysis(ImageAnalysis(
        photo_id=photo_id,
        source="vlm-test",
        model_name="fake",
        model_version="test",
        description="The image appears rotated counterclockwise.",
    ))
    catalog.close()

    assert main(["rotate", str(tmp_path), photo_id[:7], "--auto", "--vlm-only", "--no-backup"]) == 0

    output = capsys.readouterr().out
    assert "basis=fresh-vlm-only" in output
    assert "decision=skip" in output
    with Image.open(image_path) as rotated:
        assert rotated.size == (6, 10)


def test_direct_vlm_rotation_response_distinguishes_action_from_appearance():
    from kiokufux.rotation import detect_clockwise_rotation_from_vlm_response

    action = detect_clockwise_rotation_from_vlm_response({"needs_rotation": True, "action": "rotate 90 degrees counterclockwise"})
    appearance = detect_clockwise_rotation_from_vlm_response({"needs_rotation": True, "orientation": "the image appears rotated 90 degrees clockwise"})
    explicit_action = detect_clockwise_rotation_from_vlm_response({"needs_rotation": True, "action_clockwise_degrees": 270})

    assert action.degrees == 270
    assert appearance.degrees == 270
    assert explicit_action.degrees == 270


def test_vlm_verify_rechecks_once_without_second_rotation(tmp_path, capsys):
    from kiokufux.catalog import Catalog
    from kiokufux.cli import main
    from kiokufux.hashing import file_sha256, photo_id_for_hash
    from kiokufux.metadata import extract_metadata
    from kiokufux.models import Photo
    from PIL import Image

    image_path = tmp_path / "rotated-counterclockwise.jpg"
    Image.new("RGB", (6, 10), "pink").save(image_path)
    file_hash = file_sha256(image_path)
    photo_id = photo_id_for_hash(file_hash)
    catalog = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite")
    catalog.init_schema()
    catalog.upsert_photo(Photo(photo_id=photo_id, source_path=image_path, relative_path=image_path.name, file_hash=file_hash, **extract_metadata(image_path)))
    catalog.close()

    assert main(["rotate", str(tmp_path), photo_id[:7], "--auto", "--vlm-only", "--vlm-verify", "--no-backup"]) == 0

    output = capsys.readouterr().out
    assert "VLM verification after rotation" in output
    assert "no further automatic rotation will be applied" in output
    with Image.open(image_path) as rotated:
        assert rotated.size == (10, 6)


def test_vlm_compare_chooses_from_candidate_contact_sheet(tmp_path, capsys):
    from kiokufux.catalog import Catalog
    from kiokufux.cli import main
    from kiokufux.hashing import file_sha256, photo_id_for_hash
    from kiokufux.metadata import extract_metadata
    from kiokufux.models import Photo
    from PIL import Image

    image_path = tmp_path / "rotated-clockwise.jpg"
    Image.new("RGB", (6, 10), "magenta").save(image_path)
    file_hash = file_sha256(image_path)
    photo_id = photo_id_for_hash(file_hash)
    catalog = Catalog(tmp_path / ".kiokufux" / "catalog.sqlite")
    catalog.init_schema()
    catalog.upsert_photo(Photo(photo_id=photo_id, source_path=image_path, relative_path=image_path.name, file_hash=file_hash, **extract_metadata(image_path)))
    catalog.close()

    assert main(["rotate", str(tmp_path), photo_id[:7], "--auto", "--vlm-only", "--vlm-compare", "--no-backup"]) == 0

    output = capsys.readouterr().out
    assert "basis=fresh-vlm-only" in output
    assert "decision=rotate 270° clockwise" in output
    with Image.open(image_path) as rotated:
        assert rotated.size == (10, 6)
