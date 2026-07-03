from kiokufux.catalog import Catalog
from kiokufux.models import Photo


def test_catalog_insert_update_and_missing(tmp_path):
    c = Catalog(tmp_path / "catalog.sqlite"); c.init_schema()
    p = Photo("id1", tmp_path / "a.jpg", "a.jpg", "hash1", width=10)
    c.upsert_photo(p)
    assert c.get_photo("id1").width == 10
    c.upsert_photo(Photo("id1", tmp_path / "a.jpg", "a.jpg", "hash1", width=20))
    assert c.get_photo("id1").width == 20
    c.mark_missing_except([])
    assert c.get_photo("id1").missing is True


def test_catalog_photo_tags_are_normalized_and_removable(tmp_path):
    c = Catalog(tmp_path / "catalog.sqlite"); c.init_schema()
    c.upsert_photo(Photo("id1", tmp_path / "a.jpg", "a.jpg", "hash1"))

    c.add_tag("id1", "  Family  Party ")
    c.add_tag("id1", "family party")

    tags = c.list_tags("id1")
    assert len(tags) == 1
    assert tags[0].tag == "family party"
    assert tags[0].source == "manual"

    c.remove_tag("id1", "family party")
    assert c.list_tags("id1") == []


def test_catalog_resolves_full_or_seven_character_photo_ids(tmp_path):
    c = Catalog(tmp_path / "c.sqlite"); c.init_schema()
    photo_path = tmp_path / "p.jpg"; photo_path.write_text("x")
    c.upsert_photo(Photo("abcdef123456", photo_path, photo_path.name, "hash"))

    assert c.resolve_photo_id("abcdef1") == "abcdef123456"
    assert c.resolve_photo_id("abcdef123456") == "abcdef123456"
