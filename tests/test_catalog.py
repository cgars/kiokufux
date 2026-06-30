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
