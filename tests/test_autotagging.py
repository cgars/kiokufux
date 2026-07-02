from kiokufux.autotagging import LocalAutoTagger, propose_tags
from kiokufux.catalog import Catalog
from kiokufux.models import Photo


def test_local_auto_tagger_proposes_filename_tags(tmp_path):
    photo_path = tmp_path / "cow_at_lake.jpg"
    photo_path.write_text("not an image")
    photo = Photo("id", photo_path, photo_path.name, "hash")

    proposals = LocalAutoTagger().propose(photo)

    assert [p.tag for p in proposals] == ["cow", "lake"]
    assert all(p.confidence > 0 for p in proposals)


def test_propose_accept_and_reject_tag_workflow(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite"); catalog.init_schema()
    photo_path = tmp_path / "dog_party.jpg"; photo_path.write_text("x")
    catalog.upsert_photo(Photo("id", photo_path, photo_path.name, "hash"))

    assert propose_tags(catalog) == 2
    proposals = catalog.list_tag_proposals("id")
    assert [p.tag for p in proposals] == ["dog", "party"]

    catalog.accept_tag_proposal("id", "dog")
    catalog.reject_tag_proposal("id", "party")

    assert catalog.list_tags("id")[0].tag == "dog"
    statuses = {p.tag: p.status for p in catalog.list_tag_proposals("id", status=None)}
    assert statuses == {"dog": "accepted", "party": "rejected"}
