from fotofux.hashing import file_sha256, photo_id_for_hash


def test_file_sha256_and_stable_id(tmp_path):
    p = tmp_path / "a.txt"; p.write_text("hello")
    h = file_sha256(p)
    assert h == file_sha256(p)
    assert photo_id_for_hash(h) == h[:32]
