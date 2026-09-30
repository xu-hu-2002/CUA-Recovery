from pathlib import Path

from scripts.phase5.terminalize_option_a_hazards import sha256


def test_sha256_is_stable(tmp_path: Path):
    path = tmp_path / "evidence.json"
    path.write_text("evidence\n")
    assert sha256(path) == "bdcf4c994585af6dd6cb1cfbff78bcc73ab27dc30a299db5bb83766ca05b5de4"
