import discovery
import pytest
from pathlib import Path
from src.discovery import FileDiscovery

def test_get_latest_file_correct_ordering(tmp_path: Path):
    d = tmp_path / "orari"
    d.mkdir()
    (d / "20260101_120000.jpg").write_text("fake")
    (d / "20260214_120000.jpg").write_text("fake") # Questo è il più recente
    (d / "20251231_120000.jpg").write_text("fake")

    discovery = FileDiscovery(d)
    result = discovery.get_latest()

    assert result.raw_name = "20260214_120000.jpg"

def test_file_not_found_raises_exception(tmp_path: Path):
    empty_dir = tmp_path / "vuota"
    empty_dir.mkdir()

    discovery = FileDiscovery(empty_dir)

    with pytest.raises(FileNotFoundError):
        discovery.get_latest()
