"""CSV helpers used by field and MO-occupation logging."""
from pathlib import Path

from plasmol.utils.csv import init_csv, update_csv


def test_init_and_update_csv(tmp_path):
    path = tmp_path / "f.csv"
    init_csv(str(path), "unit test comment", header=["Timestamps (au)", "X Values", "Y Values", "Z Values"])
    update_csv(str(path), 0.0, 1.0, 2.0, 3.0)
    update_csv(str(path), 0.1, 1.1, 2.1, 3.1)
    text = path.read_text()
    assert "Timestamps (au)" in text
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    # blank line after comments then header + 2 data
    assert any("1.0" in ln or "1" in ln for ln in lines)
    assert len(lines) >= 3


def test_update_csv_other_columns(tmp_path):
    path = tmp_path / "mo.csv"
    init_csv(str(path), "mo occ", header=["Timestamps (au)", "MO index 0", "MO index 1"])
    update_csv(str(path), 0.0, None, None, None, 2.0, 0.1)
    lines = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    assert "2.0" in lines[-1]
