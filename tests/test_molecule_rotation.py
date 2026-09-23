"""Nuclear-frame rotation applied before SCF."""
import json
from argparse import Namespace

import numpy as np
import pytest

from plasmol.drivers.quantum import run as run_quantum
from plasmol.quantum.geometry import rotate_molecule_geometry
from plasmol.utils.params import PARAMS


def _coords(coords_str):
    rows = []
    for block in coords_str.split(";"):
        parts = block.split()
        rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(rows)


def _namespace(coords, rotation):
    return Namespace(
        molecule_atoms=["H", "H"],
        molecule_coords=coords,
        molecule_rotation=rotation,
    )


def test_axis_rotation_about_charge_center(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = _namespace(
        "H 1.0 0.0 0.0; H -1.0 0.0 0.0",
        {"axis": "z", "angle_deg": 90},
    )
    rotate_molecule_geometry(params)
    got = _coords(params.molecule_coords)
    assert got[0] == pytest.approx([0.0, 1.0, 0.0])
    assert got[1] == pytest.approx([0.0, -1.0, 0.0])
    text = (tmp_path / "rotated_geometry.xyz").read_text()
    assert text.startswith("2\n")
    assert "units: bohr" in text.splitlines()[1]
    assert params.molecule_rotation_applied is True


def test_rotation_is_applied_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = _namespace(
        "H 1.0 0.0 0.0; H -1.0 0.0 0.0",
        [{"axis": "z", "angle_deg": 90}, {"axis": "z", "angle_deg": 90}],
    )
    rotate_molecule_geometry(params)
    once = params.molecule_coords
    rotate_molecule_geometry(params)
    assert params.molecule_coords == once
    assert _coords(once)[0] == pytest.approx([-1.0, 0.0, 0.0])


def test_align_bond_then_twist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = _namespace(
        "H 0.0 0.0 0.0; H 0.0 2.0 0.0",
        {"align": {"from_atom": 0, "to_atom": 1, "axis": "x"}, "twist_deg": 0},
    )
    rotate_molecule_geometry(params)
    got = _coords(params.molecule_coords)
    bond = got[1] - got[0]
    assert bond == pytest.approx([2.0, 0.0, 0.0])
    assert got.mean(axis=0) == pytest.approx([0.0, 1.0, 0.0])


def test_missing_rotation_does_not_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = Namespace(molecule_atoms=["H"], molecule_coords="H 0.0 0.0 0.0")
    rotate_molecule_geometry(params)
    assert params.molecule_coords == "H 0.0 0.0 0.0"
    assert not (tmp_path / "rotated_geometry.xyz").exists()


def test_align_index_out_of_range(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = _namespace(
        "H 0.0 0.0 0.0; H 1.0 0.0 0.0",
        {"align": {"from_atom": 0, "to_atom": 5, "axis": "x"}},
    )
    with pytest.raises(ValueError, match="out of range"):
        rotate_molecule_geometry(params)


def _params(tmp_path, rotation, name="rot.json"):
    payload = {
        "settings": {"dt": 0.5, "t_end": 5.0, "driver": "quantum"},
        "molecule": {
            "geometry": [
                {"atom": "H", "coord": [0.0, 0.0, 0.0]},
                {"atom": "H", "coord": [0.0, 0.0, 1.4]},
            ],
            "geometry_units": "bohr",
            "charge": 0,
            "spin": 0,
            "basis": "sto3g",
            "xc": "pbe",
            "propagator": {"type": "rk4"},
            "source": {
                "type": "kick",
                "intensity": 0.001,
                "peak_time": 0.0,
                "width_steps": 1,
                "component": "z",
            },
            "rotation": rotation,
        },
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return PARAMS(Namespace(input=str(path), verbose=0, log=None, checkpoint=None))


def test_rotation_is_validated_but_not_applied_at_parse(tmp_path):
    params = _params(tmp_path, {"axis": "x", "angle_deg": 180})
    assert "0.0 0.0 1.4" in params.molecule_coords
    assert not getattr(params, "molecule_rotation_applied", False)


def test_bad_rotation_rejected(tmp_path):
    with pytest.raises(ValueError, match="cannot mix"):
        _params(tmp_path, {"axis": "z", "angle_deg": 10, "align": {"from_atom": 0, "to_atom": 1, "axis": "x"}})
    with pytest.raises(ValueError, match="axis"):
        _params(tmp_path, {"axis": "q", "angle_deg": 10})


def test_quantum_run_rotates_before_molecule(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = _params(tmp_path, {"axis": "x", "angle_deg": 180})
    seen = {}

    def fake_molecule(p):
        seen["coords"] = p.molecule_coords
        raise RuntimeError("stop-after-geometry")

    monkeypatch.setattr("plasmol.drivers.quantum.init_csv", lambda *a, **k: None)
    monkeypatch.setattr("plasmol.drivers.quantum.update_csv", lambda *a, **k: None)
    monkeypatch.setattr("plasmol.drivers.quantum.MOLECULE", fake_molecule)
    with pytest.raises(RuntimeError, match="stop-after-geometry"):
        run_quantum(params)
    got = _coords(seen["coords"])
    assert got[0, 2] == pytest.approx(1.4)
    assert got[1, 2] == pytest.approx(0.0)
    assert (tmp_path / "rotated_geometry.xyz").is_file()
