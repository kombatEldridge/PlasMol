"""Sudden core-hole construction and MO occupation logging."""
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from pyscf import dft

from plasmol.utils.params import PARAMS
from plasmol.quantum.molecule import MOLECULE
from plasmol.quantum.propagation import propagation


def _h2_core_hole_params(tmp_path, mo_removal, occ_name="mo_occ.csv"):
    cfg = {
        "settings": {"dt": 0.2, "t_end": 0.4, "driver": "core_hole"},
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
                "intensity": 1e-4,
                "peak_time": 0.0,
                "width_steps": 1,
                "component": "z",
            },
        },
        "additional_parameters": {
            "mo_removal_index_dict": mo_removal,
            "core_hole_mo_occ_filepath": str(tmp_path / occ_name),
        },
        "files": {
            "field_e_filepath": str(tmp_path / "e.csv"),
            "field_p_filepath": str(tmp_path / "p.csv"),
        },
    }
    jp = tmp_path / "ch.json"
    jp.write_text(json.dumps(cfg))
    return PARAMS(Namespace(input=str(jp), verbose=0, log=None, checkpoint=None))


def test_dch_forces_uks_and_charge(tmp_path):
    params = _h2_core_hole_params(tmp_path, {"0": 2})
    mol = MOLECULE(params)
    assert isinstance(mol.mf, dft.uks.UKS)
    assert mol.is_open_shell is True
    assert mol.mf.mol.charge == 2
    # double hole on one MO keeps closed-shell spin for parent spin=0
    assert mol.mf.mol.spin == 0
    assert getattr(mol, "_core_hole_dm0", None) is not None
    # both spin channels emptied on MO 0
    occ = np.asarray(mol.mf.mo_occ)
    assert occ[0][0] == 0
    assert occ[1][0] == 0


def test_sch_increments_spin(tmp_path):
    params = _h2_core_hole_params(tmp_path, {"0": 1}, "sch.csv")
    mol = MOLECULE(params)
    assert mol.mf.mol.charge == 1
    assert mol.mf.mol.spin == 1
    occ = np.asarray(mol.mf.mo_occ)
    assert occ[0][0] == 0
    assert occ[1][0] > 0  # beta still occupied for closed-shell parent


def test_out_of_range_mo_raises(tmp_path):
    params = _h2_core_hole_params(tmp_path, {"0": 2})
    mol = MOLECULE(params)
    nmo = np.asarray(mol.mf.mo_coeff).shape[-1]
    with pytest.raises(ValueError, match="out of range"):
        mol.remove_core_electrons({nmo + 5: 1})


def test_mo_occ_csv_initialized_and_logged(tmp_path):
    params = _h2_core_hole_params(tmp_path, {"0": 2})
    mol = MOLECULE(params)
    path = Path(params.core_hole_mo_occ_filepath)
    assert path.exists()
    text = path.read_text()
    assert "Timestamps" in text
    assert "MO index 0" in text

    # one propagation step should append a row
    from plasmol.quantum.propagators.rk4 import propagate as propagate_rk4

    prop_params = dict(params.molecule_propagator_params)
    prop_params["current_time"] = 0.0
    prop_params["has_core_hole"] = True
    mu = propagation(prop_params, mol, [0.0, 0.0, 0.0], propagate_rk4)
    assert mu.shape == (3,)
    lines = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    assert len(lines) >= 2  # header + at least one data row


def test_get_mo_occupations_hole_on_target(tmp_path):
    params = _h2_core_hole_params(tmp_path, {"0": 2}, "hole.csv")
    mol = MOLECULE(params)
    # after sudden DCH on MO 0, hole occupation (n0 - n_e) on MO 0 is ~2
    mol.get_mo_occupations(0.0)
    path = Path(params.core_hole_mo_occ_filepath)
    rows = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    # last data row
    data = rows[-1].split(",")
    hole0 = float(data[1])
    assert hole0 == pytest.approx(2.0, abs=0.2)
