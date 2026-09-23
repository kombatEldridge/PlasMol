"""PARAMS validation for molecule.core_hole (SCH / DCH) and the survey driver."""
import json
from argparse import Namespace

import pytest

from plasmol.utils.params import PARAMS


def _molecule(**extra):
    mol = {
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
    }
    mol.update(extra)
    return mol


def _production_cfg(**extra_core_hole):
    core_hole = {
        "mo_removal_index_dict": {"0": 2},
        "mo_occ_filepath": "mo_occ.csv",
    }
    core_hole.update(extra_core_hole)
    return {
        "settings": {"dt": 0.2, "t_end": 1.0, "driver": "quantum"},
        "molecule": _molecule(core_hole=core_hole),
    }


def _survey_cfg(mo_removal=None):
    if mo_removal is None:
        mo_removal = {"0": 2, "1": 2, "2": 2}
    return {
        "settings": {"dt": 0.2, "t_end": 1.0, "driver": "core_hole"},
        "molecule": _molecule(core_hole={"mo_removal_index_dict": mo_removal}),
    }


def _params(tmp_path, cfg, name="core_hole.json"):
    p = tmp_path / name
    p.write_text(json.dumps(cfg))
    return PARAMS(Namespace(input=str(p), verbose=0, log=None, checkpoint=None))


def test_core_hole_dch_parses(tmp_path):
    p = _params(tmp_path, _production_cfg())
    assert p.driver_str == "quantum"
    assert p.has_core_hole is True
    assert p.force_open_shell is False
    assert p.mo_removal_index_dict == {0: 2}
    assert p.core_hole_mo_occ_filepath == "mo_occ.csv"
    assert p.molecule_cartesian is True
    assert p.molecule_basis_coords == "cartesian"
    assert getattr(p, "molecule_grid_level", None) is None


def test_molecule_cartesian_and_grid_level_parse(tmp_path):
    cfg = _production_cfg()
    cfg["molecule"]["cartesian"] = True
    cfg["molecule"]["grid_level"] = 5
    p = _params(tmp_path, cfg, "cart_grid.json")
    assert p.molecule_cartesian is True
    assert p.molecule_basis_coords == "cartesian"
    assert p.molecule_grid_level == 5


def test_molecule_basis_coords_cartesian(tmp_path):
    cfg = _production_cfg()
    cfg["molecule"]["basis_coords"] = "cartesian"
    p = _params(tmp_path, cfg, "basis_cart.json")
    assert p.molecule_cartesian is True
    assert p.molecule_basis_coords == "cartesian"


def test_molecule_basis_coords_sph_alias(tmp_path):
    cfg = _production_cfg()
    cfg["molecule"]["basis_coords"] = "sph"
    p = _params(tmp_path, cfg, "basis_sph.json")
    assert p.molecule_cartesian is False
    assert p.molecule_basis_coords == "spherical"


def test_molecule_basis_coords_rejects_unknown(tmp_path):
    cfg = _production_cfg()
    cfg["molecule"]["basis_coords"] = "cylindrical"
    with pytest.raises(ValueError, match="basis_coords"):
        _params(tmp_path, cfg, "bad_coords.json")


def test_molecule_basis_coords_conflicts_with_cartesian(tmp_path):
    cfg = _production_cfg()
    cfg["molecule"]["basis_coords"] = "spherical"
    cfg["molecule"]["cartesian"] = True
    with pytest.raises(ValueError, match="disagree"):
        _params(tmp_path, cfg, "coords_conflict.json")


def test_molecule_grid_level_rejects_out_of_range(tmp_path):
    cfg = _production_cfg()
    cfg["molecule"]["grid_level"] = 12
    with pytest.raises(ValueError, match="grid_level"):
        _params(tmp_path, cfg, "bad_grid.json")


def test_compound_xc_mix_parses(tmp_path):
    cfg = _production_cfg()
    cfg["molecule"]["xc"] = "0.20*HF + 0.80*PBE, PBE"
    p = _params(tmp_path, cfg, "hf20.json")
    assert p.molecule_xc == "0.20*HF + 0.80*PBE, PBE"


def test_core_hole_sch_parses(tmp_path):
    p = _params(tmp_path, _production_cfg(mo_removal_index_dict={"0": 1}))
    assert p.mo_removal_index_dict == {0: 1}
    assert p.force_open_shell is True


def test_core_hole_dual_sch_parses(tmp_path):
    p = _params(tmp_path, _production_cfg(mo_removal_index_dict={"0": 1, "1": 1}))
    assert p.mo_removal_index_dict == {0: 1, 1: 1}
    assert p.force_open_shell is True


def test_core_hole_requires_mo_dict(tmp_path):
    cfg = _production_cfg()
    del cfg["molecule"]["core_hole"]["mo_removal_index_dict"]
    with pytest.raises(ValueError, match="mo_removal_index_dict"):
        _params(tmp_path, cfg, "no_dict.json")


def test_core_hole_requires_mo_occ_filepath(tmp_path):
    cfg = _production_cfg()
    del cfg["molecule"]["core_hole"]["mo_occ_filepath"]
    with pytest.raises(ValueError, match="mo_occ_filepath"):
        _params(tmp_path, cfg, "no_occ.json")


def test_core_hole_rejects_empty_dict(tmp_path):
    cfg = _production_cfg(mo_removal_index_dict={})
    with pytest.raises(ValueError, match="non-empty dictionary"):
        _params(tmp_path, cfg, "empty.json")


def test_core_hole_rejects_bad_n_remove(tmp_path):
    cfg = _production_cfg(mo_removal_index_dict={"0": 3})
    with pytest.raises(ValueError, match="must be 1 or 2"):
        _params(tmp_path, cfg, "bad_n.json")


def test_core_hole_rejects_negative_mo_index(tmp_path):
    cfg = _production_cfg(mo_removal_index_dict={"-1": 1})
    with pytest.raises(ValueError, match="non-negative"):
        _params(tmp_path, cfg, "neg.json")


def test_core_hole_rejects_three_mos(tmp_path):
    cfg = _production_cfg(mo_removal_index_dict={"0": 1, "1": 1, "2": 1})
    with pytest.raises(ValueError, match="one or two MO indices"):
        _params(tmp_path, cfg, "three.json")


def test_core_hole_watch_indices(tmp_path):
    p = _params(
        tmp_path,
        _production_cfg(watch_indices=[0, 1]),
        "watch.json",
    )
    assert p.core_hole_watch_indices == [0, 1]


def test_core_hole_bad_watch_indices(tmp_path):
    cfg = _production_cfg(watch_indices=[])
    with pytest.raises(ValueError, match="watch_indices"):
        _params(tmp_path, cfg, "bad_watch.json")


def test_core_hole_survey_driver_allows_many_mos(tmp_path):
    p = _params(tmp_path, _survey_cfg({"0": 2, "1": 1, "2": 1}), "survey.json")
    assert p.driver_str == "core_hole"
    assert p.has_core_hole is False
    assert p.mo_removal_index_dict == {0: 2, 1: 1, 2: 1}
    assert p.force_open_shell is False


def test_core_hole_survey_does_not_require_occ_filepath(tmp_path):
    p = _params(tmp_path, _survey_cfg({"0": 2}), "survey_dch.json")
    assert p.has_core_hole is False
    assert getattr(p, "core_hole_mo_occ_filepath", None) in (None, "")


def test_core_hole_survey_requires_mo_dict(tmp_path):
    cfg = _survey_cfg()
    del cfg["molecule"]["core_hole"]
    with pytest.raises(ValueError, match="mo_removal_index_dict"):
        _params(tmp_path, cfg, "survey_no_dict.json")


def test_absorption_with_core_hole_parses(tmp_path):
    cfg = _production_cfg()
    cfg["settings"]["driver"] = {
        "name": "absorption",
        "spectrum_filepath": "spectrum.png",
        "polarization": "full",
    }
    p = _params(tmp_path, cfg, "abs_ch.json")
    assert p.has_absorption is True
    assert p.has_core_hole is True
    assert p.mo_removal_index_dict == {0: 2}


def test_absorption_copies_relocate_mo_occ(tmp_path, monkeypatch):
    from plasmol.drivers.custom_drivers.absorption.setup import (
        make_plasmol_direction_copy,
        set_up_params_copy_molecule,
    )

    monkeypatch.chdir(tmp_path)
    cfg = _production_cfg()
    cfg["settings"]["driver"] = {
        "name": "absorption",
        "spectrum_filepath": "spectrum.png",
        "polarization": "full",
    }
    p = _params(tmp_path, cfg, "abs_ch_copy.json")
    copies = set_up_params_copy_molecule(p)
    assert copies[0].core_hole_mo_occ_filepath == "x_dir/mo_occ.csv"
    hybrid = make_plasmol_direction_copy(p, "z", flat=True)
    assert hybrid.core_hole_mo_occ_filepath == "fields/mo_occ.csv"


def test_core_hole_unknown_section_key(tmp_path):
    cfg = _production_cfg()
    cfg["molecule"]["core_hole"]["not_a_key"] = True
    with pytest.raises(ValueError, match="Unknown key"):
        _params(tmp_path, cfg, "unknown_ch.json")
