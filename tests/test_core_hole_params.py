"""PARAMS validation for driver: core_hole (SCH / DCH modes)."""
import copy
import json
from argparse import Namespace

import pytest

from plasmol.utils.params import PARAMS


def _base_core_hole_cfg(**extra_add):
    cfg = {
        "settings": {"dt": 0.2, "t_end": 1.0, "driver": "core_hole"},
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
        },
        "additional_parameters": {
            "mo_removal_index_dict": {"0": 2},
            "core_hole_mo_occ_filepath": "mo_occ.csv",
        },
    }
    cfg["additional_parameters"].update(extra_add)
    return cfg


def _params(tmp_path, cfg, name="core_hole.json"):
    p = tmp_path / name
    p.write_text(json.dumps(cfg))
    return PARAMS(Namespace(input=str(p), verbose=0, log=None, checkpoint=None))


def test_core_hole_dch_parses(tmp_path):
    p = _params(tmp_path, _base_core_hole_cfg())
    assert p.driver_str == "core_hole"
    assert p.has_core_hole is True
    assert p.force_open_shell is True
    assert p.mo_removal_index_dict == {0: 2}
    assert p.core_hole_mo_occ_filepath == "mo_occ.csv"


def test_core_hole_sch_parses(tmp_path):
    p = _params(tmp_path, _base_core_hole_cfg(mo_removal_index_dict={"0": 1}))
    assert p.mo_removal_index_dict == {0: 1}


def test_core_hole_dual_sch_parses(tmp_path):
    p = _params(tmp_path, _base_core_hole_cfg(mo_removal_index_dict={"0": 1, "1": 1}))
    assert p.mo_removal_index_dict == {0: 1, 1: 1}


def test_core_hole_requires_mo_dict(tmp_path):
    cfg = _base_core_hole_cfg()
    del cfg["additional_parameters"]["mo_removal_index_dict"]
    with pytest.raises(ValueError, match="mo_removal_index_dict"):
        _params(tmp_path, cfg, "no_dict.json")


def test_core_hole_requires_mo_occ_filepath(tmp_path):
    cfg = _base_core_hole_cfg()
    del cfg["additional_parameters"]["core_hole_mo_occ_filepath"]
    with pytest.raises(ValueError, match="core_hole_mo_occ_filepath"):
        _params(tmp_path, cfg, "no_occ.json")


def test_core_hole_rejects_empty_dict(tmp_path):
    cfg = _base_core_hole_cfg(mo_removal_index_dict={})
    with pytest.raises(ValueError, match="non-empty dictionary"):
        _params(tmp_path, cfg, "empty.json")


def test_core_hole_rejects_bad_n_remove(tmp_path):
    cfg = _base_core_hole_cfg(mo_removal_index_dict={"0": 3})
    with pytest.raises(ValueError, match="must be 1 or 2"):
        _params(tmp_path, cfg, "bad_n.json")


def test_core_hole_rejects_negative_mo_index(tmp_path):
    cfg = _base_core_hole_cfg(mo_removal_index_dict={"-1": 1})
    with pytest.raises(ValueError, match="non-negative"):
        _params(tmp_path, cfg, "neg.json")


def test_core_hole_rejects_three_mos(tmp_path):
    cfg = _base_core_hole_cfg(mo_removal_index_dict={"0": 1, "1": 1, "2": 1})
    with pytest.raises(ValueError, match="one or two MO indices"):
        _params(tmp_path, cfg, "three.json")


def test_core_hole_watch_indices(tmp_path):
    p = _params(
        tmp_path,
        _base_core_hole_cfg(core_hole_watch_indices=[0, 1]),
        "watch.json",
    )
    assert p.core_hole_watch_indices == [0, 1]


def test_core_hole_bad_watch_indices(tmp_path):
    cfg = _base_core_hole_cfg(core_hole_watch_indices=[])
    with pytest.raises(ValueError, match="core_hole_watch_indices"):
        _params(tmp_path, cfg, "bad_watch.json")


def test_core_hole_survey_mode_allows_dict_only(tmp_path):
    cfg = _base_core_hole_cfg(
        check_mo_contrib_by_atom=True,
        mo_removal_index_dict={"0": 2, "1": 1, "2": 1},
    )
    # survey mode skips hole-count constraints
    p = _params(tmp_path, cfg, "survey.json")
    assert p.check_mo_contrib_by_atom is True
    assert p.has_core_hole is True
