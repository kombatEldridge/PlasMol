"""settings.driver as a string or a dict with name + driver keys."""
import json
from argparse import Namespace

import pytest

from plasmol.utils.params import PARAMS
from plasmol.utils.params_helpers.parse_input import normalize_settings_driver


def _params(tmp_path, cfg, name="run.json"):
    path = tmp_path / name
    path.write_text(json.dumps(cfg))
    return PARAMS(Namespace(input=str(path), verbose=0, log=None, checkpoint=None))


def _absorption_cfg():
    return {
        "settings": {
            "dt": 0.05,
            "t_end": 2.0,
            "driver": {
                "name": "absorption",
                "gamma": 0.01,
                "spectrum_filepath": "spectrum.png",
                "polarization": "full",
            },
        },
        "molecule": {
            "geometry": [{"atom": "H", "coord": [0.0, 0.0, 0.0]}],
            "geometry_units": "bohr",
            "basis": "sto3g",
            "xc": "pbe0",
            "source": {"type": "kick"},
        },
        "files": {"spectra_e_vs_p_filepath": "output.png"},
    }


def test_driver_string_selects_absorption(tmp_path):
    cfg = _absorption_cfg()
    cfg["settings"]["driver"] = "absorption"
    cfg["files"]["spectra_e_vs_p_filepath"] = "spectrum.png"
    p = _params(tmp_path, cfg, "abs_str.json")
    assert p.driver_str == "absorption"
    assert p.has_absorption is True
    assert p.absorption_polarization == "full"
    assert p.absorption_spectrum_filepath == "spectrum.png"


def test_driver_dict_absorption_keys(tmp_path):
    p = _params(tmp_path, _absorption_cfg(), "abs_dict.json")
    assert p.driver_str == "absorption"
    assert p.has_absorption is True
    assert p.absorption_gamma == 0.01
    assert p.absorption_spectrum_filepath == "spectrum.png"
    assert p.absorption_polarization == "full"


def test_driver_dict_requires_name():
    with pytest.raises(ValueError, match="requires a 'name'"):
        normalize_settings_driver({
            "settings": {"dt": 0.1, "t_end": 1.0, "driver": {"gamma": 0.01}},
            "molecule": {},
        })


def test_driver_dict_unknown_key():
    with pytest.raises(ValueError, match="Unknown key"):
        normalize_settings_driver({
            "settings": {
                "driver": {"name": "absorption", "not_a_real_key": 1},
            },
        })


def test_driver_dict_tune_rejects_extra_keys():
    with pytest.raises(ValueError, match="Unknown key"):
        normalize_settings_driver({
            "settings": {"driver": {"name": "tune", "gamma": 0.01}},
        })


def test_legacy_additional_parameters_absorption_still_parses(tmp_path):
    cfg = _absorption_cfg()
    cfg["settings"]["driver"] = "absorption"
    cfg["additional_parameters"] = {
        "absorption": {
            "gamma": 0.02,
            "spectrum_filepath": "legacy.png",
        }
    }
    p = _params(tmp_path, cfg, "legacy.json")
    assert p.absorption_gamma == 0.02
    assert p.absorption_spectrum_filepath == "legacy.png"


def test_conflict_between_driver_dict_and_legacy():
    with pytest.raises(ValueError, match="Conflicting"):
        normalize_settings_driver({
            "settings": {
                "driver": {
                    "name": "absorption",
                    "gamma": 0.01,
                },
            },
            "additional_parameters": {"absorption": {"gamma": 0.02}},
        })


def test_legacy_core_hole_flat_keys(tmp_path):
    cfg = {
        "settings": {"dt": 0.2, "t_end": 1.0, "driver": "core_hole"},
        "molecule": {
            "geometry": [{"atom": "H", "coord": [0.0, 0.0, 0.0]}],
            "geometry_units": "bohr",
            "basis": "sto3g",
            "xc": "pbe",
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
    p = _params(tmp_path, cfg, "ch_legacy.json")
    assert p.driver_str == "core_hole"
    assert p.has_core_hole is False  # survey driver does not ionize
    assert p.mo_removal_index_dict == {0: 2}
    assert p.core_hole_mo_occ_filepath == "mo_occ.csv"


def test_legacy_core_hole_driver_dict_migrates_to_molecule(tmp_path):
    cfg = {
        "settings": {
            "dt": 0.2,
            "t_end": 1.0,
            "driver": {
                "name": "quantum",
                "mo_removal_index_dict": {"0": 2},
                "core_hole_mo_occ_filepath": "mo_occ.csv",
                "core_hole_watch_indices": [0, 1],
            },
        },
        "molecule": {
            "geometry": [{"atom": "H", "coord": [0.0, 0.0, 0.0]}],
            "geometry_units": "bohr",
            "basis": "sto3g",
            "xc": "pbe",
            "source": {
                "type": "kick",
                "intensity": 0.001,
                "peak_time": 0.0,
                "width_steps": 1,
                "component": "z",
            },
        },
    }
    p = _params(tmp_path, cfg, "ch_driver_legacy.json")
    assert p.driver_str == "quantum"
    assert p.has_core_hole is True
    assert p.mo_removal_index_dict == {0: 2}
    assert p.core_hole_mo_occ_filepath == "mo_occ.csv"
    assert p.core_hole_watch_indices == [0, 1]


def test_decay_stop_on_driver_dict_moves_to_additional_parameters():
    params = {
        "settings": {
            "driver": {
                "name": "np_abs_cross_sec",
                "n_flux_freqs": 20,
                "decay_stop": True,
                "decay_threshold": 1e-4,
            },
        },
    }
    normalize_settings_driver(params)
    assert params["settings"]["driver"] == {
        "name": "np_abs_cross_sec",
        "n_flux_freqs": 20,
    }
    assert params["additional_parameters"]["decay_stop"] is True
    assert params["additional_parameters"]["decay_threshold"] == 1e-4


def test_string_driver_no_extras_leaves_name_only():
    params = {"settings": {"driver": "quantum"}}
    normalize_settings_driver(params)
    assert params["settings"]["driver"] == {"name": "quantum"}


def test_comparison_driver_dict(tmp_path):
    cfg = {
        "settings": {
            "dt": 0.1,
            "t_end": 1.0,
            "driver": {
                "name": "comparison",
                "bases": ["sto3g", "6-31g"],
                "xcs": ["pbe", "pbe0"],
                "dir_name": "mo_comparison",
            },
        },
        "molecule": {
            "geometry": [{"atom": "H", "coord": [0.0, 0.0, 0.0]}],
            "geometry_units": "bohr",
            "basis": "sto3g",
            "xc": "pbe",
        },
    }
    p = _params(tmp_path, cfg, "cmp.json")
    assert p.driver_str == "comparison"
    assert p.comparison_bases == ["sto3g", "6-31g"]
    assert p.comparison_xcs == ["pbe", "pbe0"]


def test_mismatched_legacy_block_errors():
    with pytest.raises(ValueError, match="do not match"):
        normalize_settings_driver({
            "settings": {"driver": "absorption"},
            "additional_parameters": {"comparison": {"bases": ["sto3g"]}},
        })
