# tests/test_params.py
import copy
import json
import pytest
import numpy as np
from argparse import Namespace
from plasmol.utils.input.params import PARAMS
from plasmol.quantum.electric_field import ELECTRICFIELD

def test_minimal_json_parses_cleanly(minimal_params):
    """Basic sanity check on our minimal test JSON."""
    p = minimal_params
    assert p.dt == 0.5
    assert p.t_end == 5.0
    assert p.has_molecule is True
    assert p.has_plasmon is False
    assert p.run_molecule_simulation is True
    assert p.molecule_propagator_str == "rk4"
    assert p.molecule_source_type == "kick"
    assert p.molecule_source_component == "z"
    print("Minimal JSON parsed successfully!")


def test_minimal_params_has_required_attributes(minimal_params):
    """Make sure the most important objects were created."""
    p = minimal_params
    assert hasattr(p, "times")
    assert hasattr(p, "molecule_source_field")
    assert len(p.times) > 0
    assert p.molecule_source_field.shape == (len(p.times), 3)


def test_plasmon_only_parses_cleanly(plasmon_only_args):
    """Test that PARAMS correctly handles a plasmon-only simulation (no molecule)."""
    p = PARAMS(plasmon_only_args)
    assert p.has_plasmon is True
    assert p.run_plasmon_simulation is True
    assert p.run_molecule_simulation is False
    assert p.has_molecule is False
    assert p.plasmon_cell_length == 0.2
    assert p.plasmon_pml_thickness == 0.02
    assert p.plasmon_surrounding_material_index == 1.0
    assert p.plasmon_source_type == "gaussian"
    assert hasattr(p, "plasmon_source_object")
    assert p.cell_volume is not None
    assert p.plasmon_symmetries is not None
    print("Plasmon-only config parsed and validated successfully!")


def test_geometry_construction_angstrom_conversion(tmp_path):
    """Test _construct_geometry: Angstrom → Bohr conversion + correct coords string format."""
    json_content = {
        "settings": {"dt": 0.5, "t_end": 5.0},
        "molecule": {
            "geometry": [
                {"atom": "O", "coord": [0.0, 0.0, 0.0]},
                {"atom": "H", "coord": [0.0, 0.0, 1.0]}
            ],
            "geometry_units": "angstrom",
            "charge": 0,
            "spin": 0,
            "basis": "sto3g",
            "xc": "pbe",
            "source": {"type": "kick", "intensity": 0.001, "peak_time": 0.0, "width_steps": 1, "component": "z"},
            "files": {"field_e_filepath": "e.csv", "field_p_filepath": "p.csv", "spectra_e_vs_p_filepath": "spec.png"}
        }
    }
    json_path = tmp_path / "geo_angstrom.json"
    with open(json_path, "w") as f:
        json.dump(json_content, f)

    args = Namespace(input=str(json_path), verbose=0, log=None, checkpoint=None)
    params = PARAMS(args)

    assert params.molecule_atoms == ["O", "H"]
    assert params.molecule_geometry_units == "bohr"
    assert "1.889" in params.molecule_coords
    assert "O 0.0 0.0 0.0; H" in params.molecule_coords
    print("Geometry conversion (Angstrom → Bohr) successful")


def test_electricfield_pulse_and_kick():
    """Test ELECTRICFIELD for both 'pulse' and 'kick'."""
    times = np.linspace(0, 40, 401)
    # Pulse
    params_pulse = Namespace(
        times=times, dt=0.1, molecule_source_type="pulse",
        molecule_source_intensity=0.01, molecule_source_peak_time=10.0,
        molecule_source_width_steps=50, molecule_source_component="z",
        molecule_source_additional_parameters={"wavelength": 0.8}
    )
    ef = ELECTRICFIELD(params_pulse)
    assert ef.field.shape == (len(times), 3)
    assert np.max(np.abs(ef.field[:, 2])) > 0.005

    # Kick
    params_kick = Namespace(
        times=times, dt=0.1, molecule_source_type="kick",
        molecule_source_intensity=0.05, molecule_source_peak_time=5.0,
        molecule_source_width_steps=1, molecule_source_component="x"
    )
    ef = ELECTRICFIELD(params_kick)
    assert abs(ef.field[np.argmin(np.abs(times - 5.0)), 0] - 0.05) < 1e-10
    print("ELECTRICFIELD pulse/kick generation correct")


# ====================== TESTS FOR EVERY PROTECTION IN _attribute_checks() ======================

BASE_CONFIG = {
    "settings": {
        "dt": 0.5,
        "t_end": 5.0
    },
    "molecule": {
        "geometry": [{"atom": "H", "coord": [0, 0, 0]}],
        "geometry_units": "bohr",
        "charge": 0,
        "spin": 0,
        "basis": "sto3g",
        "xc": "pbe",
        "source": {
            "type": "kick",
            "intensity": 0.001,
            "peak_time": 0,
            "width_steps": 1,
            "component": "z"
        },
        "files": {
            "field_e_filepath": "e.csv",
            "field_p_filepath": "p.csv",
            "spectra_e_vs_p_filepath": "spec.png"
        }
    },
    "plasmon": {
        "simulation": {
            "tolerance_field_e": 1e-12,
            "cell_length": 0.2,
            "pml_thickness": 0.02,
            "surrounding_material_index": 1.0
        },
        "molecule_position": [0,0,0]
    }
}

def make_bad_config(tmp_path, config_dict, filename="bad_config.json"):
    """Helper: write the (possibly modified) config and return ready-to-use args."""
    json_path = tmp_path / filename
    with open(json_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    return Namespace(input=str(json_path), verbose=0, log=None, checkpoint=None)


def _test_validation_error(tmp_path, modify_func, exc_type, match, filename=None):
    """One helper = zero duplication. 
    modify_func receives the full config dict and mutates it in place."""
    bad = copy.deepcopy(BASE_CONFIG)
    modify_func(bad)
    json_name = filename or "bad_config.json"
    args = make_bad_config(tmp_path, bad, json_name)
    with pytest.raises(exc_type, match=match):
        PARAMS(args)


def test_missing_dt_raises(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["settings"].pop("dt"), ValueError, "Missing required parameter: 'dt'")


def test_missing_t_end_raises(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["settings"].pop("t_end"), ValueError, "Missing required parameter: 't_end'")


def test_negative_dt_raises(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["settings"].update({"dt": -0.1}), RuntimeError, "'dt' must be a positive value")


def test_negative_t_end_raises(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["settings"].update({"t_end": -5.0}), RuntimeError, "'t_end' must be a positive value")


def test_dt_greater_than_t_end_raises(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["settings"].update({"dt": 10.0}), RuntimeError, "'dt' cannot be larger than 't_end'")


def test_t_end_not_multiple_of_dt_raises(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["settings"].update({"dt": 0.3}), RuntimeError, "'t_end' must be a multiple of 'dt'")


def test_plasmon_tolerance_non_positive(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["plasmon"]["simulation"].update({"tolerance_field_e": -1e-12}), ValueError, "tolerance_field_e' must be a positive value")


def test_plasmon_cell_length_non_positive(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["plasmon"]["simulation"].update({"cell_length": -0.2}), ValueError, "cell_length' must be a positive value")


def test_plasmon_pml_thickness_non_positive(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["plasmon"]["simulation"].update({"pml_thickness": -0.02}), ValueError, "pml_thickness' must be a positive value")


def test_plasmon_surrounding_index_below_1(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["plasmon"]["simulation"].update({"surrounding_material_index": 0.9}), ValueError, "surrounding_material_index' must be >= 1.0")


def test_plasmon_symmetries_odd_length(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["plasmon"]["simulation"].update({"symmetries": ["Y", 1, "Z"]}), ValueError, "Invalid plasmon symmetry")


def test_plasmon_source_missing_required_fields(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["plasmon"].update({"source": {"type": "gaussian"}}), ValueError, "Source requires")


def test_molecule_missing_required_fields(tmp_path):
    # special case — completely different base config
    bad = {"settings": {"dt": 0.5, "t_end": 5.0}, "molecule": {"geometry": [{"atom": "H", "coord": [0, 0, 0]}]}}
    args = make_bad_config(tmp_path, bad, "missing_mol_fields.json")
    with pytest.raises(ValueError, match="Molecule requires"):
        PARAMS(args)


def test_invalid_molecule_propagator(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["molecule"].update({"propagator": {"type": "invalid"}}), ValueError, "Unsupported propagator")


def test_invalid_molecule_geometry_units(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["molecule"].update({"geometry_units": "invalid"}), ValueError, "Invalid 'molecule_geometry_units'")


def test_fourier_with_plasmon(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["molecule"].update({"modifiers": {"fourier": {}}}), ValueError, "Fourier modifier cannot be used with plasmon modifier")


def test_fourier_missing_spectrum_filepath(tmp_path):
    _test_validation_error(tmp_path,
        lambda c: (c.pop("plasmon", None), c["molecule"].update({"modifiers": {"fourier": {}}})),
        ValueError, "Fourier modifier requires 'spectrum_filepath' attribute")


def test_checkpoint_missing_filepath(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["molecule"].update({"files": {"checkpoint": {"frequency": 100}}}), ValueError, "Checkpointing requires 'filepath'")


def test_field_filepath_not_string(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["molecule"]["files"].update({"field_e_filepath": ""}), ValueError, "Filepath for 'field_e_filepath' must be a non-empty string")


def test_plasmon_molecule_position_wrong_length(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["plasmon"].update({"molecule_position": [0, 0]}), ValueError, "Molecule position must be an array of three numbers")


def test_plasmon_molecule_position_non_numeric(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["plasmon"].update({"molecule_position": [0, 0, "nan"]}), ValueError, "Invalid molecule position")


def test_plasmon_molecule_position_without_molecule(tmp_path):
    _test_validation_error(tmp_path, lambda c: c.pop("molecule", None), ValueError, "Molecule position properly specified .* but without 'molecule' section")


def test_plasmon_symmetries_invalid_axis(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["plasmon"]["simulation"].update({"symmetries": ["Q", 1, "Z", -1]}), ValueError, "Invalid plasmon symmetry")


def test_plasmon_symmetries_invalid_phase(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["plasmon"]["simulation"].update({"symmetries": ["Y", 2, "Z", -1]}), ValueError, "Invalid plasmon symmetry")


def test_plasmon_source_invalid_center(tmp_path):
    _test_validation_error(tmp_path,
        lambda c: c["plasmon"].update({"source": {"type": "gaussian", "center": [0, 0, "nan"], "size": [1, 1, 0], "component": "z", "additional_parameters": {"wavelength": 0.5}}}),
        ValueError, "Invalid plasmon source center")


def test_molecule_and_plasmon_source_conflict(tmp_path):
    _test_validation_error(tmp_path,
        lambda c: c["plasmon"].update({"source": {"type": "continuous", "center": [-0.04, 0, 0], "size": [0, 0.1, 0.1], "component": "z", "additional_parameters": {"wavelength": 0.5}}}),
        ValueError, "Source found in both plasmon and molecule sections")


def test_molecule_source_missing_intensity(tmp_path):
    _test_validation_error(tmp_path,
        lambda c: c["molecule"].update({"source": {"type": "kick", "peak_time": 0, "width_steps": 1, "component": "z"}}),
        ValueError, "Molecule source requires 'intensity'")


def test_molecule_source_negative_peak_time(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["molecule"]["source"].update({"peak_time": -1}), ValueError, "peak_time must be a positive value")


def test_molecule_source_invalid_type(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["molecule"]["source"].update({"type": "sine"}), ValueError, "Molecule source must be of type 'pulse' or 'kick'")


def test_molecule_source_pulse_missing_frequency(tmp_path):
    _test_validation_error(tmp_path,
        lambda c: c["molecule"].update({"source": {"type": "pulse", "intensity": 0.01, "peak_time": 0, "width_steps": 50, "component": "z", "additional_parameters": {}}}),
        ValueError, "Molecule source of type 'pulse' requires 'wavelength' or 'frequency'")


def test_checkpoint_missing_frequency(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["molecule"].update({"files": {"checkpoint": {"filepath": "cp.npz"}}}), ValueError, "Checkpointing requires 'frequency'")


def test_checkpoint_negative_frequency(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["molecule"].update({"files": {"checkpoint": {"filepath": "cp.npz", "frequency": -10}}}), ValueError, "Checkpointing 'frequency' must be a positive value")


def test_fourier_gamma_non_positive(tmp_path):
    _test_validation_error(tmp_path,
        lambda c: (c.pop("plasmon", None), c["molecule"].update({"modifiers": {"fourier": {"gamma": -0.01, "spectrum_filepath": "spec.png"}}})),
        ValueError, "Fourier modifier 'gamma' must be a positive value")


def test_fourier_damping_gamma_non_positive(tmp_path):
    _test_validation_error(tmp_path,
        lambda c: (c.pop("plasmon", None), c["molecule"].update({"modifiers": {"fourier": {"gamma": 0.01, "spectrum_filepath": "spec.png", "damping_gamma": -0.1}}})),
        ValueError, "Damping 'gamma' must be a positive value")


def test_broadening_missing_type(tmp_path):
    _test_validation_error(tmp_path,
        lambda c: (c.pop("plasmon", None), c["molecule"].update({"modifiers": {"broadening": {"gam0": 1.0}}})),
        ValueError, "Broadening modifier requires 'type' attribute")


def test_broadening_invalid_type(tmp_path):
    _test_validation_error(tmp_path,
        lambda c: (c.pop("plasmon", None), c["molecule"].update({"modifiers": {"broadening": {"type": "invalid"}}})),
        ValueError, "Broadening 'type' must be 'static' or 'dynamic'")


def test_broadening_gam0_non_positive(tmp_path):
    _test_validation_error(tmp_path,
        lambda c: (c.pop("plasmon", None), c["molecule"].update({"modifiers": {"broadening": {"type": "static", "gam0": -1.0}}})),
        ValueError, "Broadening 'gam0' must be a positive value")


def test_broadening_xi_negative(tmp_path):
    _test_validation_error(tmp_path,
        lambda c: (c.pop("plasmon", None), c["molecule"].update({"modifiers": {"broadening": {"type": "dynamic", "xi": -0.1}}})),
        ValueError, "Broadening 'xi' must be a non-negative value")


def test_broadening_eps0_negative(tmp_path):
    _test_validation_error(tmp_path,
        lambda c: (c.pop("plasmon", None), c["molecule"].update({"modifiers": {"broadening": {"type": "dynamic", "eps0": -0.01}}})),
        ValueError, "Broadening 'eps0' must be a non-negative value")


def test_broadening_clamp_non_positive(tmp_path):
    _test_validation_error(tmp_path,
        lambda c: (c.pop("plasmon", None), c["molecule"].update({"modifiers": {"broadening": {"type": "static", "clamp": -10}}})),
        ValueError, "Broadening 'clamp' must be a positive value")


def test_nanoparticle_missing_required_fields(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["plasmon"].update({"nanoparticle": {}}), ValueError, "Nanoparticle requires")


def test_nanoparticle_invalid_center(tmp_path):
    _test_validation_error(tmp_path,
        lambda c: c["plasmon"].update({"nanoparticle": {"material": "Au_JC_visible", "radius": 0.05, "center": [0, 0, "nan"]}}),
        ValueError, "Invalid nanoparticle center")


def test_images_missing_timesteps_between(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["plasmon"].update({"images": {"additional_parameters": ["-Zc dkbluered"]}}), ValueError, "Images requires 'timesteps_between' attribute")


def test_comparison_missing_bases_or_xcs(tmp_path):
    _test_validation_error(tmp_path,
        lambda c: (c.pop("plasmon", None), c["molecule"].update({"modifiers": {"comparison": {"bases": ["sto3g"]}}})),
        ValueError, "Comparison mode requires both 'bases' and 'xcs'")


def test_invalid_xc_functional(tmp_path):
    _test_validation_error(tmp_path,
        lambda c: (c.pop("plasmon", None), c["molecule"].update({"xc": "nonexistent_xc"})),
        ValueError, "Error checking xc functional")


def test_molecule_geometry_bad_format(tmp_path):
    _test_validation_error(tmp_path, lambda c: c["molecule"].update({"geometry": [{"atom": "H"}]}), ValueError, "Each geometry entry must be a dict with 'atom'")


def test_comparison_missing_bases(tmp_path):
    _test_validation_error(tmp_path,
        lambda c: (c.pop("plasmon", None), c["molecule"].update({"modifiers": {"comparison": {"xcs": ["pbe"]}}})),
        ValueError, "Comparison mode requires both 'bases' and 'xcs'")


def test_comparison_invalid_lrc_parameter(tmp_path):
    _test_validation_error(tmp_path,
        lambda c: (c.pop("plasmon", None), c["molecule"].update({
            "modifiers": {
                "comparison": {
                    "bases": ["sto3g"],
                    "xcs": ["pbe0"],
                    "lrc_parameters": {"pbe0": -0.5}
                }
            }
        })),
        ValueError, "Error checking xc functional")