"""Custom nanoparticle materials loaded from a user Python file."""
import json

import meep as mp
import pytest
from argparse import Namespace

from plasmol.utils.params import PARAMS
from plasmol.utils.params_helpers.common import load_meep_material


CUSTOM_NAME = "PlasMolNaDrude"
CUSTOM_FILE = """
import meep as mp
import meep.materials as materials

materials.PlasMolNaDrude = mp.Medium(
    epsilon=1.0,
    E_susceptibilities=[
        mp.DrudeSusceptibility(frequency=3.6340635899, gamma=0.0427473847, sigma=1.0),
        mp.LorentzianSusceptibility(frequency=2.5, gamma=0.2, sigma=1.5),
    ],
)
"""

OVERWRITE_FILE = """
import meep as mp
import meep.materials as materials

materials.Au = mp.Medium(epsilon=9.0)
materials.PlasMolNaDrude = mp.Medium(epsilon=2.5)
"""

NO_ASSIGNMENT_FILE = """
import meep as mp
NaDrude = mp.Medium(epsilon=2.0)
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_builtin_material_unchanged():
    import meep.materials as materials
    medium = load_meep_material("Au")
    assert medium is materials.Au
    assert isinstance(medium, mp.Medium)


def test_custom_file_assigns_a_new_medium(tmp_path):
    import meep.materials as materials
    _write(tmp_path, "na_drude.py", CUSTOM_FILE)
    assert not hasattr(materials, CUSTOM_NAME)
    medium = load_meep_material(CUSTOM_NAME, "na_drude.py", input_path=str(tmp_path / "job.json"))
    assert medium is getattr(materials, CUSTOM_NAME)
    assert len(medium.E_susceptibilities) == 2
    assert isinstance(medium.E_susceptibilities[0], mp.DrudeSusceptibility)
    assert isinstance(medium.E_susceptibilities[1], mp.LorentzianSusceptibility)
    delattr(materials, CUSTOM_NAME)


def test_custom_name_cannot_be_a_builtin(tmp_path):
    import meep.materials as materials
    original = materials.Au
    with pytest.raises(ValueError, match="already exists in meep.materials"):
        load_meep_material("Au", "na_drude.py", input_path=str(tmp_path / "job.json"))
    assert materials.Au is original


def test_file_cannot_replace_a_builtin(tmp_path):
    import meep.materials as materials
    original = materials.Au
    n_susc = len(original.E_susceptibilities)
    _write(tmp_path, "overwrite.py", OVERWRITE_FILE)
    with pytest.raises(ValueError, match="replaces built-in"):
        load_meep_material(CUSTOM_NAME, "overwrite.py", input_path=str(tmp_path / "job.json"))
    assert materials.Au is original
    assert len(materials.Au.E_susceptibilities) == n_susc
    assert not hasattr(materials, CUSTOM_NAME)


def test_file_must_assign_the_named_medium(tmp_path):
    import meep.materials as materials
    _write(tmp_path, "local_only.py", NO_ASSIGNMENT_FILE)
    with pytest.raises(ImportError, match="did not set meep.materials.PlasMolNaDrude"):
        load_meep_material(CUSTOM_NAME, "local_only.py", input_path=str(tmp_path / "job.json"))
    assert not hasattr(materials, CUSTOM_NAME)


def test_missing_material_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_meep_material(CUSTOM_NAME, "missing.py", input_path=str(tmp_path / "job.json"))


def test_params_loads_material_file_beside_the_json(tmp_path):
    import meep.materials as materials
    _write(tmp_path, "na_drude.py", CUSTOM_FILE)
    if hasattr(materials, CUSTOM_NAME):
        delattr(materials, CUSTOM_NAME)
    cfg = {
        "settings": {"dt": 0.5, "t_end": 5.0, "driver": "classical"},
        "plasmon": {
            "simulation": {
                "cell_length": 0.2,
                "pml_thickness": 0.02,
                "surrounding_material_index": 1.0,
            },
            "source": {
                "type": "gaussian",
                "center": [0, 0, 0],
                "size": [0.2, 0, 0],
                "component": "z",
                "additional_parameters": {"wavelength": 0.5},
            },
            "nanoparticle": {
                "material": CUSTOM_NAME,
                "material_file": "na_drude.py",
                "radius": 0.025,
                "center": [0, 0, 0],
            },
        },
    }
    json_path = tmp_path / "custom_np.json"
    json_path.write_text(json.dumps(cfg))
    params = PARAMS(Namespace(input=str(json_path), verbose=0, log=None, checkpoint=None))
    assert params.nanoparticle_material is getattr(materials, CUSTOM_NAME)
    assert params.nanoparticle.material is params.nanoparticle_material
    delattr(materials, CUSTOM_NAME)
