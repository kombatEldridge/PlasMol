# tests/conftest.py
import pytest
from argparse import Namespace
from plasmol.utils.params import PARAMS

MINIMAL_MOLECULE_JSON = """
{
  "settings": {
    "dt": 0.5,
    "t_end": 5.0
  },
  "molecule": {
    "geometry": [
      {"atom": "H", "coord": [0.0, 0.0, 0.0]},
      {"atom": "H", "coord": [0.0, 0.0, 1.4]}
    ],
    "geometry_units": "bohr",
    "charge": 0,
    "spin": 0,
    "basis": "sto3g",
    "xc": "pbe",
    "propagator": {
      "type": "rk4"
    },
    "source": {
      "type": "kick",
      "intensity": 0.001,
      "peak_time": 0.0,
      "width_steps": 1,
      "component": "z"
    }
  }
}
"""

PLASMON_ONLY_JSON = """
{
  "settings": {
    "dt": 0.5,
    "t_end": 10.0
  },
  "plasmon": {
    "simulation": {
      "tolerance_field_e": 1e-12,
      "cell_length": 0.2,
      "pml_thickness": 0.02,
      "surrounding_material_index": 1.0,
      "symmetries": ["Y", 1, "Z", -1]
    },
    "source": {
      "type": "gaussian",
      "center": [0, 0, 0],
      "size": [0.2, 0, 0],
      "component": "z",
      "amplitude": 1.0,
      "is_integrated": true,
      "additional_parameters": {
        "wavelength": 0.5
      }
    }
  }
}
"""

# Hydrogen atom — simplest possible open-shell (1 electron, doublet)
H_ATOM_JSON = """
{
  "settings": { "dt": 0.2, "t_end": 1.0 },
  "molecule": {
    "geometry": [{"atom": "H", "coord": [0.0, 0.0, 0.0]}],
    "geometry_units": "bohr",
    "charge": 0,
    "spin": 1,
    "basis": "sto3g",
    "xc": "pbe",
    "propagator": {"type": "rk4"},
    "source": {
      "type": "kick", "intensity": 0.001, "peak_time": 0.0,
      "width_steps": 1, "component": "z"
    }
  }
}
"""

LI_ATOM_JSON = H_ATOM_JSON.replace('"atom": "H"', '"atom": "Li"')

H2_CATION_JSON = """
{
  "settings": { "dt": 0.2, "t_end": 1.0 },
  "molecule": {
    "geometry": [
      {"atom": "H", "coord": [0.0, 0.0, 0.0]},
      {"atom": "H", "coord": [0.0, 0.0, 1.4]}
    ],
    "geometry_units": "bohr",
    "charge": 1, "spin": 1,
    "basis": "sto3g", "xc": "pbe",
    "propagator": {"type": "rk4"},
    "source": {
      "type": "kick", "intensity": 0.001, "peak_time": 0.0,
      "width_steps": 1, "component": "z"
    }
  }
}
"""

@pytest.fixture
def minimal_json_path(tmp_path):
    """Creates a temporary minimal JSON file and returns its path."""
    json_path = tmp_path / "minimal_molecule.json"
    json_path.write_text(MINIMAL_MOLECULE_JSON.strip())
    return str(json_path)

@pytest.fixture
def minimal_args(minimal_json_path):
    """CLI arguments pointing to our minimal JSON."""
    return Namespace(
        input=minimal_json_path,
        verbose=0,
        log=None,
        checkpoint=None
    )

@pytest.fixture
def minimal_params(minimal_args):
    """The actual PARAMS object from the minimal JSON (fastest possible)."""
    return PARAMS(minimal_args)

@pytest.fixture
def plasmon_only_json_path(tmp_path):
    """Creates a temporary plasmon-only JSON file."""
    json_path = tmp_path / "plasmon_only.json"
    json_path.write_text(PLASMON_ONLY_JSON.strip())
    return str(json_path)

@pytest.fixture
def plasmon_only_args(plasmon_only_json_path):
    return Namespace(input=plasmon_only_json_path, verbose=0, log=None, checkpoint=None)


def _json_to_params(tmp_path, json_str, name):
    p = tmp_path / name
    p.write_text(json_str.strip())
    args = Namespace(input=str(p), verbose=0, log=None, checkpoint=None)
    return PARAMS(args)

@pytest.fixture
def h_atom_params(tmp_path):
    return _json_to_params(tmp_path, H_ATOM_JSON, "h_atom.json")

@pytest.fixture
def li_atom_params(tmp_path):
    return _json_to_params(tmp_path, LI_ATOM_JSON, "li_atom.json")

@pytest.fixture
def h2_cation_params(tmp_path):
    return _json_to_params(tmp_path, H2_CATION_JSON, "h2_cation.json")

@pytest.fixture
def h_atom_molecule(h_atom_params):
    """A fully built open-shell MOLECULE. SCF runs once and is reused."""
    from plasmol.quantum.molecule import MOLECULE
    return MOLECULE(h_atom_params)

@pytest.fixture
def li_atom_molecule(li_atom_params):
    from plasmol.quantum.molecule import MOLECULE
    return MOLECULE(li_atom_params)

@pytest.fixture
def h2_cation_molecule(h2_cation_params):
    from plasmol.quantum.molecule import MOLECULE
    return MOLECULE(h2_cation_params)

@pytest.fixture
def h2_closed_shell_molecule(minimal_params):
    """The existing closed-shell H2 as a control."""
    from plasmol.quantum.molecule import MOLECULE
    return MOLECULE(minimal_params)
