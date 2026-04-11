# tests/conftest.py
import pytest
from argparse import Namespace
from plasmol.utils.input.params import PARAMS

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
    },
    "files": {
      "field_e_filepath": "field_e.csv",
      "field_p_filepath": "field_p.csv",
      "spectra_e_vs_p_filepath": "spectrum.png"
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