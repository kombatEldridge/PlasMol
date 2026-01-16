# tests/test_parser.py
import pytest
import numpy as np
import json
from plasmol.utils.input.parser import postprocess_quantum, parseInputFile
from argparse import Namespace  # For mocking args

# Sample JSON content for testing (based on converted examples)
SAMPLE_JSON_CLASSICAL = """
{
  "settings": {
    "dt": 0.1,
    "t_end": 1,
    "eField_path": "eField.csv"
  },
  "classical": {
    "source": {
      "sourceType": "continuous",
      "sourceCenter": [-0.04, 0, 0],
      "sourceSize": [0, 0.1, 0.1],
      "frequency": 20,
      "isIntegrated": true
    },
    "simulation": {
      "eFieldCutOff": 1e-12,
      "cellLength": 0.1,
      "pmlThickness": 0.01,
      "symmetries": ["Y", 1, "Z", -1],
      "surroundingMaterialIndex": 1.33
    },
    "object": {
      "material": "Au",
      "radius": 0.03,
      "center": [0, 0, 0]
    },
    "hdf5": {
      "timestepsBetween": 1,
      "intensityMin": 3,
      "intensityMax": 4,
      "imageDirName": "hello"
    }
  }
}
"""

SAMPLE_JSON_QUANTUM = """
{
  "settings": {
    "dt": 0.1,
    "t_end": 40,
    "eField_path": "eField.csv"
  },
  "quantum": {
    "rttddft": {
      "geometry": [
        {"atom": "O", "coords": [0.0, 0.0, -0.1302052882]},
        {"atom": "H", "coords": [1.4891244004, 0.0, 1.0332262019]},
        {"atom": "H", "coords": [-1.4891244004, 0.0, 1.0332262019]}
      ],
      "units": "bohr",
      "check_tolerance": 1e-12,
      "charge": 0,
      "spin": 0,
      "basis": "6-31g",
      "xc": "pbe0",
      "resplimit": 1e-20,
      "propagator": "magnus2",
      "pc_convergence": 1e-12,
      "maxiter": 200
    },
    "files": {
      "checkpoint": {
        "frequency": 100,
        "path": "checkpoint.npz"
      },
      "pField_path": "pField.csv",
      "pField_Transform_path": "pField-transformed.csv",
      "eField_vs_pField_path": "output.png",
      "eV_spectrum_path": "spectrum.png"
    }
  }
}
"""

SAMPLE_JSON_PLASMOL = """
{
  "settings": {
    "dt": 0.1,
    "t_end": 40,
    "eField_path": "eField.csv"
  },
  "classical": {
    "source": {
      "sourceType": "continuous",
      "sourceCenter": [-0.04, 0, 0],
      "sourceSize": [0, 0.1, 0.1],
      "frequency": 5,
      "isIntegrated": true
    },
    "simulation": {
      "eFieldCutOff": 1e-12,
      "cellLength": 0.1,
      "pmlThickness": 0.01,
      "symmetries": ["Y", 1, "Z", -1],
      "surroundingMaterialIndex": 1.33
    },
    "object": {
      "material": "Au",
      "radius": 0.03,
      "center": [0, 0, 0]
    },
    "hdf5": {
      "timestepsBetween": 1,
      "intensityMin": 3,
      "intensityMax": 10,
      "imageDirName": "hello"
    },
    "molecule": {
      "center": [0, 0, 0]
    }
  },
  "quantum": {
    "rttddft": {
      "geometry": [
        {"atom": "O", "coords": [0.0, 0.0, -0.1302052882]},
        {"atom": "H", "coords": [1.4891244004, 0.0, 1.0332262019]},
        {"atom": "H", "coords": [-1.4891244004, 0.0, 1.0332262019]}
      ],
      "units": "bohr",
      "check_tolerance": 1e-12,
      "charge": 0,
      "spin": 0,
      "basis": "6-31g",
      "xc": "pbe0",
      "resplimit": 1e-20,
      "propagator": "magnus2",
      "pc_convergence": 1e-12,
      "maxiter": 200
    },
    "files": {
      "checkpoint": {
        "frequency": 100,
        "path": "checkpoint.npz"
      },
      "pField_path": "pField.csv",
      "pField_Transform_path": "pField-transformed.csv",
      "eField_vs_pField_path": "output.png",
      "eV_spectrum_path": "spectrum.png"
    }
  }
}
"""

@pytest.fixture
def temp_json_file(tmp_path):
    def _create_file(content):
        file_path = tmp_path / "input.json"
        file_path.write_text(content)
        return file_path
    return _create_file

def test_inputFilePrepare_classical(temp_json_file, caplog):
    file_path = temp_json_file(SAMPLE_JSON_CLASSICAL)
    args = Namespace(input=str(file_path))
    preparams = parseInputFile(args)
    
    assert preparams['simulation_type'] == 'Classical'
    assert 'classical' in preparams
    assert 'quantum' not in preparams
    assert preparams['settings']['dt'] == 0.1
    assert 'Only classical parameters given' in caplog.text

def test_inputFilePrepare_quantum(temp_json_file, caplog):
    file_path = temp_json_file(SAMPLE_JSON_QUANTUM)
    args = Namespace(input=str(file_path))
    preparams = parseInputFile(args)
    
    assert preparams['simulation_type'] == 'Quantum'
    assert 'quantum' in preparams
    assert 'classical' not in preparams
    assert preparams['quantum']['rttddft']['geometry']['atoms'] == ['O', 'H', 'H']
    assert 'Only quantum parameters given' in caplog.text

def test_inputFilePrepare_plasmol(temp_json_file):
    file_path = temp_json_file(SAMPLE_JSON_PLASMOL)
    args = Namespace(input=str(file_path))
    preparams = parseInputFile(args)
    
    assert preparams['simulation_type'] == 'PlasMol'
    assert 'classical' in preparams
    assert 'quantum' in preparams
    assert preparams['classical']['molecule'] == {'center': [0, 0, 0]}

def test_inputFilePrepare_missing_required(temp_json_file):
    invalid_json = '{"settings": {"dt": 0.1}}'  # Missing t_end
    file_path = temp_json_file(invalid_json)
    args = Namespace(input=str(file_path))
    with pytest.raises(RuntimeError, match="No 't_end' value"):
        parseInputFile(args)

def test_postprocess_quantum():
    quantum_params = {
        'rttddft': {
            'geometry': [
                {'atom': 'O', 'coords': [0.0, 0.0, -0.1302052882]},
                {'atom': 'H', 'coords': [1.4891244004, 0.0, 1.0332262019]},
                {'atom': 'H', 'coords': [-1.4891244004, 0.0, 1.0332262019]}
            ],
            'units': 'bohr',
            'other': 'params'
        }
    }
    processed = postprocess_quantum(quantum_params)
    
    assert processed['rttddft']['geometry']['atoms'] == ['O', 'H', 'H']
    coords = processed['rttddft']['geometry']['coords']
    assert np.allclose(coords['O1'], [0.0, 0.0, -0.1302052882])  # Since units=bohr, no conversion
    assert 'molecule_coords' in processed['rttddft']['geometry']
    assert processed['rttddft']['other'] == 'params'  # Other keys preserved

def test_postprocess_quantum_angstrom():
    quantum_params = {
        'rttddft': {
            'geometry': [
                {'atom': 'O', 'coords': [0.0, 0.0, 0.0]},
                {'atom': 'H', 'coords': [1.0, 0.0, 0.0]}
            ],
            'units': 'angstrom'
        }
    }
    processed = postprocess_quantum(quantum_params)
    factor = 1.8897259886
    assert np.allclose(processed['rttddft']['geometry']['coords']['H2'], [1.0 * factor, 0.0, 0.0])

def test_postprocess_quantum_invalid():
    invalid_quantum = {'rttddft': {'geometry': 'invalid'}}
    with pytest.raises(ValueError, match="list of atom objects"):
        postprocess_quantum(invalid_quantum)
    
    no_units = {'rttddft': {'geometry': [{'atom': 'O', 'coords': [0,0,0]}]}}
    with pytest.raises(ValueError, match="specify 'units'"):
        postprocess_quantum(no_units)