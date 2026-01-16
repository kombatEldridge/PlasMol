# tests/test_params.py
import pytest
import meep as mp
import numpy as np
from unittest.mock import patch  # For mocking logger and datetime
from plasmol.utils.input.params import PARAMS
from plasmol.classical.sources import MEEPSOURCE

# Sample preparams for different simulation types
SAMPLE_PREPARAMS_CLASSICAL = {
    'simulation_type': 'Classical',
    'args': {'restart': False, 'do_nothing': False},
    'settings': {'dt': 0.1, 't_end': 1, 'eField_path': 'eField.csv'},
    'classical': {
        'simulation': {'eFieldCutOff': 1e-12, 'cellLength': 0.1, 'pmlThickness': 0.01, 'surroundingMaterialIndex': 1.33},
        'source': {'sourceType': 'continuous', 'sourceCenter': [-0.04, 0, 0], 'sourceSize': [0, 0.1, 0.1], 'frequency': 20, 'is_integrated': True},
        'object': {'material': 'Au', 'radius': 0.03, 'center': [0, 0, 0]},
        'hdf5': {'timestepsBetween': 1, 'intensityMin': 3, 'intensityMax': 4},
        'simulation': {'symmetries': ['Y', 1, 'Z', -1]}  # Note: duplicated key, but assuming correct in real
    }
}

SAMPLE_PREPARAMS_QUANTUM = {
    'simulation_type': 'Quantum',
    'args': {'restart': False, 'do_nothing': False},
    'settings': {'dt': 0.1, 't_end': 40, 'eField_path': 'eField.csv'},
    'quantum': {
        'files': {
            'pField_path': 'pField.csv', 'pField_Transform_path': 'pField-transformed.csv',
            'eField_vs_pField_path': 'output.png', 'eV_spectrum_path': 'spectrum.png',
            'checkpoint': {'frequency': 100, 'path': 'checkpoint.npz'}
        },
        'rttddft': {
            'geometry': {'atoms': ['O', 'H', 'H'], 'coords': {'O1': [0,0,0], 'H2': [1,0,0], 'H3': [-1,0,0]}, 'molecule_coords': 'O 0 0 0;H 1 0 0;H -1 0 0'},
            'basis': '6-31g', 'charge': 0, 'spin': 0, 'xc': 'pbe0', 'propagator': 'magnus2',
            'check_tolerance': 1e-12, 'pc_convergence': 1e-12, 'maxiter': 200
        }
    }
}

SAMPLE_PREPARAMS_PLASMOL = {
    'simulation_type': 'PlasMol',
    'args': {'restart': False, 'do_nothing': False},
    'settings': {'dt': 0.1, 't_end': 40, 'eField_path': 'eField.csv'},
    'classical': SAMPLE_PREPARAMS_CLASSICAL['classical'],
    'quantum': SAMPLE_PREPARAMS_QUANTUM['quantum']
}
SAMPLE_PREPARAMS_PLASMOL['classical']['molecule'] = {'center': [0, 0, 0]}  # Required for PlasMol

def test_PARAMS_classical(caplog):
    params = PARAMS(SAMPLE_PREPARAMS_CLASSICAL)
    
    assert params.type == 'Classical'
    assert params.dt == 0.1
    assert params.t_end == 1
    assert params.eField_path == 'eField.csv'
    assert isinstance(params.source, MEEPSOURCE)
    assert params.source.component == mp.Ez  # Default 'z'
    assert isinstance(params.nanoparticle, mp.Sphere)
    assert len(params.symmetry) == 2  # Y and Z mirrors
    assert params.hdf5['timestepsBetween'] == 1
    assert 'resolution' in params.simulation  # Auto-added
    assert 'No picture output' not in caplog.text  # Since hdf5 present

def test_PARAMS_quantum():
    params = PARAMS(SAMPLE_PREPARAMS_QUANTUM)
    
    assert params.type == 'Quantum'
    assert params.pField_path == 'pField.csv'
    assert params.propagator == 'magnus2'
    assert params.maxiter == 200
    assert params.atoms == ['O', 'H', 'H']
    assert params.charge == 0

def test_PARAMS_plasmol():
    params = PARAMS(SAMPLE_PREPARAMS_PLASMOL)
    
    assert params.type == 'PlasMol'
    # Has both classical and quantum attrs
    assert hasattr(params, 'source')
    assert hasattr(params, 'nanoparticle')
    assert hasattr(params, 'propagator')
    assert params.molecule_position == {'center': [0, 0, 0]}

def test_PARAMS_missing_required():
    invalid_preparams = {'simulation_type': 'Classical', 'args': {}, 'settings': {'dt': 0.1}, 'classical': {}}
    with pytest.raises(RuntimeError, match="No 't_end' value"):
        PARAMS(invalid_preparams)

def test_PARAMS_comparison_mode():
    preparams = SAMPLE_PREPARAMS_QUANTUM.copy()
    preparams['quantum']['comparison'] = {'bases': ['6-31g'], 'xcs': ['pbe0'], 'num_virtual': 5, 'y_min': 0, 'y_max': 10}
    params = PARAMS(preparams)
    
    assert params.bases == ['6-31g']
    assert params.num_occupied == 5

def test_PARAMS_quantum_source_warning(caplog):
    preparams = SAMPLE_PREPARAMS_PLASMOL.copy()
    preparams['quantum']['source'] = {'shape': 'pulse'}  # Add quantum source
    params = PARAMS(preparams)
    
    assert not hasattr(params, 'shape')
    assert 'Ignoring quantum source' in caplog.text

@patch('plasmol.utils.input.params.datetime')
def test_PARAMS_hdf5_auto_dir(mock_datetime):
    mock_datetime.now.return_value.strftime.return_value = '01012000_000000'
    preparams = SAMPLE_PREPARAMS_CLASSICAL.copy()
    preparams['classical']['hdf5'] = {'timestepsBetween': 1, 'intensityMin': 3, 'intensityMax': 4}  # No imageDirName
    params = PARAMS(preparams)
    
    assert params.hdf5['imageDirName'] == 'classical-01012000_000000'

def test_PARAMS_invalid_propagator():
    preparams = SAMPLE_PREPARAMS_QUANTUM.copy()
    preparams['quantum']['rttddft']['propagator'] = 'invalid'
    with pytest.raises(ValueError, match="Unsupported propagator"):
        PARAMS(preparams)

def test_PARAMS_resolution_adjustment():
    preparams = SAMPLE_PREPARAMS_CLASSICAL.copy()
    preparams['classical']['simulation']['resolution'] = 10  # Mismatch with dt=0.1
    with patch('plasmol.utils.input.params.constants.convertTimeMeep2Atomic', 1):  # Simplify calc
        params = PARAMS(preparams)
        assert params.simulation['resolution'] != 10  # Adjusted