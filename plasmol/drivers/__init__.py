# drivers/__init__.py
from plasmol.drivers.classical import run as run_classical
from plasmol.drivers.plasmol import run as run_plasmol
from plasmol.drivers.quantum import run as run_quantum
from plasmol.drivers.custom_drivers.comparison import run as run_comparison
from plasmol.drivers.custom_drivers.fourier import run as run_fourier
from plasmol.drivers.custom_drivers.scatter_response_fxn import run as scatter_response_fxn
from plasmol.drivers.custom_drivers.np_abs_cross_sec import run as run_np_abs_cross_sec
from plasmol.drivers.custom_drivers.plasmol_abs_cross_sec import run as run_plasmol_abs_cross_sec
from plasmol.drivers.custom_drivers.verify_source import run as run_verify_source
from plasmol.drivers.tune import run as run_tune

def get_driver(driver_str):
    if driver_str == 'classical':
        return run_classical
    elif driver_str == 'plasmol':
        return run_plasmol
    elif driver_str == 'quantum':
        return run_quantum
    elif driver_str == 'comparison':
        return run_comparison
    elif driver_str == 'fourier':
        return run_fourier
    elif driver_str == 'scatter_response_fxn':
        return scatter_response_fxn
    elif driver_str == 'np_abs_cross_sec':
        return run_np_abs_cross_sec
    elif driver_str == 'plasmol_abs_cross_sec':
        return run_plasmol_abs_cross_sec
    elif driver_str == 'verify_source':
        return run_verify_source
    elif driver_str == 'tune':
        return run_tune
    else:
        raise ValueError(f"Unknown driver: {driver_str}. Please add your custom driver to the drivers/__init__.py file.")

__all__ = [
    'get_driver',
    'run_classical', 
    'run_plasmol', 
    'run_quantum', 
    'run_comparison', 
    'run_fourier', 
    'scatter_response_fxn',
    'run_np_abs_cross_sec',
    'run_verify_source',
    'run_tune',
    ]