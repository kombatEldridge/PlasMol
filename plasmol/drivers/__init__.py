# drivers/__init__.py
from plasmol.drivers.classical import run as run_classical
from plasmol.drivers.plasmol import run as run_plasmol
from plasmol.drivers.quantum import run as run_quantum
from plasmol.drivers.custom_drivers.comparison import run as run_comparison
from plasmol.drivers.custom_drivers.fourier import run as run_fourier
from plasmol.drivers.custom_drivers.chen2010_fig1 import run as chen2010_fig1
from plasmol.drivers.custom_drivers.abs_cross_sec import run as run_abs_cross_sec

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
    elif driver_str == 'chen2010_fig1':
        return chen2010_fig1
    elif driver_str == 'abs_cross_sec':
        return run_abs_cross_sec
    else:
        raise ValueError(f"Unknown driver: {driver_str}. Please add your custom driver to the drivers/__init__.py file.")

__all__ = [
    'get_driver',
    'run_classical', 
    'run_plasmol', 
    'run_quantum', 
    'run_comparison', 
    'run_fourier', 
    'chen2010_fig1',
    'run_abs_cross_sec',
    ]