# drivers/fourier.py
import os
import copy
import logging
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt

from plasmol import constants
from plasmol.quantum.electric_field import ELECTRICFIELD
from plasmol.drivers import *

from plasmol.utils.plotting import plot_fields
from plasmol.utils.csv import init_csv, update_csv, read_field_csv
from plasmol.utils.logging import setup_logging

logger = logging.getLogger("main")

class DirectionFilter(logging.Filter):
    def __init__(self, direction):
        super().__init__()
        self.prefix = f"[{direction}-dir]"

    def filter(self, record):
        record.msg = f"{self.prefix} {record.msg}"
        return True
    
def _run_quantum_with_prefix(params_copy):
    # Setup logging (honors --log if provided so *everything* goes to the log
    # file only with no terminal output; otherwise outputs to terminal).
    setup_logging(
        getattr(params_copy, 'verbose', 1),
        getattr(params_copy, 'log', None)
    )
    f = DirectionFilter(params_copy.molecule_source_dict['component'])
    logging.getLogger("main").addFilter(f)
    logging.getLogger().addFilter(f)
    try:
        run_quantum(params_copy)
    finally:
        logging.getLogger("main").removeFilter(f)
        logging.getLogger().removeFilter(f)

def apply_damping(mu_arrs, gamma):
    """
    Apply damping to the polarizability array.

    Applies a damping factor to the polarizability values based on the provided parameters.
    The damping is applied as per the formula: mu_damped = mu * exp(-t/tau).
    Parameters:
    mu_arr : list of float
        The polarizability values to be damped.
    tau : float
        The damping time constant.

    Returns:
    list of float
        The damped polarizability values.
    """
    t = np.array(mu_arrs[0])
    damped_mu_x = mu_arrs[1] * np.exp(-t / gamma)
    damped_mu_y = mu_arrs[2] * np.exp(-t / gamma)
    damped_mu_z = mu_arrs[3] * np.exp(-t / gamma)
    return damped_mu_x, damped_mu_y, damped_mu_z

def fold(file_x, file_y, file_z):
    def read_dipole_component(filename, column):
        df = pd.read_csv(filename, delimiter=',', header=0, comment='#')
        time = np.array(df['Timestamps (au)'].values, dtype=float)
        dipole = np.array(df[column].values, dtype=float)
        logger.debug(f"Loaded {len(time)} points from {filename}")
        return time, dipole

    # Load components
    tx, dx = read_dipole_component(file_x, 'X Values')
    ty, dy = read_dipole_component(file_y, 'Y Values')
    tz, dz = read_dipole_component(file_z, 'Z Values')

    dtx = tx[1] - tx[0]
    dty = ty[1] - ty[0]
    dtz = tz[1] - tz[0]
    if not (np.isclose(dtx, dty) and np.isclose(dtx, dtz)):
        raise ValueError("Inconsistent timesteps across files!")

    # Find the length of the shortest file
    min_length = min(len(tx), len(ty), len(tz))

    # Trim all arrays to the shortest length
    time_points = tx[:min_length]
    dx = dx[:min_length]
    dy = dy[:min_length]
    dz = dz[:min_length]

    dipole_moment = np.vstack([dx, dy, dz])  # Shape: (3, N)

    return time_points, dipole_moment

def fourier(time, dipole, file, damp):
    dt = time[1] - time[0]
    abs_real = [[], [], []]
    abs_imag = [[], [], []]
    freqs_out = []

    # Calculate frequencies once, as they are the same for all axes
    freqs_au = np.fft.fftfreq(len(time), d=dt) * 2 * np.pi
    freqs_ev = freqs_au * 27.211386
    mask = (freqs_ev >= 0) & (freqs_ev <= 50)

    for axis in (0, 1, 2):
        logger.debug(f"Starting Fourier transform of direction { {0:'x', 1:'y', 2:'z'}[axis] }")
        dipole_windowed = dipole[axis] * np.exp(-damp * time)
        S = np.fft.fft(dipole_windowed) * dt 
        abs_real[axis] = S.real[mask]
        abs_imag[axis] = S.imag[mask]

    # Keep only frequencies in the 0 to 50 eV range
    freqs_out = freqs_ev[mask]

    logger.debug("Fourier transform done!")
    for i in range(3):
        abs_real[i] = np.array(abs_real[i])
        abs_imag[i] = np.array(abs_imag[i])

    np.savez(file, abs_imag=abs_imag, freqs=freqs_out)
    logger.debug(f"Fourier transform saved to {file}!")

    return abs_imag, freqs_out

def absorption(imag, freqs):
    fullsum = imag[0] + imag[1] + imag[2]
    return - 4 * np.pi * freqs / 3 / constants.C_AU * fullsum

def run(params):
    # Log running fourier with gamma
    if params.resumed_from_checkpoint:
        logger.info(f"Resuming Fourier transform process with gamma: {params.fourier_gamma}")
    else:
        logger.info(f"Running Fourier transform process with gamma: {params.fourier_gamma}")

    processes = []
    params_copies = []

    # Set up params copies and paths for each direction (always, to support transform)
    for dir in params.xyz:
        # Create a deep copy of params to ensure process safety
        params_copy = copy.deepcopy(params)
        params_copy.molecule_source_dict['component'] = dir
        params_copy.molecule_source_field = ELECTRICFIELD(params_copy).field
        params_copy.dir_path = f"{dir}_dir"
        params_copy.field_e_filepath = getattr(params_copy, f'field_e_{dir}_filepath')
        params_copy.field_p_filepath = getattr(params_copy, f'field_p_{dir}_filepath')
        params_copy.spectra_e_vs_p_filepath = getattr(params_copy, f'spectra_e_{dir}_vs_p_{dir}_filepath')
        if not params.resumed_from_checkpoint:
            os.makedirs(params_copy.dir_path, exist_ok=True)
        params_copies.append(params_copy)

    # Create and start a process for each direction
    for params_copy in params_copies:
        process = multiprocessing.Process(target=_run_quantum_with_prefix, args=(params_copy,))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    # Add damping to the polarizability fields if mu_damping is set
    for params_copy in params_copies:
        if params.fourier_damp:
            mu_arrs = read_field_csv(params_copy.field_p_filepath)
            mu_x, mu_y, mu_z = apply_damping(mu_arrs, params.fourier_damping_gamma)
            params_copy.field_p_filepath = params_copy.field_p_filepath.replace('.csv', f'_damped.csv')
            init_csv(params_copy.field_p_filepath, f"# Molecule\'s Polarizability Field intensity in atomic units but damped with mu_damped = mu * exp(-t/tau) where tau = {params.fourier_damping_gamma}")
            for t, x, y, z in zip(mu_arrs[0], mu_x, mu_y, mu_z):
                update_csv(params_copy.field_p_filepath, t, x, y, z)
            logging.info(f"Damped polarizability field written to {params_copy.field_p_filepath}")

        # plot_fields(params_copy.field_e_filepath, params_copy.field_p_filepath, params_copy.spectra_e_vs_p_filepath)

    # Apply Fourier transform to the field_p CSV files
    time_points, dipole_moment = fold(params_copies[0].field_p_filepath, params_copies[1].field_p_filepath, params_copies[2].field_p_filepath)
    abs_imag, freqs = fourier(time_points, dipole_moment, params.fourier_npz_filepath, params.fourier_gamma)
    abs = absorption(abs_imag, freqs)

    plt.figure(figsize=(14, 8))
    plt.plot(freqs, abs/max(abs), color='green', label='Spectrum')
    plt.xlabel('Angular frequency ω (eV)', fontsize=16)
    plt.ylabel('Absorption', fontsize=16)
    plt.title('Absorption Spectrum', fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(params.fourier_spectrum_filepath, dpi=600)
