# drivers/custom_drivers/fourier.py
import os
import copy
import logging
import meep as mp
import numpy as np
import pandas as pd
import multiprocessing
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

from plasmol.utils import constants
from plasmol.quantum.sources import QUANTUMSOURCE
from plasmol.classical.sources import MEEPSOURCE
from plasmol.drivers import *
from plasmol.drivers.plasmol import run as run_plasmol
from plasmol.classical.meep_verbosity import meep_io_context

from plasmol.utils.csv import init_csv, update_csv, read_field_csv
from plasmol.utils.logging import setup_logging
from plasmol.utils.checkpoint import merge_per_direction_checkpoints, merge_final_checkpoints

logger = logging.getLogger("main")

class PrefixFilter(logging.Filter):
    def __init__(self, direction):
        super().__init__()
        self.prefix = f"[{direction}-dir]"

    def filter(self, record):
        record.msg = f"{self.prefix} {record.msg}"
        return True
    
def _run_quantum_with_prefix(params_copy):
    setup_logging(
        getattr(params_copy, 'verbose', 1),
        getattr(params_copy, 'log', None)
    )
    f = PrefixFilter(params_copy.molecule_source_component)
    logging.getLogger("main").addFilter(f)
    logging.getLogger().addFilter(f)
    try:
        run_quantum(params_copy)
    finally:
        logging.getLogger("main").removeFilter(f)
        logging.getLogger().removeFilter(f)

def _build_plasmol_meep_objects(params):
    """Create Meep source/nanoparticle objects on params (not picklable across processes)."""
    with meep_io_context(getattr(params, 'verbose', 1)):
        params.plasmon_source_object = MEEPSOURCE(
            source_type=getattr(params, 'plasmon_source_type').lower().strip(),
            source_center=getattr(params, 'plasmon_source_center'),
            source_size=getattr(params, 'plasmon_source_size'),
            component=getattr(params, 'plasmon_source_component'),
            is_integrated=getattr(params, 'plasmon_source_is_integrated'),
            **{k: v for k, v in getattr(params, 'plasmon_source_additional_parameters', {}).items()}
        )
        if getattr(params, 'has_nanoparticle', False):
            mat_name = params.nanoparticle_dict["material"]
            params.nanoparticle_material = params._load_meep_material(mat_name)
            params.nanoparticle = mp.Sphere(
                radius=getattr(params, 'nanoparticle_radius'),
                center=mp.Vector3(*getattr(params, 'nanoparticle_center')),
                material=params.nanoparticle_material
            )

def _run_plasmol_with_prefix(params_copy):
    setup_logging(
        getattr(params_copy, 'verbose', 1),
        getattr(params_copy, 'log', None)
    )
    _build_plasmol_meep_objects(params_copy)
    f = PrefixFilter(params_copy.plasmon_source_component)
    logging.getLogger("main").addFilter(f)
    logging.getLogger().addFilter(f)
    try:
        run_plasmol(params_copy)
    finally:
        logging.getLogger("main").removeFilter(f)
        logging.getLogger().removeFilter(f)

def apply_damping(mu_arrs, tau):
    """
    Apply damping to the polarizability array.

    Applies a damping factor to the polarizability values based on the provided parameters.
    The damping is applied as per the formula: mu_damped = mu * exp(-t/tau).
    Parameters:
    mu_arrs : list of lists
        The time + polarizability arrays from CSV.
    tau : float
        The damping time constant.

    Returns:
    tuple of arrays
        The damped polarizability values (x, y, z).
    """
    t = np.array(mu_arrs[0])
    damped_mu_x = mu_arrs[1] * np.exp(-t / tau)
    damped_mu_y = mu_arrs[2] * np.exp(-t / tau)
    damped_mu_z = mu_arrs[3] * np.exp(-t / tau)
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

def load_dipole_from_csv(file_path):
    """Load x/y/z dipole components from a single field CSV."""
    time_values, dx, dy, dz = read_field_csv(file_path)
    if len(time_values) < 2:
        raise ValueError(
            f"No induced dipole data found in {file_path}. "
            "Check that the plasmon field exceeds plasmon_tolerance_field_e and that t_end is long enough."
        )
    time_points = np.array(time_values, dtype=float)
    dipole_moment = np.vstack([dx, dy, dz])
    return time_points, dipole_moment

def _apply_tau_damping(field_p_filepath, tau, time_rounding_decimals):
    if np.isclose(tau, 0):
        return field_p_filepath

    mu_arrs = read_field_csv(field_p_filepath)
    mu_x, mu_y, mu_z = apply_damping(mu_arrs, tau)
    damped_filepath = f"{Path(field_p_filepath).with_suffix('')}_damped.csv"
    init_csv(
        damped_filepath,
        f"# Molecule's Polarizability Field intensity in atomic units but damped with "
        f"mu_damped = mu * exp(-t/tau) where tau = {tau}",
    )
    for t, x, y, z in zip(mu_arrs[0], mu_x, mu_y, mu_z):
        update_csv(damped_filepath, round(t, time_rounding_decimals), x, y, z)
    logger.info(f"Damped polarizability field written to {damped_filepath}")
    return damped_filepath

def fourier(time, dipole, damp, min_ev, max_ev, npz=None):
    dt = time[1] - time[0]
    abs_real = [[], [], []]
    abs_imag = [[], [], []]
    freqs_out = []

    # Calculate frequencies once, as they are the same for all axes
    freqs_au = np.fft.fftfreq(len(time), d=dt) * 2 * np.pi
    freqs_ev = freqs_au * 27.211386
    mask = (freqs_ev >= min_ev) & (freqs_ev <= max_ev)

    logger.debug(f"Performing Fourier transform with damping gamma={damp} and frequency range {min_ev}-{max_ev} eV...")

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

    if npz:
        np.savez(npz, abs_imag=abs_imag, freqs=freqs_out)
        logger.debug(f"Fourier transform saved to {npz}!")

    return abs_imag, freqs_out

def absorption(imag, freqs):
    fullsum = imag[0] + imag[1] + imag[2]
    return - 4 * np.pi * freqs / 3 / constants.C_AU * fullsum

def _set_up_params_copy_plasmol(params):
    params_copies = []
    for dir in params.xyz:
        params_copy = copy.deepcopy(params)
        params_copy.plasmon_source_component = dir
        params_copy.dir_path = f"{dir}_dir"
        params_copy.field_e_filepath = getattr(params_copy, f'field_e_{dir}_filepath')
        params_copy.field_p_filepath = getattr(params_copy, f'field_p_{dir}_filepath')
        params_copy.spectra_e_vs_p_filepath = getattr(params_copy, f'spectra_e_{dir}_vs_p_{dir}_filepath')
        os.makedirs(params_copy.dir_path, exist_ok=True)
        params_copies.append(params_copy)
    return params_copies

def _set_up_params_copy_molecule(params):
    params_copies = []
    for dir in params.xyz:
        params_copy = copy.deepcopy(params)
        params_copy.molecule_source_component = dir
        params_copy.molecule_source_field = QUANTUMSOURCE(params_copy).field
        params_copy.dir_path = f"{dir}_dir"
        params_copy.field_e_filepath = getattr(params_copy, f'field_e_{dir}_filepath')
        params_copy.field_p_filepath = getattr(params_copy, f'field_p_{dir}_filepath')
        params_copy.spectra_e_vs_p_filepath = getattr(params_copy, f'spectra_e_{dir}_vs_p_{dir}_filepath')
        if not params.resumed_from_checkpoint:
            os.makedirs(params_copy.dir_path, exist_ok=True)
        params_copies.append(params_copy)
    return params_copies

def run(params):
    if getattr(params, 'has_plasmon', False) and getattr(params, 'has_checkpoint', False):
        params.has_checkpoint = False

    # Log running fourier with gamma
    if params.has_plasmon:
        params_copies = _set_up_params_copy_plasmol(params)
        simulation = "plasmol"
    else:
        params_copies = _set_up_params_copy_molecule(params)
        simulation = "molecule"

    logger.info(f"Running {len(params_copies)} directional {simulation} simulations in parallel...")

    try:
        with ProcessPoolExecutor(max_workers=len(params_copies)) as executor:
            if params.has_plasmon:
                future_to_dir = {
                    executor.submit(_run_plasmol_with_prefix, params_copy): params_copy.plasmon_source_component
                    for params_copy in params_copies
                }
            else:
                future_to_dir = {
                    executor.submit(_run_quantum_with_prefix, params_copy): params_copy.molecule_source_component
                    for params_copy in params_copies
                }

            for future in as_completed(future_to_dir):
                direction = future_to_dir[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"{direction}-dir {simulation} run failed: {e}")
    finally:
        if getattr(params, 'has_checkpoint', False):
            reg_fp = getattr(params, 'checkpoint_filepath', None)
            if reg_fp:
                try:
                    merge_per_direction_checkpoints(params, reg_fp)
                except Exception as me:
                    logger.error(f"Failed to merge per-direction regular checkpoints: {me}")

            final_fp = getattr(params, 'final_checkpoint_filepath', None)
            if final_fp:
                try:
                    merge_final_checkpoints(params, final_fp)
                except Exception as me:
                    logger.error(f"Failed to merge per-direction final checkpoints: {me}")

        for params_copy in params_copies:
            params_copy.field_p_filepath = _apply_tau_damping(params_copy.field_p_filepath, params.fourier_tau, params.time_rounding_decimals)

        time_points, dipole_moment = fold(params_copies[0].field_p_filepath, params_copies[1].field_p_filepath, params_copies[2].field_p_filepath)
        
        abs_imag, freqs = fourier(
            time_points,
            dipole_moment,
            params.fourier_gamma,
            params.fourier_min_ev,
            params.fourier_max_ev,
            npz=getattr(params, 'fourier_npz_filepath', None),
        )

        abs_vals = absorption(abs_imag, freqs)
        if len(freqs) == 0:
            raise ValueError("No valid frequencies found for Fourier transform. Try running the simulation for longer.")

        peak = max(abs_vals)
        normalized = abs_vals / peak if peak else abs_vals
        pd.DataFrame({'Frequency': freqs, 'Absorption': normalized}).to_csv(
            Path(params.fourier_spectrum_filepath).with_suffix(".csv"), index=False
        )
        plt.figure(figsize=(14, 8))
        plt.plot(freqs, normalized, color='green', label='Spectrum')
        plt.xlabel('Energy (eV)', fontsize=16)
        plt.ylabel('Absorption', fontsize=16)
        plt.title('Absorption Spectrum', fontsize=20)
        plt.xlim(params.fourier_min_ev, params.fourier_max_ev)
        plt.grid(True)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(params.fourier_spectrum_filepath, dpi=600)
