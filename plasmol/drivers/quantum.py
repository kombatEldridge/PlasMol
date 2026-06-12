# drivers/quantum.py
import math
import os
import csv
import logging
import numpy as np

from plasmol.quantum.molecule import MOLECULE

from plasmol.quantum.propagators import *
from plasmol.quantum.propagation import propagation
from plasmol.utils.checkpoint import add_field_e_checkpoint, update_checkpoint, init_checkpoint, cleanup_checkpoint

from plasmol.utils.plotting import plot_fields
from plasmol.utils.csv import init_csv, update_csv, read_field_csv

def run(params):
    logger = logging.getLogger("main")
    logger.info(" === RUNNING QUANTUM ONLY SIMULATION === ")
    if not params.resumed_from_checkpoint:
        # Only initialize CSV files for new runs (checkpoint runs already have them)
        init_csv(params.field_e_filepath, "Electric Field intensity in atomic units")
        init_csv(params.field_p_filepath, "Molecule's Polarizability Field intensity in atomic units")
        logger.debug(f"Field files successfully initialized: {params.field_e_filepath} and {params.field_p_filepath}")
        rows = [(round(t, params.time_rounding_decimals), i0, i1, i2) for t, (i0, i1, i2) in zip(params.times, params.molecule_source_field)]
        for row in rows:
            update_csv(params.field_e_filepath, *row)
        logger.debug(f"Electric field initialized in {params.field_e_filepath}.")
    else:
        # find index within params.times for checkpoint_time
        dir_component = getattr(params, 'molecule_source_component') if params.has_fourier else None
        suffix = f"_{dir_component}" if params.has_fourier and dir_component else ""
        checkpoint_time = params.values_from_checkpoint[f"checkpoint_time{suffix}"]
        index = next(i for i, t in enumerate(params.times) if t >= checkpoint_time) + 1
        rows = [(round(t, params.time_rounding_decimals), i0, i1, i2) for t, (i0, i1, i2) in zip(params.times[index:], params.molecule_source_field[index:])]
        for row in rows:
            update_csv(params.field_e_filepath, *row)

    if params.has_checkpoint:
        add_field_e_checkpoint(params, params.field_e_filepath)

    molecule = MOLECULE(params)

    total_steps = len(params.times)
    report_interval = max(1, total_steps // 10)
    time = 0.0 
    source_has_been_zero = True
    try:
        for index, current_time in enumerate(params.times):
            if params.resumed_from_checkpoint:
                dir_component = getattr(params, 'molecule_source_component') if params.has_fourier else None
                suffix = f"_{dir_component}" if params.has_fourier and dir_component else ""
                if params.times[-1] < params.values_from_checkpoint[f'checkpoint_time{suffix}']:
                    raise ValueError(f"The latest checkpoint time {params.values_from_checkpoint[f'checkpoint_time{suffix}']} is past the t_end. Please adjust your input file to continue past this point.")
                if params.times[-1] == params.values_from_checkpoint[f'checkpoint_time{suffix}']:
                    time = current_time
                if current_time <= params.values_from_checkpoint[f'checkpoint_time{suffix}']:
                    continue
            if current_time == params.times[-1]:
                break
            if current_time == 0:
                update_csv(params.field_p_filepath, current_time, *np.zeros(3))
            if (params.molecule_source_field[index] == 0).all() and source_has_been_zero:
                mu_arr = np.zeros(3)
            else:
                mu_arr = propagation(params.molecule_propagator_params, molecule, params.molecule_source_field[index], params.molecule_propagator)
                source_has_been_zero = False
            logging.debug(f"At {np.round(params.times[index+1], params.time_rounding_decimals)} au, the induced dipole is {mu_arr} in au")
            update_csv(params.field_p_filepath, round(params.times[index+1], params.time_rounding_decimals), *mu_arr)
            time = current_time
            if (index) % report_interval == 0 or (index) == total_steps:
                percent = int(round((index) / total_steps * 100))
                logger.info(f"Simulation progress: {percent}% done ({index}/{total_steps} steps || {time}/{params.times[-1]} au)")
            if params.has_checkpoint and not current_time == 0.0:
                n_steps = round(index / params.checkpoint_frequency_steps)
                reconstructed = n_steps * params.checkpoint_frequency_steps
                if math.isclose(reconstructed, index, rel_tol=1e-9, abs_tol=1e-12):
                    update_checkpoint(params, molecule, current_time)
                    params.checkpoint_written_after_init = True
    except Exception as e:
        logger.error(f"An error occurred during simulation: {e}")
    finally:
        add_field_e_checkpoint(params, params.field_e_filepath, params.final_checkpoint_filepath)
        update_checkpoint(params, molecule, time, params.final_checkpoint_filepath)
        params.final_checkpoint_written_after_init = True
        base, _ = os.path.splitext(params.spectra_e_vs_p_filepath)
        plot_fields([(params.field_e_filepath, 'Incident Electric Field'), (params.field_p_filepath, 'Molecule\'s Response')], output_image_path=base)


