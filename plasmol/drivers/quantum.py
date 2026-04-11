# drivers/quantum.py
import os
import csv
import logging
import numpy as np

from plasmol.quantum.molecule import MOLECULE

from plasmol.quantum.propagators import *
from plasmol.quantum.propagation import propagation
from plasmol.utils.checkpoint import add_field_e_checkpoint, update_checkpoint

from plasmol.utils.plotting import plot_fields
from plasmol.utils.csv import init_csv, update_csv, read_field_csv

def run(params):
    logger = logging.getLogger("main")

    if not params.resumed_from_checkpoint:
        # Only initialize CSV files for new runs (checkpoint runs already have them)
        init_csv(params.field_e_filepath, "Electric Field intensity in atomic units")
        init_csv(params.field_p_filepath, "Molecule's Polarizability Field intensity in atomic units")
        logger.debug(f"Field files successfully initialized: {params.field_e_filepath} and {params.field_p_filepath}")
        rows = ((t, i0, i1, i2) for t, (i0, i1, i2) in zip(params.times, params.molecule_source_field))
        with open(params.field_e_filepath, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
        logger.debug(f"Electric field initialized in {params.field_e_filepath}.")
        add_field_e_checkpoint(params, params.field_e_filepath)
    else:
        logger.debug("Resuming from checkpoint - skipping CSV initialization.")

    molecule = MOLECULE(params)
    
    logger.debug(f"Electric field successfully added to {params.field_e_filepath}")

    for index, current_time in enumerate(params.times):
        if params.resumed_from_checkpoint:
            dir_component = getattr(params, 'molecule_source_dict', {}).get('component') if params.has_fourier else None
            suffix = f"_{dir_component}" if params.has_fourier and dir_component else ""
            if current_time <= params.checkpoint_dict[f'checkpoint_time{suffix}']:
                continue
        mu_arr = propagation(params.molecule_propagator_params, molecule, params.molecule_source_field[index], params.molecule_propagator)
        logging.info(f"At {np.round(current_time, params.time_rounding_decimals)} au, the induced dipole is {mu_arr} in au")
        update_csv(params.field_p_filepath, current_time, *mu_arr)
        if params.has_checkpoint and index % params.checkpoint_snapshot_frequency == 0 and not current_time == 0.0:
            update_checkpoint(params, molecule, current_time)

    base, _ = os.path.splitext(params.spectra_e_vs_p_filepath)
    plot_fields([(params.field_e_filepath, 'Incident Electric Field'), (params.field_p_filepath, 'Molecule\'s Response')], output_image_path=base)

