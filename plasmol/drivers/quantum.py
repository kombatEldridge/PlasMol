# drivers/quantum.py
import os
import csv
import logging
import numpy as np

from plasmol.quantum.molecule import MOLECULE

from plasmol.quantum.propagators import *
from plasmol.quantum.propagation import propagation
from plasmol.utils.checkpoint import update_checkpoint

from plasmol.utils.plotting import plot_fields
from plasmol.utils.csv import init_csv, update_csv, read_field_csv
from plasmol.utils.checkpoint import resume_from_checkpoint

def run(params):
    logger = logging.getLogger("main")

    if params.resume_from_checkpoint:
        resume_from_checkpoint(params)
    else:
        init_csv(params.field_e_filepath, "Electric Field intensity in atomic units")
        init_csv(params.field_p_filepath, "Molecule's Polarizability Field intensity in atomic units")
        logger.debug(f"Field files successfully initialized: {params.field_e_filepath} and {params.field_p_filepath}")
        rows = ((t, i0, i1, i2) for t, (i0, i1, i2) in zip(params.times, params.molecule_source_field))
        with open(params.field_e_filepath, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
        logger.debug(f"Electric field initialized in {params.field_e_filepath}.")

    molecule = MOLECULE(params)
    
    logger.debug(f"Electric field successfully added to {params.field_e_filepath}")

    for index, current_time in enumerate(params.times):
        if params.resume_from_checkpoint:
            if current_time < params.checkpoint_dict['checkpoint_time']:
                continue
        mu_arr = propagation(params.molecule_propagator_params, molecule, params.molecule_source_field[index], params.molecule_propagator)
        logging.info(f"{params.molecule_source_component}-dir: At {current_time} au, the induced dipole is {mu_arr} in au")
        update_csv(params.field_p_filepath, current_time, *mu_arr)
        if params.has_checkpoint and index % params.checkpoint_snapshot_frequency == 0 and not current_time == 0.0:
            update_checkpoint(params, molecule, current_time)
            logger.info(f"Checkpoint updated at time {current_time}")


    base, _ = os.path.splitext(params.spectra_e_vs_p_filepath)
    plot_fields(params.field_e_filepath, params.field_p_filepath, output_image_path=base)
