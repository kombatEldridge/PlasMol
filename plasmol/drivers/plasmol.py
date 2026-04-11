# drivers/plasmol.py
import csv
import os
import sys
import logging

from plasmol.classical.simulation import SIMULATION
from plasmol.quantum.molecule import MOLECULE
from plasmol.utils.plotting import plot_fields
from plasmol.utils.csv import init_csv, read_field_csv
from plasmol.utils.checkpoint import add_field_e_checkpoint, update_checkpoint, init_checkpoint

def run(params):
    try:
        logger = logging.getLogger("main")
        
        if not params.resumed_from_checkpoint:
            # Only initialize CSV files for new runs (checkpoint runs already have them)
            init_csv(params.field_e_filepath, "Electric Field intensity in atomic units")
            init_csv(params.field_p_filepath, "Molecule's Polarizability Field intensity in atomic units")
            logger.debug(f"Field files successfully initialized: {params.field_e_filepath} and {params.field_p_filepath}")
        else:
            logger.debug("Resuming from checkpoint - skipping CSV initialization.")
       
        if not params.resumed_from_checkpoint and params.has_checkpoint:
                init_checkpoint(params)

        plasmon = SIMULATION(params)
        plasmon.run()
        
        base, _ = os.path.splitext(params.spectra_e_vs_p_filepath)
        plot_fields([(params.field_e_filepath, 'Incident Electric Field'), (params.field_p_filepath, 'Molecule\'s Response')], output_image_path=base)
        logging.info("Simulation completed successfully.")
    except Exception as err:
        logger.error(f"Simulation failed: {err}", exc_info=True)
        sys.exit(1)