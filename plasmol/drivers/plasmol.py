# drivers/plasmol.py
import os
import sys
import logging

from plasmol.classical.simulation import SIMULATION
from plasmol.quantum.molecule import MOLECULE
from plasmol.utils.plotting import plot_fields
from plasmol.utils.csv import init_csv, read_field_csv

def run(params):
    try:
        logger = logging.getLogger("main")
        
        molecule = MOLECULE(params)

        if params.checkpoint_filepath is not None and os.path.exists(params.checkpoint_filepath):
            try:
                _ = read_field_csv(params.field_e_filepath)
                _ = read_field_csv(params.field_p_filepath)
                logger.debug(f"Checkpoint file {params.checkpoint_filepath} found as well as properly formatted field fields: {params.field_e_filepath} and {params.field_p_filepath}. Skipping electric/polarizability file generation.")
            except Exception as e:
                print(f"Error reading file: {e}")
        else:            
            init_csv(params.field_e_filepath, "Electric Field intensity in atomic units")
            init_csv(params.field_p_filepath, "Molecule's Polarizability Field intensity in atomic units")
            logger.debug(f"Field files successfully initialized: {params.field_e_filepath} and {params.field_p_filepath}")
        
        simDriver = SIMULATION(params, molecule)
        simDriver.run()
        
        # Plot the results using the interpolated electric field data
        plot_fields(params.field_e_filepath, params.field_p_filepath)
        logging.info("Simulation completed successfully.")
    except Exception as err:
        logger.error(f"Simulation failed: {err}", exc_info=True)
        sys.exit(1)