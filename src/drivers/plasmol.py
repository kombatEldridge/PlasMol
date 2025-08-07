# drivers/plasmol.py
import os
import sys
import logging

from ..classical.simulation import Simulation
from ..quantum.molecule import MOLECULE
from ..utils.plotting import show_eField_pField
from ..utils.csv import initCSV, read_field_csv

def run(params):
    try:
        logger = logging.getLogger("main")
        
        molecule = MOLECULE(params)

        if params.checkpoint_path is not None and os.path.exists(params.checkpoint_path):
            try:
                _ = read_field_csv(params.eField_path)
                _ = read_field_csv(params.pField_path)
                logger.debug(f"Checkpoint file {params.checkpoint_path} found as well as properly formatted field fields: {params.eField_path} and {params.pField_path}. Skipping electric/polarizability file generation.")
            except Exception as e:
                print(f"Error reading file: {e}")
        else:            
            initCSV(params.eField_path, "Electric Field intensity in atomic units")
            initCSV(params.pField_path, "Molecule's Polarizability Field intensity in atomic units")
            logger.debug(f"Field files successfully initialized: {params.eField_path} and {params.pField_path}")
        
        simDriver = Simulation(params, molecule)
        simDriver.run()
        
        # Plot the results using the interpolated electric field data
        show_eField_pField(params.eField_path, params.pField_path)
        logging.info("Simulation completed successfully.")
    except Exception as err:
        logger.error(f"Simulation failed: {err}", exc_info=True)
        sys.exit(1)