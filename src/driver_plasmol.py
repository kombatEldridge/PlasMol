# driver_plasmol.py
import os
import sys
import logging

import simulation as sim

from molecule import MOLECULE
from plotting import show_eField_pField
from csv_utils import initCSV, read_field_csv

def run(params):
    try:
        logger = logging.getLogger("main")
        if params.restart:
            for path in [params.eField_path, params.pField_path, params.pField_Transform_path, params.chkfile_path, params.eField_vs_pField_path, params.eV_spectrum_path]:
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                        logger.info(f"Deleted {path}")
                    except OSError as e:
                        logger.error(f"Error deleting {path}: {e}")
                else:
                    logger.debug(f"No such file: {path}")
        
        molecule = MOLECULE(params)

        if params.chkfile_path is not None and os.path.exists(params.chkfile_path):
            try:
                _ = read_field_csv(params.eField_path)
                _ = read_field_csv(params.pField_path)
                logger.debug(f"Checkpoint file {params.chkfile_path} found as well as properly formatted field fields: {params.eField_path} and {params.pField_path}. Skipping electric/polarizability file generation.")
            except Exception as e:
                print(f"Error reading file: {e}")
        else:            
            initCSV(params.eField_path, "Electric Field intensity in atomic units")
            initCSV(params.pField_path, "Molecule's Polarizability Field intensity in atomic units")
            logger.debug(f"Field files successfully initialized: {params.eField_path} and {params.pField_path}")
        
        simDriver = sim.Simulation(params, molecule)
        simDriver.run()

        # Plot the results using the interpolated electric field data
        show_eField_pField(params.eField_path, params.pField_path)
        logging.info("Simulation completed successfully.")
    except Exception as err:
        logger.error(f"Simulation failed: {err}", exc_info=True)
        sys.exit(1)