# driver_meep.py
import os
import sys
import logging
from ..meep.simulation import Simulation

def run(params):
    try:
        logger = logging.getLogger("main")
        if params.restart:
            for path in ['eField_path', 'pField_path', 'pField_Transform_path', 'chkfile_path', 'eField_vs_pField_path', 'eV_spectrum_path']:
                if hasattr(params, path):
                    if os.path.isfile(path):
                        try:
                            os.remove(path)
                            logger.info(f"Deleted {path}")
                        except OSError as e:
                            logger.error(f"Error deleting {path}: {e}")
                    else:
                        logger.debug(f"No such file: {path}")
                
        simDriver = Simulation(params)
        simDriver.run()
        
        logging.info("Simulation completed successfully.")
    except Exception as err:
        logger.error(f"Simulation failed: {err}", exc_info=True)
        sys.exit(1)