# drivers/classical.py
import os
import sys
import logging
from ..classical.simulation import Simulation

def run(params):
    try:
        logger = logging.getLogger("main")
                
        simDriver = Simulation(params)
        simDriver.run()
        
        logging.info("Simulation completed successfully.")
    except Exception as err:
        logger.error(f"Simulation failed: {err}", exc_info=True)
        sys.exit(1)