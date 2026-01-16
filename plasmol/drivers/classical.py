# drivers/classical.py
import os
import sys
import logging
from plasmol.classical.simulation import SIMULATION

def run(params):
    try:
        logger = logging.getLogger("main")
                
        simDriver = SIMULATION(params)
        simDriver.run()
        
        logging.info("Simulation completed successfully.")
    except Exception as err:
        logger.error(f"Simulation failed: {err}", exc_info=True)
        sys.exit(1)