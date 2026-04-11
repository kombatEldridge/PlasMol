# drivers/classical.py
import os
import sys
import logging
from plasmol.classical.simulation import SIMULATION
from plasmol.utils.checkpoint import init_checkpoint

def run(params):
    try:
        logger = logging.getLogger("main")
        plasmon = SIMULATION(params)
        plasmon.run()
        logging.info("Simulation completed successfully.")
    except Exception as err:
        logger.error(f"Simulation failed: {err}", exc_info=True)
        sys.exit(1)
