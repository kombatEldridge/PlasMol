# driver_meep.py
import os
import sources
import logging
# import meep as mp

import simulation as sim

logger = logging.getLogger("main")

def run(params):
    simDriver = setParameters(params)
    simDriver.run()
    return
