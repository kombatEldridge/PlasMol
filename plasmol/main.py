# main.py
import os
import sys

if __name__ == "__main__" and __package__ is None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    __package__ = os.path.basename(current_dir)

import logging
import numpy as np

from plasmol import constants
from plasmol.drivers import *
from plasmol.utils.logging import PRINTLOGGER
from plasmol.utils.input.cli import parse_arguments
from plasmol.utils.input.params import PARAMS

if __name__ == "__main__":
    # Set up logging
    log_format = '%(levelname)s: %(message)s'
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    # Step 1: Grab CLI args
    args = parse_arguments()

    if args.verbose >= 2:
        logger.setLevel(logging.DEBUG)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    # Use FileHandler if a log file is specified; otherwise, use StreamHandler.
    if args.log:
        handler = logging.FileHandler(args.log, mode='w')
    else:
        handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)
    logger.propagate = False

    sys.stdout = PRINTLOGGER(logger, logging.INFO)
    logging.getLogger("h5py").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("PIL").setLevel(logging.INFO)

    # Step 2: Fill in PARAMS dataclass
    params = PARAMS(args)
    logger.debug(f"PARAMS successfully created: ")
    for key, value in vars(params).items():
        logger.debug(f"\t{key}: {value}")
    
    if args.verbose >= 2:
        with open("run_parameters", "w") as f:
            f.write("=== Run Parameters ===\n\n")
            for key in sorted(vars(params).keys()):
                value = getattr(params, key)
                f.write(f"{key}: {value}\n")

    logger.info(f"The timestep for this simulation is {params.dt} au (roughly {np.round(params.dt / constants.T_AU_FS, decimals=5)} fs).")
    logger.info(f"The simulation will propagate until {params.t_end} au (roughly {np.round(params.t_end / constants.T_AU_FS, decimals=5)} fs).")

    # Step 3: Execute proper workflow
    if params.run_molecule_simulation:
        if params.has_comparison:
            run_comparison(params)
        elif params.has_fourier:
            run_fourier(params)
        else:
            run_quantum(params)
    elif params.run_plasmon_simulation:
        run_classical(params)
    else:
        run_plasmol(params)
