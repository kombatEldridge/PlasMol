# main.py
import os
import sys

# Dynamically set up package context for direct execution (python main.py)
if __name__ == "__main__" and __package__ is None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    __package__ = os.path.basename(current_dir)  # Sets __package__ to "src" (or whatever your directory is named)

import logging
import numpy as np

from . import constants

from .drivers import *
from .input.params import PARAMS
from .utils.logging import PRINTLOGGER

from .input.cli import parse_arguments
from .input.parser import inputFilePrepare

# main.py
if __name__ == "__main__":
    try:
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

        # Step 2: Identify parsing workflow from CLI args
        preparams = inputFilePrepare(args)
        logger.debug(f"Arguments given and pre-parsed successfully: {preparams}")

        # Step 3: Merge all found parameters
        logger.debug("Merging parameters from input file(s) with the CLI inputs. CLI takes priority for duplicate values.")
        params = PARAMS(preparams)
        
        logger.debug(f"Arguments given and parsed successfully: ")
        for key, value in vars(params).items():
            logger.debug(f"\t\t{key}: {value}")

        if params.restart:
            for attr in ['eField_path', 'pField_path', 'pField_Transform_path', 'checkpoint_path', 'eField_vs_pField_path', 'eV_spectrum_path']:
                if hasattr(params, attr):
                    file_path = getattr(params, attr)
                    if file_path is not None and os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                            logger.info(f"Deleted {file_path}")
                        except OSError as e:
                            logger.error(f"Error deleting {file_path}: {e}")
                    else:
                        logger.debug(f"No such file: {file_path}")

        time_values = np.arange(0, params.t_end + params.dt, params.dt)
        interpolated_times = np.linspace(0, time_values[-1], len(time_values))
        logger.info(f"The timestep for this simulation is {params.dt} in au or {params.dt / constants.T_AU_FS} in fs.")
        logger.info(f"The simulation will propagate until {params.t_end} in au or {params.t_end / constants.T_AU_FS} in fs.")
        logger.debug(f"There will be {len(interpolated_times)} timesteps until the simulation finishes.")

        # Step 4: Execute proper workflow
        if params.type == 'PlasMol':
            run_plasmol(params)
        elif params.type == 'Quantum':
            run_quantum(params)
        elif params.type == 'Classical':
            run_classical(params)

    except Exception as err:
        logger.error(f"Simulation failed: {err}", exc_info=True)
        sys.exit(1)