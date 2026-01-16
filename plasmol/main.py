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
from plasmol.utils.input.parser import parseInputFile

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

        # Step 2: Parse input file
        parsed = parseInputFile(args)
        logger.debug(f"Arguments given and parsed successfully: {parsed}")

        # Step 3: Merge all found parameters
        logger.debug("Merging parameters from input file(s) with the CLI inputs. CLI takes priority for duplicate values.")
        params = PARAMS(parsed)
        
        logger.debug(f"Arguments successfully recieved: ")
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
            # if hasattr(params, 'bases') and params.bases:  # Check for comparison mode
            #     from plasmol.drivers import run_comparison
            #     logger.info("Running comparison of MO energies and Gamma matrices across basis sets and XC functionals.")
            #     run_comparison(params)
            # else:
                run_quantum(params)
        elif params.type == 'Classical':
            run_classical(params)

    except Exception as err:
        logger.error(f"Simulation failed: {err}", exc_info=True)
        sys.exit(1)