# /Users/bldrdge1/.conda/envs/meep1.29/bin/python /Users/bldrdge1/Downloads/repos/PlasMol/bohr/driver.py -m meep.in -b pyridine.in -l plasmol.log -vv
import sys
import logging
import numpy as np

import constants
import driver_meep
import driver_rttddft
import driver_plasmol

from params import PARAMS
from cli import parse_arguments
from logging_utils import PRINTLOGGER
from input_parser import inputFilePrepare

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
        
        # Step 3: Merge all found parameters
        logger.debug("Merging parameters from input file(s) with the CLI inputs. CLI takes priority for duplicate values.")
        params = PARAMS(preparams)
        logger.debug(f"Arguments given and parsed successfully: {params}")

        time_values = np.arange(0, params.t_end + params.dt, params.dt)
        interpolated_times = np.linspace(0, time_values[-1], len(time_values))
        logger.info(f"The timestep for this simulation is {params.dt} in au or {params.dt / constants.T_AU_FS} in fs.")
        logger.info(f"The simulation will propagate until {params.t_end} in au or {params.t_end / constants.T_AU_FS} in fs.")
        logger.debug(f"There will be {len(interpolated_times)} timesteps until the simulation finishes.")

        # Step 4: Execute proper workflow
        if params.type == 'PlasMol':
            driver_plasmol.run(params)
        elif params.type == 'RT-TDDFT':
            driver_rttddft.run(params)
        elif params.type == 'Meep':
            driver_meep.run(params)
        
        logger.info("Simulation completed successfully.")
    except Exception as err:
        logger.error(f"Simulation failed: {err}", exc_info=True)
        sys.exit(1)