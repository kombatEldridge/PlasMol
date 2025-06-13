# /Users/bldrdge1/.conda/envs/meep1.29/bin/python /Users/bldrdge1/Downloads/repos/PlasMol/bohr/driver.py -m meep.in -b pyridine.in -l plasmol.log -vv
import sys
import logging

import driver_plasmol
import driver_rttddft
import driver_meep

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

        # Step 2: Identify parsing workflow from CLI args
        building_params, simulation_type = inputFilePrepare(args.pmif, args.mif, args.qif)

        # Step 3: Merge all found parameters
        logger.debug("Merging parameters from input file(s) with the CLI inputs. CLI takes priority for duplicate values.")
        
        # TODO, give PARAMS the building_params and args to merge properly
        params = PARAMS()
        logger.debug(f"Arguments given and parsed successfully: {params}")

        # Step 4: Execute proper workflow
        if simulation_type == 'PlasMol':
            driver_plasmol.run(params)
        elif simulation_type == 'RT-TDDFT':
            """
            Needs:
              dt
              t_end
              restart
              eField_path
              pField_path
              pField_Transform_path
              chkfile_path
              eField_vs_pField_path
              eV_spectrum_path
              molecule_coords
              basis
              charge
              spin
              xc
              tol_zero
              chkfile
              wavelength_nm
              peak_time_au
              width_steps
              dir
              qif
              propagator
              max_iter
              pc_convergence
            """
            driver_rttddft.run(params)
        elif simulation_type == 'Meep':
            driver_meep.run(params)
        
        logging.info("Simulation completed successfully.")
    except Exception as err:
        logger.error(f"Simulation failed: {err}", exc_info=True)
        sys.exit(1)