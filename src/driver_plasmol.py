# bohr.py
# /Users/bldrdge1/.conda/envs/meep1.29/bin/python /Users/bldrdge1/Downloads/repos/PlasMol/bohr/driver.py -m meep.in -b pyridine.in -l plasmol.log -vv
import os
import sys
import logging
import pprint
import numpy as np

import os
# import re
import sys
# import sources
import logging
import argparse
# import meep as mp
# import simulation as sim
# from datetime import datetime

import constants
from params import PARAMS
from molecule import MOLECULE
from logging_utils import PRINTLOGGER
from electric_field import ELECTRICFIELD


from propagation import propagation
from plotting import show_eField_pField
from csv_utils import initCSV, updateCSV
from cli import parse_arguments, minimum_cli_sufficiency
from input_parser import parsePlasMolInputFile, parseRTTDDFTInputFile, minimum_molecule_sufficiency

# main.py
def run(params):
    try:
        logger = logging.getLogger("main")

        dt_au = args.dt
        t_end = args.t_end

        time_values = np.arange(0, t_end + dt_au, dt_au)
        interpolated_times = np.linspace(0, time_values[-1], int(len(time_values) * args.mult))
        logger.info(f"The timestep for this simulation is {dt_au} in au or {dt_au / constants.T_AU_FS} in fs.")
        logger.info(f"The simulation will propagate until {t_end} in au or {t_end / constants.T_AU_FS} in fs.")
        logger.debug(f"There will be {len(interpolated_times)} timesteps until the simulation finishes.")
        logger.debug(f"Arguments given and parsed successfully: {args}")
        
        if args.restart:
            for path in ['eField.csv', 'pField.csv', 'pField_spectrum.csv', 'chkfile.npz', 'output.png', 'spectrum.png']:
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                        logger.info(f"Deleted {path}")
                    except OSError as e:
                        logger.error(f"Error deleting {path}: {e}")
                else:
                    logger.debug(f"No such file: {path}")

        params = PARAMS(
            pcconv=args.pcconv, 
            tol_zero=args.tol_zero, 
            dt=dt_au, 
            max_iter=args.max_iter, 
            chkfile=args.chkfile,
            chkfile_path=args.chkfile_path,
            chkfile_freq=args.chkfile_freq,
            peak_time_au=args.peak_time_au,
            width_steps=args.width_steps,
            shape=args.shape,
            smoothing=args.smoothing,
            intensity_au=args.intensity_au,
            eFieldFile=args.eFieldFile,
            pFieldFile=args.pFieldFile,
            )
        
        molecule = MOLECULE(args.bohr, params)

        logger.debug("Building an interpolation profile with ElectricFieldInterpolator")
        field = ELECTRICFIELD(
            interpolated_times, 
            args.dir,
            peak_time_au=params.peak_time_au,
            width_steps=params.width_steps,
            dt=dt_au,
            shape=params.shape,
            smoothing=params.smoothing,
            intensity_au=params.intensity_au,
        )

        if molecule.chkfile_path is not None and os.path.exists(molecule.chkfile_path):
            # assume the eField and pField files have already been built and you do not need to re-initialize them
            logger.debug(f"Checkpoint file {molecule.chkfile_path} found. Skipping electric/polarizability field generation.")
            interpolated_e_field_csv = params.eFieldFile
            polarizability_csv = params.pFieldFile
        else:            
            interpolated_e_field_csv = "eField.csv"  # TODO: add to input file
            initCSV(interpolated_e_field_csv, "Electric Field intensity in atomic units")

            # Initialize CSV file for the polarizability field output
            polarizability_csv = "pField.csv"  # TODO: add to input file
            initCSV(polarizability_csv, "Molecule's Polarizability Field intensity in atomic units")
            
        # Append the interpolated data rows to the CSV file
        for t, intensity in zip(interpolated_times, field.field):
            updateCSV(interpolated_e_field_csv, t, intensity[0], intensity[1], intensity[2])
        
        logger.debug(f"Electric field successfully built and saved to {interpolated_e_field_csv}")

        # Log non-comment lines from the Bohr input file
        logger.info("Bohr input file processed:")
        bohr_input_path = os.path.abspath(args.bohr)
        with open(bohr_input_path, 'r') as bohr_file:
            for line in bohr_file:
                if not line.strip().startswith(('#', '--', '%')):
                    logger.info('\t%s', line.rstrip('\n'))

        logger.debug(f"Moleucle's response will be saved to {polarizability_csv}")
    
        # Main code call
        propagation(params, molecule, field, polarizability_csv)

        # from driver.py
        # -----------------------
        meepinputfile = args.meep
        bohrinputfile = args.bohr if args.bohr else None
        simDriver = parseInputFile(meepinputfile)
        logging.info("Input file successfully parsed. Beginnin" \
        "g simulation")
        simDriver.run()
        # -----------------------

        # Plot the results using the interpolated electric field data
        show_eField_pField(interpolated_e_field_csv, polarizability_csv)
        logging.info("Simulation completed successfully.")
    except Exception as err:
        logger.error(f"Simulation failed: {err}", exc_info=True)
        sys.exit(1)