# driver_rttddft.py
import os
import sys
import csv
import logging
import numpy as np

import constants
from molecule import MOLECULE
from electric_field import ELECTRICFIELD

from propagation import propagation
from plotting import show_eField_pField
from csv_utils import initCSV, updateCSV

def run(params):
    try:
        logger = logging.getLogger("main")

        dt_au = params.dt
        t_end = params.t_end

        time_values = np.arange(0, t_end + dt_au, dt_au)
        interpolated_times = np.linspace(0, time_values[-1], int(len(time_values)))
        logger.info(f"The timestep for this simulation is {dt_au} in au or {dt_au / constants.T_AU_FS} in fs.")
        logger.info(f"The simulation will propagate until {t_end} in au or {t_end / constants.T_AU_FS} in fs.")
        logger.debug(f"There will be {len(interpolated_times)} timesteps until the simulation finishes.")
        
        if params['restart']:
            for path in [params.eField_path, params.pField_path, params.pField_Transform_path, params.chkfile_path, params.eField_vs_pField_path, params.eV_spectrum_path]:
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                        logger.info(f"Restart requested, deleted {path}")
                    except OSError as e:
                        logger.error(f"Error deleting {path}: {e}")
                else:
                    logger.debug(f"Restart requested, but no such file found: {path}. Continuing.")


        molecule = MOLECULE(params)
        field = ELECTRICFIELD(interpolated_times, params)

        if params.chkfile_path is not None and os.path.exists(params.chkfile_path):
            # assume the eField and pField files have already been built and you do not need to re-initialize them
            logger.debug(f"Checkpoint file {params.chkfile_path} found. Skipping electric/polarizability field generation.")
        else:            
            initCSV(params.eField_path, "Electric Field intensity in atomic units")
            initCSV(params.pField_path, "Molecule's Polarizability Field intensity in atomic units")
            
        # Append the interpolated data rows to the CSV file
        rows = ((t, i0, i1, i2) for t, (i0, i1, i2) in zip(interpolated_times, field.field))
        with open(params.eField_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
        
        logger.debug(f"Electric field successfully built and saved to {params.eField_path}")

        # Log non-comment lines from the Bohr input file
        logger.info("Bohr input file processed:")
        bohr_input_path = os.path.abspath(params.qif)
        with open(bohr_input_path, 'r') as bohr_file:
            for line in bohr_file:
                if not line.strip().startswith(('#', '--', '%')):
                    logger.info('\t%s', line.rstrip('\n'))

        logger.debug(f"Moleucle's response will be saved to {params.pField_path}")
    
        # Main code call
        propagation(params, molecule, field)

        # Plot the results using the interpolated electric field data
        show_eField_pField(params.eField_path, params.pField_path)

    except Exception as err:
        logger.error(f"RT-TDDFT failed: {err}", exc_info=True)
        sys.exit(1)