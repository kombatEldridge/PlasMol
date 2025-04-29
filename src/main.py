import os
import sys
import logging
import numpy as np

from params import PARAMS
from molecule import MOLECULE
from logging_utils import PRINTLOGGER
from electric_field import ELECTRICFIELD

from fourier import fourier
from csv_utils import initCSV
from cli import parse_arguments
from propagation import propagation
from plotting import show_eField_pField

# main.py
if __name__ == "__main__":
    try:
        # Set up logging
        log_format = '%(levelname)s: %(message)s'
        args = parse_arguments()
        logger = logging.getLogger()
        # Clear any pre-existing handlers.
        if logger.hasHandlers():
            logger.handlers.clear()

        # Set log level based on verbosity.
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

        # Optional: Prevent propagation to avoid duplicate logging
        logger.propagate = False
    
        sys.stdout = PRINTLOGGER(logger, logging.INFO)
        logging.getLogger("h5py").setLevel(logging.INFO)
    
        T_AU_FS = 41.3413733  # Time units in au/fs

        # Convert time in fs to au
        if (args.time_units == 'fs'):
            dt_au = args.dt * T_AU_FS
            dt_fs = args.dt
            t_end = args.t_end * T_AU_FS
            t_end_fs = args.t_end
        elif (args.time_units == 'au'):
            dt_au = args.dt
            dt_fs = args.dt / T_AU_FS
            t_end = args.t_end
            t_end_fs = args.t_end / T_AU_FS
        else: 
            raise ValueError(f"The timestep unit for this simulation should only either be 'fs' or 'au'.")

        # Time step check
        if (dt_au > 0.1):
            raise ValueError(f"The timestep for this simulation are too large to elicit physical results.")

        time_values = np.arange(0, t_end + dt_au, dt_au)
        interpolated_times = np.linspace(0, time_values[-1], int(len(time_values) * args.mult))
        logger.info(f"The timestep for this simulation is {dt_au} in au or {dt_fs} in fs.")
        logger.info(f"The simulation will propagate until {t_end} in au or {t_end_fs} in fs.")
        logger.info(f"There will be {len(interpolated_times)} timesteps until the simulation finishes.")

        logger.debug(f"Arguments given and parsed successfully: {args}")

        logger.debug("Building an interpolation profile with ElectricFieldInterpolator")
        field = ELECTRICFIELD(interpolated_times, args.dir, peak_time_au=5.0, 
                              width_steps=5, dt=dt_au, shape='kick', 
                              smoothing=False, intensity_au=5e-5)

        # Initialize the interpolated electric field CSV file using initCSV
        interpolated_e_field_csv = "interpolated-E-Field.csv"
        initCSV(interpolated_e_field_csv, "Interpolated Electric Field in au.", args.time_units)

        # Append the interpolated data rows to the CSV file
        with open(interpolated_e_field_csv, 'a', newline='') as csvfile:
            import csv
            writer = csv.writer(csvfile)
            for t, intensity in zip(interpolated_times, field.field):
                writer.writerow([t, intensity[0], intensity[1], intensity[2]])

        # Initialize CSV file for the polarizability field output
        polarizability_csv = "magnus-P-Field.csv"
        initCSV(polarizability_csv, "Molecule's Polarizability Field in au.", args.time_units)

        # Log non-comment lines from the Bohr input file
        bohr_input_path = os.path.abspath(args.bohr)
        with open(bohr_input_path, 'r') as bohr_file:
            for line in bohr_file:
                if not line.strip().startswith(('#', '--', '%')):
                    logger.info('\t%s', line.rstrip('\n'))
    
        params = PARAMS(
            pcconv=args.pcconv, 
            tol_zero=args.tol_zero, 
            doublecheck=args.doublecheck, 
            exp_method=args.exp_method, 
            dt=dt_au, 
            terms_interpol=args.terms_interpol,
            max_iter=args.max_iter)

        molecule = MOLECULE(args.bohr)

        propagation(params, molecule, field, polarizability_csv)

        # Plot the results using the interpolated electric field data
        show_eField_pField(interpolated_e_field_csv, polarizability_csv)
        fourier(polarizability_csv)
        logging.info("Simulation completed successfully.")

    except Exception as err:
        logging.exception(f"Simulation aborted due to an error: {err}")
