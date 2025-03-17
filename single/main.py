# main.py
import os
import sys
import logging
import numpy as np
# import meep as mp  # type: ignore
import threading
import queue

from molecule import MOLECULE
from logging_utils import PrintLogger
from interpolation import ElectricFieldInterpolator
from csv_utils import initCSV, read_electric_field_csv, updateCSV
from plotting import show_eField_pField
from cli import parse_arguments
from mu import run

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
    
        # mp.verbosity(min(3, args.verbose))
        sys.stdout = PrintLogger(logger, logging.INFO)
        logging.getLogger("h5py").setLevel(logging.INFO)
    
        logger.debug(f"Arguments given and parsed correctly: {args}")
    
        # Initialize the molecule and extract parameters
        molecule = MOLECULE(args.bohr, args.pcconv)
    
        # Read the original electric field CSV file
        time_values, electric_x, electric_y, electric_z = read_electric_field_csv(args.csv)
    
        # Create an interpolator for the electric field components
        logger.debug("Building an interpolation profile with ElectricFieldInterpolator")

        # Electric fields are given in au and time in fs
        field_interpolator = ElectricFieldInterpolator(time_values, electric_x, electric_y, electric_z)
    
        # Generate a new time grid with adjusted resolution (args.mult is a multiplier)
        interpolated_times = np.linspace(0, time_values[-1], int(len(time_values) * args.mult))
        time_step_fs = interpolated_times[1] - interpolated_times[0]
        interpolated_fields = field_interpolator.get_field_at(interpolated_times)

        # Convert time in fs to au
        dt = time_step_fs * 41.3413733

        # Initialize the interpolated electric field CSV file using initCSV
        interpolated_e_field_csv = "interpolated-E-Field.csv"
        initCSV(interpolated_e_field_csv, "Interpolated Electric Field measured in au.")
    
        # Append the interpolated data rows to the CSV file
        with open(interpolated_e_field_csv, 'a', newline='') as csvfile:
            import csv
            writer = csv.writer(csvfile)
            for t, field in zip(interpolated_times, interpolated_fields):
                writer.writerow([t, field[0], field[1], field[2]])
    
        # Initialize CSV file for the polarizability field output
        polarizability_csv = "magnus-P-Field.csv"
        initCSV(polarizability_csv, "Molecule's Polarizability Field measured in au.")
    
        # Log non-comment lines from the Bohr input file
        bohr_input_path = os.path.abspath(args.bohr)
        with open(bohr_input_path, 'r') as bohr_file:
            for line in bohr_file:
                if not line.strip().startswith(('#', '--', '%')):
                    logger.info('\t%s', line.rstrip('\n'))
        
        for index, current_time in enumerate(interpolated_times):
            exc = interpolated_fields[index]
            collapsed_output = run(dt, molecule, exc)
            logging.debug(f"At {current_time} fs, combined Bohr output is {collapsed_output} in au")
            updateCSV(polarizability_csv, current_time, *collapsed_output)

        # Plot the results using the interpolated electric field data
        show_eField_pField(interpolated_e_field_csv, polarizability_csv)
        logging.info("Simulation completed successfully.")
    except Exception as err:
        logging.exception(f"Simulation aborted due to an error: {err}")