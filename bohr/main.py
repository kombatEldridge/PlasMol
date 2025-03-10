# main.py
import os
import sys
import logging
import numpy as np
import meep as mp  # type: ignore

from molecule_parallel import MOLECULE
from logging_utils import PrintLogger
from interpolation import ElectricFieldInterpolator
from volume import get_volume
from magnus_parallel import propagate_direction_worker, combine_propagation_results
from csv_utils import initCSV, read_electric_field_csv
from plotting import show_eField_pField
from cli import parse_arguments
import threading
import queue

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
    
        mp.verbosity(min(3, args.verbose))
        sys.stdout = PrintLogger(logger, logging.INFO)
        logging.getLogger("h5py").setLevel(logging.INFO)
    
        # Initialize the molecule and extract parameters
        molecule = MOLECULE(args.bohr)
    
        # Read the original electric field CSV file
        time_values, electric_x, electric_y, electric_z = read_electric_field_csv(args.csv)
    
        # Create an interpolator for the electric field components
        field_interpolator = ElectricFieldInterpolator(time_values, electric_x, electric_y, electric_z)
    
        # Generate a new time grid with adjusted resolution (args.mult is a multiplier)
        interpolated_times = np.linspace(0, time_values[-1], int(len(time_values) * args.mult))
        time_step = interpolated_times[1] - interpolated_times[0]
        interpolated_fields = field_interpolator.get_field_at(interpolated_times)
    
        # Initialize the interpolated electric field CSV file using initCSV
        interpolated_e_field_csv = "interpolated-E-Field.csv"
        initCSV(interpolated_e_field_csv, "Interpolated Electric Field measured in atomic units.")
    
        # Append the interpolated data rows to the CSV file
        with open(interpolated_e_field_csv, 'a', newline='') as csvfile:
            import csv
            writer = csv.writer(csvfile)
            for t, field in zip(interpolated_times, interpolated_fields):
                writer.writerow([t, field[0], field[1], field[2]])
    
        # Initialize CSV file for the polarizability field output
        polarizability_csv = "magnus-P-Field.csv"
        initCSV(polarizability_csv, "Molecule's Polarizability Field measured in atomic units.")
    
        # Log non-comment lines from the Bohr input file
        bohr_input_path = os.path.abspath(args.bohr)
        with open(bohr_input_path, 'r') as bohr_file:
            for line in bohr_file:
                if not line.strip().startswith(('#', '--', '%')):
                    logger.info('\t%s', line.rstrip('\n'))
    
        # Create a thread-safe queue for each spatial direction: 0 -> x, 1 -> y, 2 -> z.
        result_queues = {d: queue.Queue() for d in range(3)}
        threads = []
    
        # Launch one worker thread per direction.
        for d in range(3):
            t = threading.Thread(target=propagate_direction_worker,
                                 args=(None, d, interpolated_times,
                                       interpolated_fields, time_step, molecule, result_queues[d]),
                                 kwargs={'propagator': None})
            # We pass the propagate_density_matrix function via kwargs; here, we override the placeholder.
            # Alternatively, you can import it from propagation and pass it directly.
            from magnus import propagate_density_matrix
            t = threading.Thread(target=propagate_direction_worker,
                                 args=(propagate_density_matrix, d, interpolated_times,
                                       interpolated_fields, time_step, molecule, result_queues[d]))
            t.start()
            threads.append(t)
    
        volume = get_volume(molecule.molecule["coords"])
    
        # In the main thread, combine results as soon as a complete set is available.
        combine_propagation_results(interpolated_times, molecule, result_queues, polarizability_csv, volume)
    
        # Wait for all worker threads to finish.
        for t in threads:
            t.join()
    
        # Plot the results using the interpolated electric field data
        show_eField_pField(interpolated_e_field_csv, polarizability_csv)
        logging.info("Simulation completed successfully.")
    except Exception as err:
        logging.exception(f"Simulation aborted due to an error: {err}")