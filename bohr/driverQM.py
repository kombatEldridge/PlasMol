import os
import sys
import csv
import logging
import argparse
import numpy as np
import meep as mp  # type: ignore
from scipy.interpolate import interp1d

import molecule as mol
from bohr import run
import matrix_handler as mh
from csv_handler import initCSV, updateCSV, show_eField_pField


class PrintLogger:
    """Redirects print statements to a logger."""
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, message):
        message = message.rstrip()
        if message:
            self.logger.log(self.level, message)

    def flush(self):
        pass


class ElectricFieldInterpolator:
    """Interpolates electric field components over time."""
    def __init__(self, time_values, electric_x, electric_y, electric_z):
        self.interp_x = interp1d(time_values, electric_x, kind='cubic', fill_value="extrapolate")
        self.interp_y = interp1d(time_values, electric_y, kind='cubic', fill_value="extrapolate")
        self.interp_z = interp1d(time_values, electric_z, kind='cubic', fill_value="extrapolate")

    def get_field_at(self, query_times):
        """Returns the interpolated electric field components at the specified times."""
        x_interp = self.interp_x(query_times)
        y_interp = self.interp_y(query_times)
        z_interp = self.interp_z(query_times)
        return np.column_stack((x_interp, y_interp, z_interp))


def read_electric_field_csv(file_path):
    """Reads electric field values from a CSV file."""
    time_values, electric_x, electric_y, electric_z = [], [], [], []
    
    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        # Skip lines until header is found
        for row in reader:
            if row and row[0].startswith("Timestamps"):
                break
        # Read the data rows
        for row in reader:
            if len(row) < 4:
                continue
            time_values.append(float(row[0]))
            electric_x.append(float(row[1]))
            electric_y.append(float(row[2]))
            electric_z.append(float(row[3]))
    
    return time_values, electric_x, electric_y, electric_z


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Meep simulation with Bohr dipole moment calculation."
    )
    parser.add_argument('-b', '--bohr', type=str, help="Path to the Bohr input file.")
    parser.add_argument('-e', '--csv', type=str, help="Path to the electric field CSV file.")
    parser.add_argument('-m', '--mult', type=float, help="Multiplier for the electric field interpolator resolution.")
    parser.add_argument('-l', '--log', help="Log file name")
    parser.add_argument('-v', '--verbose', action='count', default=0, help="Increase verbosity")
    
    args = parser.parse_args()
    if args.log and args.verbose == 0:
        args.verbose = 1
    
    return args


if __name__ == "__main__":
    # Set up logging
    log_format = '%(levelname)s: %(message)s'
    args = parse_arguments()
    
    logger = logging.getLogger()
    if args.verbose >= 2:
        logger.setLevel(logging.DEBUG)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    
    log_handler = logging.FileHandler(args.log, mode='w') if args.log else logging.StreamHandler(sys.stdout)
    log_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(log_handler)
    
    mp.verbosity(min(3, args.verbose))
    sys.stdout = PrintLogger(logger, logging.INFO)
    logging.getLogger("h5py").setLevel(logging.INFO)
    
    # Clear previous matrix files
    mh.clear_Matrix_Files()

    # Initialize the molecule and extract parameters
    molecule = mol.MOLECULE(args.bohr)
    propagator_method = molecule.method["propagator"]
    molecular_coordinates = molecule.molecule["coords"]
    wavefunction = molecule.wfn
    initial_dipole_moment = mh.get_D_mo_0()
    
    # Read the original electric field CSV file
    time_values, electric_x, electric_y, electric_z = read_electric_field_csv(args.csv)
    
    # Create an interpolator for the electric field components
    field_interpolator = ElectricFieldInterpolator(time_values, electric_x, electric_y, electric_z)
    
    # Generate a new time grid with adjusted resolution (args.mult is a multiplier)
    interpolated_times = np.linspace(0, time_values[-1], int(len(time_values) * args.mult))
    convertTimefs2Atomic = 1 / 0.024188843
    time_step = (interpolated_times[1] - interpolated_times[0]) * convertTimefs2Atomic
    interpolated_fields = field_interpolator.get_field_at(interpolated_times)
    
    # Initialize the interpolated electric field CSV file using initCSV
    interpolated_e_field_csv = "interpolated-E-Field.csv"
    initCSV(interpolated_e_field_csv, "Interpolated Electric Field measured in atomic units.")
    
    # Append the interpolated data rows to the CSV file
    with open(interpolated_e_field_csv, 'a', newline='') as csvfile:
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
    
    # Run the simulation for each time step
    for index, current_time in enumerate(interpolated_times):
        bohr_output = run(
            time_step,
            interpolated_fields[index],
            propagator_method,
            molecular_coordinates,
            wavefunction,
            initial_dipole_moment
        )
        logging.debug(f"At {current_time} fs, the Bohr output is {bohr_output} in AU")
        updateCSV(polarizability_csv, current_time, *bohr_output)
    
    # Plot the results using the interpolated electric field data
    show_eField_pField(interpolated_e_field_csv, polarizability_csv)
    logging.info("Simulation completed successfully.")