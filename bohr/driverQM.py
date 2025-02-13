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
    def __init__(self, Etime, Ex_values, Ey_values, Ez_values):
        self.interp_ex = interp1d(Etime, Ex_values, kind='cubic', fill_value="extrapolate")
        self.interp_ey = interp1d(Etime, Ey_values, kind='cubic', fill_value="extrapolate")
        self.interp_ez = interp1d(Etime, Ez_values, kind='cubic', fill_value="extrapolate")

    def get_field_at(self, t_query):
        return np.column_stack((self.interp_ex(t_query), self.interp_ey(t_query), self.interp_ez(t_query)))


def read_electric_field_csv(file_path):
    """Reads electric field values from a CSV file."""
    time, x_values, y_values, z_values = [], [], [], []
    
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0].startswith("Timestamps"):
                break
        for row in reader:
            if len(row) < 4:
                continue
            time.append(float(row[0]))
            x_values.append(float(row[1]))
            y_values.append(float(row[2]))
            z_values.append(float(row[3]))
    
    return time, x_values, y_values, z_values


def process_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Meep simulation with Bohr dipole moment calculation.")
    parser.add_argument('-b', '--bohr', type=str, help="Path to the Bohr input file.")
    parser.add_argument('-e', '--csv', type=str, help="Path to the E field file.")
    parser.add_argument('-m', '--mult', type=int, help="Multiplier for E Field interpolator.")
    parser.add_argument('-l', '--log', help="Log file name")
    parser.add_argument('-v', '--verbose', action='count', default=0, help="Increase verbosity")
    
    args = parser.parse_args()
    if args.log and args.verbose == 0:
        args.verbose = 1
    
    return args


if __name__ == "__main__":
    log_format = '%(levelname)s: %(message)s'
    args = process_arguments()
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if args.verbose >= 2 else logging.INFO if args.verbose == 1 else logging.WARNING)
    
    handler = logging.FileHandler(args.log, mode='w') if args.log else logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)
    
    mp.verbosity(min(3, args.verbose))
    sys.stdout = PrintLogger(logger, logging.INFO)
    logging.getLogger("h5py").setLevel(logging.INFO)
    
    mh.clear_Matrix_Files()

    moleculeObject = mol.MOLECULE(args.bohr)
    method, coords, wfn = moleculeObject.method["propagator"], moleculeObject.molecule["coords"], moleculeObject.wfn
    D_mo_0 = mh.get_D_mo_0()
    
    Etime, Ex_values, Ey_values, Ez_values = read_electric_field_csv(args.csv)
    ef_interp = ElectricFieldInterpolator(Etime, Ex_values, Ey_values, Ez_values)
    t_new = np.linspace(0, Etime[-1], len(Etime) * args.mult)
    dt = t_new[1] - t_new[0] 
    eArrArr = ef_interp.get_field_at(t_new)
    
    pFieldFileName = "magnus-P-Field.csv"
    initCSV(pFieldFileName, "Molecule's Polarizability Field measured in atomic units.")
    
    with open(os.path.abspath(args.bohr), 'r') as file:
        for line in file:
            if not line.strip().startswith(('#', '--', '%')):
                logger.info('\t%s', line.rstrip('\n'))
    
    for i in range(len(t_new)):
    # for i in range(100):
        bohrResponse = run(dt, eArrArr[i], method, coords, wfn, D_mo_0)
        logging.debug(f"At {t_new[i]} fs, the Bohr output is {bohrResponse} in AU")
        updateCSV(pFieldFileName, t_new[i], *bohrResponse)
    
    show_eField_pField(args.csv, pFieldFileName)
    logging.info("Simulation completed successfully.")
