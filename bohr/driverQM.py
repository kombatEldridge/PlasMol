import molecule as mol
from bohr import run
import csv
import numpy as np
import sys
import argparse
import logging
import sys
import os
import meep as mp
import matrix_handler as mh
from csv_handler import initCSV, updateCSV, show_eField_pField


class PrintLogger(object):
    """Intercepts print statements and redirects them to a logger."""
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, message):
        message = message.rstrip()
        if message:
            self.logger.log(self.level, message)

    def flush(self):
        pass


def read_electric_field_csv(file_path):
    """
    Returns the values found in the E filed csv files (which are in AU)
    """
    time = []
    x_values = []
    y_values = []
    z_values = []
    
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        
        # Skip header lines until the actual data starts
        for row in reader:
            if row and row[0].startswith("Timestamps"):
                break
        
        # Read the data rows
        for row in reader:
            if len(row) < 4:
                continue  # Skip incomplete rows
            
            time.append(float(row[0]))
            x_values.append(float(row[1]))
            y_values.append(float(row[2]))
            z_values.append(float(row[3]))
    
    return time, x_values, y_values, z_values


def processArguments():
    """
    Parses command line arguments for the Bohr simulation script.

    Command line arguments:
    - `-b` or `--bohr`: Path to the Bohr input file.
    - `-e` or `--csv`: Path to the E field file.
    - `-l` or `--log`: Log file name.
    - `-v` or `--verbose`: Increase verbosity of logging.

    Returns:
        argparse.Namespace: Parsed arguments.

    Exits:
        Exits the program with status code 1 if required arguments are not provided.
    """
    logging.debug("Processing command line arguments.")
    parser = argparse.ArgumentParser(description="Meep simulation with Bohr dipole moment calculation.")
    parser.add_argument('-b', '--bohr', '--molecule', type=str, help="Path to the Bohr input file.")
    parser.add_argument('-e', '--csv', type=str, help="Path to the E field file.")
    parser.add_argument('-l', '--log', help="Log file name")
    parser.add_argument('-v', '--verbose', action='count', default=0, help="Increase verbosity")

    args = parser.parse_args()

    if args.log and args.verbose == 0:
        args.verbose = 1
    
    logging.info(f"Electric field file: {os.path.abspath(args.csv)}")

    return args


if __name__ == "__main__":
    log_format = '%(levelname)s: %(message)s'
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bohr', '--molecule', type=str, help="Path to the Bohr input file.")
    parser.add_argument('-e', '--csv', type=str, help="Path to the E field file.")
    parser.add_argument('-l', '--log', help="Log file name")
    parser.add_argument('-v', '--verbose', action='count', default=0, help="Increase verbosity")
    temp_args = parser.parse_known_args()[0]

    if temp_args.log and temp_args.verbose == 0:
        temp_args.verbose = 1

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if temp_args.log:
        file_handler = logging.FileHandler(temp_args.log, mode='w')
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)
    
    if temp_args.verbose >= 2:
        logger.setLevel(logging.DEBUG)
        mp.verbosity(3)
    elif temp_args.verbose == 1:
        logger.setLevel(logging.INFO)
        mp.verbosity(2)
    else:
        logger.setLevel(logging.WARNING)
        mp.verbosity(0)

    sys.stdout = PrintLogger(logger, logging.INFO)
    logging.getLogger("h5py").setLevel(logging.INFO)

    mh.clear_Matrix_Files()

    args = processArguments()
    
    courant = 0.5
    resolution = 1000
    timeStepMeep = courant / resolution
    convertTimeMeep2fs =  10 / 3
    convertTimeAtomic2fs = 0.024188843
    convertTimeMeep2Atomic = convertTimeMeep2fs / convertTimeAtomic2fs

    dt = 2 * timeStepMeep * convertTimeMeep2Atomic

    moleculeObject = mol.MOLECULE(args.bohr)
    method = moleculeObject.method["propagator"]
    coords = moleculeObject.molecule["coords"]
    wfn = moleculeObject.wfn
    D_mo_0 = mh.get_D_mo_0()

    eFieldFileName = args.csv
    Etime, Ex_values, Ey_values, Ez_values = read_electric_field_csv(eFieldFileName)
    eArrArr = np.array([Ex_values, Ey_values, Ez_values])  # Convert list of lists to a NumPy array

    pFieldFileName = "magnus-P-Field.csv"
    p_field_comment = f"Molecule's Polarizability Field measured\nat the molecule's position in atomic units."
    initCSV(pFieldFileName, p_field_comment)

    logging.info(f"Bohr input file: {os.path.abspath(args.bohr)}")
    with open(os.path.abspath(args.bohr), 'r') as file:
        for line in file:
            if line.strip().startswith('#') or line.strip().startswith('--') or line.strip().startswith('%'):
                continue
            logger.info('\t%s', line.rstrip('\n'))
    logger.info("")

    for i in range(len(Etime)):
        eArr = eArrArr[:,i]
        bohrResponse = run(dt, eArr, method, coords, wfn, D_mo_0)
        logging.debug(f"At {Etime[i]} fs, the Bohr output is {bohrResponse} in AU")
        updateCSV(pFieldFileName, Etime[i], bohrResponse[0], bohrResponse[1], bohrResponse[2])

    show_eField_pField(eFieldFileName, pFieldFileName, "output", "")

    logging.info("Input file successfully parsed. Beginning simulation")
