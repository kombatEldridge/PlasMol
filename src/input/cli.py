# input/cli.py
import argparse

def parse_arguments():
    """
    Parse command-line arguments for the RT-TDDFT simulation.

    This function sets up an argument parser with options for simulation parameters,
    file paths, and verbosity levels, returning the parsed arguments.

    Parameters:
    None

    Returns:
    argparse.Namespace
        An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Meep simulation with Bohr dipole moment calculation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-f', '--input', required=True, type=str, help="Path to the PlasMol input file.")
    parser.add_argument('-l', '--log', help="Log file name.")
    parser.add_argument('-v', '--verbose', action='count', default=1, help="Increase verbosity (use up to -vv).")
    parser.add_argument('-r', '--restart', dest='restart', action='store_true', help="Removes eField.csv, pField.csv, pField_spectrum.csv, pField-transformed.csv, chkfile.npz, output.png, and spectrum.png.")

    args = parser.parse_args()
    if args.log and args.verbose == 0:
        args.verbose = 1

    return args
