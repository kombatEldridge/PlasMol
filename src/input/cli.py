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
    parser.add_argument('-r', '--restart', dest='restart', action='store_true', help="Before simulation starts, removes old files: eField_path, pField_path, pField_Transform_path, chkfile_path, eField_vs_pField_path, and eV_spectrum_path.")

    args = parser.parse_args()
    if args.log and args.verbose == 0:
        args.verbose = 1

    return args
