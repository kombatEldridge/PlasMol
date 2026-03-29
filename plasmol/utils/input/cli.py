# utils/input/cli.py
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

    parser.add_argument('input', nargs='?', default=None, help="Path to the PlasMol input file (required unless --checkpoint is used).")
    parser.add_argument('-l', '--log', help="Log file name.")
    parser.add_argument('-v', '--verbose', action='count', default=1, help="Increase verbosity (use up to -vv).")
    parser.add_argument('-c', '--checkpoint', help="Path to checkpoint file to resume simulation from. When provided, the input file is NOT required.")

    args = parser.parse_args()
    if args.input is None and args.checkpoint is None:
        parser.error("Either an input file or --checkpoint must be provided.")
    if args.log and args.verbose == 0:
        args.verbose = 1

    return args
