# cli.py
import argparse

def parse_arguments():
    """
    Parses command-line arguments for the simulation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
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