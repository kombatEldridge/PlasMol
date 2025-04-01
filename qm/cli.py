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
    parser.add_argument('-e', '--csv', default="", type=str, help="(Optional) Path to the electric field CSV file.")
    parser.add_argument('-m', '--mult', type=float, default=1, help="(Optional) Multiplier for the electric field interpolator resolution.")
    parser.add_argument('-l', '--log', help="(Optional) Log file name")
    parser.add_argument('-v', '--verbose', action='count', default=0, help="(Optional) Increase verbosity.")
    parser.add_argument('-i', '--pcconv', type=float, default=1e-12, help="(Optional) Iteration convergence for Predictor-Corrector scheme in Magnus propagator.")
    parser.add_argument('-dt', type=float, default=1e-12, help="(Optional) The time step used in the simulation in fs.")
    parser.add_argument('-d', '--dir', type=str, default='z', help="(Optional) Direction string (x, y, or z) for the excited electric field.")

    args = parser.parse_args()
    if args.log and args.verbose == 0:
        args.verbose = 1

    return args

