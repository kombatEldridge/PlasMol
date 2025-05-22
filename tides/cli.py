# cli.py
import argparse

def parse_arguments():
    """
    Parses command-line arguments for the simulation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Meep simulation with Bohr dipole moment calculation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-b', '--bohr', type=str, required=True, help="Path to the Bohr input file.")
    parser.add_argument('-e', '--csv', default="", type=str, help="(Optional) Path to the electric field CSV file.")
    parser.add_argument('-m', '--mult', type=float, default=1, help="(Optional) Multiplier for the electric field interpolator resolution.")
    parser.add_argument('-l', '--log', help="(Optional) Log file name")
    parser.add_argument('-v', '--verbose', action='count', default=0, help="(Optional) Increase verbosity.")
    parser.add_argument('-pc', '--pcconv', type=float, default=1e-12, help="(Optional) Iteration convergence for Predictor-Corrector scheme in Magnus propagator.")
    parser.add_argument('-tol', '--tol_zero', type=float, default=1e-12, help="(Optional) Tolerance for numerical checks (e.g., Hermitian, unitary).")
    parser.add_argument('-mi', '--max_iter', type=float, default=200, help="(Optional) Maximum iterations for SCF before failure")
    parser.add_argument('-dc', '--doublecheck', type=bool, default=True, help="(Optional) Check for Fock matrix is Hermitian and U matrix is Unitary.")
    parser.add_argument('-exp', '--exp_method', type=int, default=2, help="(Optional) Method for matrix exponentiation (1 for series, 2 for diagonalization).")
    parser.add_argument('-tint', '--terms_interpol', type=int, default=2, help="(Optional) Number of identical iterations for convergence.")
    parser.add_argument('-dt', type=float, default=0.01, help="(Optional) The time step used in the simulation. Units defined with -u.")
    parser.add_argument('-t', '--t_end', type=float, default=50, help="(Optional) Duration of simulation. Units defined with -u.")
    parser.add_argument('-d', '--dir', type=str, default='z', help="(Optional) Direction string (x, y, or z) for the excited electric field.")
    parser.add_argument('-u', '--time_units', type=str, default="au", help="(Optional) The unit of time. Currently only support 'fs' or 'au'")
    parser.add_argument('-chk', '--chkfile', type=bool, default=True, help="(Optional) Save a checkpoint file incase job quits unexpectedly.")
    parser.add_argument('-chkf', '--chkfile_freq', type=int, default=10, help="(Optional) Amount of time steps between saved checkpoints.")
    parser.add_argument('-chkp', '--chkfile_path', type=str, default="chkfile.txt", help="(Optional) Custom path to checkpoint file.")

    args = parser.parse_args()
    if args.log and args.verbose == 0:
        args.verbose = 1

    return args

