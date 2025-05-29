# cli.py
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

    # required inputs
    parser.add_argument('-b', '--bohr', type=str, required=True, help="Path to the Bohr input file.")
    
    # general controls
    parser.add_argument('-m', '--mult', type=float, default=1, help="Multiplier for electric field interpolator resolution.")
    parser.add_argument('-l', '--log', help="Log file name.")
    parser.add_argument('-v', '--verbose', action='count', default=0, help="Increase verbosity (use up to -vv).")
    
    # numeric tolerances & iterations
    parser.add_argument('--pcconv', type=float, default=1e-12, help="Convergence for Predictor‑Corrector (Magnus).")
    parser.add_argument('--tol_zero', type=float, default=1e-12, help="Tolerance for Hermitian/unitary checks.")
    parser.add_argument('--max_iter', type=int, default=200, help="Max SCF iterations before failure.")
   
    # time settings
    parser.add_argument('--dt', type=float, default=0.01, help="Time step (unit via -u).")
    parser.add_argument('-t', '--t_end', type=float, default=50, help="Simulation duration (unit via -u).")
    parser.add_argument('-d', '--dir', type=str, default='z', help="Field direction: x, y or z.")
    parser.add_argument('-u', '--time_units', type=str, default="au", help="Time unit: 'au' or 'fs'.")

    # checkpointing
    parser.add_argument('-r', '--restart', dest='restart', action='store_true', help="Removes eField.csv, pField.csv, pField_spectrum.csv, chkfile.npz, output.png, and spectrum.png.")
    parser.add_argument('--nochkfile', dest='chkfile', action='store_false', help="Do not save checkpoints.")
    parser.add_argument('--chkfile_freq', type=int, default=100, help="Steps between checkpoint saves.")
    parser.add_argument('--chkfile_path', type=str, default="chkfile.npz", help="Custom path for checkpoint file.")

    # electric field shape
    parser.add_argument('--peak_time_au', type=float, default=1.0, help="Field peak time (au).")
    parser.add_argument('-w', '--width_steps', type=int, default=5, help="Field width in time steps.")
    parser.add_argument('-s', '--shape', type=str, default="kick", help="Field shape: 'kick' or 'pulse'.")
    
    parser.set_defaults(smoothing=True)
    parser.add_argument('--smoothing', action='store_true', help="Apply smoothing ramp.")
    parser.add_argument('--nosmoothing', dest='smoothing', action='store_false', help="Disable smoothing ramp.")

    # output filenames
    parser.add_argument('--intensity_au', type=float, default=5e-5, help="Electric field intensity (au).")
    parser.add_argument('--eFieldFile', type=str, default="eField.csv", help="Filename for electric field CSV.")
    parser.add_argument('--pFieldFile', type=str, default="pField.csv", help="Filename for polarization CSV.")

    args = parser.parse_args()
    if args.log and args.verbose == 0:
        args.verbose = 1

    return args
