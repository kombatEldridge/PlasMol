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

    parser.add_argument('-p', '--pmif', type=str, help="Path to the PlasMol Input File (pmif).")
    parser.add_argument('-m', '--mif', type=str, help="Path to the Meep Input File (mif). This should only be included if the user is only running a Meep calculation, and is not running a PlasMol (Meep + TDDFT) calculation.")
    parser.add_argument('-q', '--qif', type=str, help="Path to the TDDFT (Quantum) Input File (qif). This should only be included if the user is only running a TDDFT calculation, and is not running a PlasMol (Meep + TDDFT) calculation.")

    # general controls
    parser.add_argument('-l', '--log', help="Log file name.")
    parser.add_argument('-v', '--verbose', action='count', help="Increase verbosity (use up to -vv).")
    
    # numeric tolerances & iterations
    parser.add_argument('--pcconv', type=float, help="Convergence for Predictor‑Corrector (Magnus).")
    parser.add_argument('--tol_zero', type=float, help="Tolerance for Hermitian/unitary checks.")
    parser.add_argument('--max_iter', type=int,  help="Max SCF iterations before failure.")
   
    # time settings
    parser.add_argument('--dt', type=float, help="Time step (unit via -u).")
    parser.add_argument('-t', '--t_end', type=float, help="Simulation duration (unit via -u).")
    parser.add_argument('-d', '--dir', type=str, help="Field direction: x, y or z.")
    parser.add_argument('-u', '--time_units', type=str, help="Time unit: 'au' or 'fs'.")

    # checkpointing
    parser.add_argument('-r', '--restart', dest='restart', action='store_true', help="Removes eField.csv, pField.csv, pField_spectrum.csv, chkfile.npz, output.png, and spectrum.png.")
    parser.add_argument('--nochkfile', dest='chkfile', action='store_false', help="Do not save checkpoints.")
    parser.add_argument('--chkfile_freq', type=int, help="Steps between checkpoint saves.")
    parser.add_argument('--chkfile_path', type=str, help="Custom path for checkpoint file.")

    # electric field shape
    parser.add_argument('--peak_time_au', type=float, help="Field peak time (au).")
    parser.add_argument('-w', '--width_steps', type=int, help="Field width in time steps.")
    parser.add_argument('-s', '--shape', type=str, help="Field shape: 'kick' or 'pulse'.")
    
    parser.set_defaults(smoothing=True)
    parser.add_argument('--smoothing', action='store_true', help="Apply smoothing ramp.")
    parser.add_argument('--nosmoothing', dest='smoothing', action='store_false', help="Disable smoothing ramp.")

    # output filenames
    parser.add_argument('--intensity_au', type=float, help="Electric field intensity (au).")
    parser.add_argument('--eFieldFile', type=str, help="Filename for electric field CSV.")
    parser.add_argument('--pFieldFile', type=str, help="Filename for polarization CSV.")

    args = parser.parse_args()
    if args.log and args.verbose == 0:
        args.verbose = 1

    # Convert time in fs to au
    # T_AU_FS = 41.3413733  # Time units in au/fs
    # if (args.time_units == 'fs'):
    #     args.dt = args.dt * T_AU_FS
    #     args.t_end = args.t_end * T_AU_FS
    # elif (args.time_units == 'au'):
    #     pass
    # else: 
    #     raise ValueError(f"The timestep unit for this simulation should only either be 'fs' or 'au'.")

    # Time step check
    # if (args.dt > 0.1):
    #     raise ValueError(f"The timestep for this simulation is too large to elicit physical results.")

    return args

def minimum_sufficiency(args):

    return True