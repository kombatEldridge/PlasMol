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
    parser.add_argument('-v', '--verbose', action='count', default=1, help="Increase verbosity (use up to -vv).")
       
    # checkpointing
    parser.add_argument('-r', '--restart', dest='restart', action='store_true', help="Removes eField.csv, pField.csv, pField_spectrum.csv, chkfile.npz, output.png, and spectrum.png.")
    parser.add_argument('--nochkfile', dest='chkfile', action='store_false', help="Do not save checkpoints.")
    parser.add_argument('--chkfile_freq', type=int, help="Steps between checkpoint saves.")
    parser.add_argument('--chkfile_path', type=str, default='chkfile.npz', help="Custom path for checkpoint file.")

    # output filenames
    parser.add_argument('--eField_path', type=str, help="Filename to store the electric field felt by the molecule.")
    parser.add_argument('--pField_path', type=str, help="Filename for polarization produced by the molecule.")
    parser.add_argument('--pField_Transform_path', type=str, default='pField_Transform.csv', help="Filename for Fourier transformed polarization CSV.")
    parser.add_argument('--eField_vs_pField_path', type=str, default='output.png', help="Filename for spectrum comparing electric field and polarization field.")
    parser.add_argument('--eV_spectrum_path', type=str, default='spectrum.png', help="Filename for Fourier transformed spectrum in eV.")

    args = parser.parse_args()
    if args.log and args.verbose == 0:
        args.verbose = 1

    return args
