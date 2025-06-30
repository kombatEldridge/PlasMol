# chkfile.py
import numpy as np
import logging
import _pickle

logger = logging.getLogger("main")

def update_chkfile(params, molecule, current_time):
    """
    Save a checkpoint file containing the current state of the simulation.

    This function creates a .npz file with simulation data including the current time,
    initial density matrix, molecular orbital coefficients, Fock matrix in orthogonal basis,
    and an additional array based on the propagator method ('step' or 'magnus2').
    The file is saved with the name specified in molecule.chkfile_path, appending .npz if not present.

    Parameters:
    molecule : object
        The molecule object containing simulation data.
    current_time : float
        The current time in the simulation in atomic units.

    Returns:
    None
    """
    # ensure .npz extension
    if not params.chkfile_path.endswith(".npz"):
        params.chkfile_path = params.chkfile_path + ".npz"
    fn = params.chkfile_path

    method = params.propagator.lower()
    save_dict = {
        "current_time": current_time,
        "D_ao_0":       molecule.D_ao_0,
        "mo_coeff":     molecule.mf.mo_coeff,
    }

    if method == "step":
        save_dict["C_orth_ndt"] = molecule.C_orth_ndt
    elif method == "magnus2":
        save_dict["F_orth_n12dt"] = molecule.F_orth_n12dt

    try:
        np.savez(fn, **save_dict)
    except IOError as e:
        logger.error(f"Failed to write checkpoint file {fn}: {e}")
        raise

    logger.debug(f"Wrote checkpoint to {fn} with keys: {list(save_dict)}")


def restart_from_chkfile(molecule, params):
    """
    Load the simulation state from a checkpoint file.

    This function loads data from a .npz checkpoint file specified in molecule.chkfile_path,
    updating the molecule object with the saved state. It includes common data and
    propagator-specific arrays based on the method ('step' or 'magnus2').

    Parameters:
    molecule : object
        The molecule object to be updated with checkpoint data.

    Returns:
    None
    """
    fn = params.chkfile_path

    try:
        data = np.load(fn, allow_pickle=True)
    except FileNotFoundError:
        logger.error(f"Checkpoint file {fn} not found.")
        raise
    except KeyError as e:
        logger.error(f"Missing key in checkpoint file {fn}: {e}")
        raise
    except (_pickle.UnpicklingError, ValueError):
        logger.error(f"{fn} is not a valid checkpoint archive.")
        raise
    
    logger.debug(f"Loading checkpoint from {fn}")

    # common data
    molecule.current_time = float(data["current_time"])
    molecule.D_ao_0       = data["D_ao_0"]
    molecule.mf.mo_coeff = data["mo_coeff"]

    # conditional
    method = params.propagator.lower()
    if method == "step" and "C_orth_ndt" in data:
        molecule.C_orth_ndt = data["C_orth_ndt"]
    elif method == "magnus2" and "F_orth_n12dt" in data:
        molecule.F_orth_n12dt = data["F_orth_n12dt"]

    logger.debug(
        f"Restarted at t={molecule.current_time} au; "
        f"loaded propagator='{method}', "
        f"arrays: {list(data.keys())}"
    )