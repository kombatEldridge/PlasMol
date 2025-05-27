# chkfile.py
import numpy as np
import logging

logger = logging.getLogger("main")

def update_chkfile(molecule, current_time):
    """
    Save out a .npz checkpoint containing:
      – current_time
      – molecule.D_ao_0
      – molecule.scf.mo_coeff
      – molecule.F_orth
      – plus one extra array depending on molecule.propagator
    """
    # ensure .npz extension
    fn = molecule.chkfile
    if not fn.endswith(".npz"):
        fn = fn + ".npz"

    method = molecule.propagator.lower()
    save_dict = {
        "current_time": current_time,
        "D_ao_0":       molecule.D_ao_0,
        "mo_coeff":     molecule.scf.mo_coeff,
        "F_orth":       molecule.F_orth,
    }

    if method == "step":
        save_dict["C_orth_ndt"] = molecule.C_orth_ndt
    elif method == "magnus2":
        save_dict["F_orth_n12dt"] = molecule.F_orth_n12dt

    np.savez(fn, **save_dict)
    logger.debug(f"Wrote checkpoint to {fn} with keys: {list(save_dict)}")


def restart_from_chkfile(molecule):
    """
    Load from the .npz checkpoint created by update_chkfile.
    """
    fn = molecule.chkfile
    if not fn.endswith(".npz"):
        fn = fn + ".npz"

    logger.debug(f"Loading checkpoint from {fn}")
    data = np.load(fn, allow_pickle=True)

    # common data
    molecule.current_time = float(data["current_time"])
    molecule.D_ao_0       = data["D_ao_0"]
    molecule.scf.mo_coeff = data["mo_coeff"]
    molecule.F_orth       = data["F_orth"]

    # conditional
    method = molecule.propagator.lower()
    if method == "step" and "C_orth_ndt" in data:
        molecule.C_orth_ndt = data["C_orth_ndt"]
    elif method == "magnus2" and "F_orth_n12dt" in data:
        molecule.F_orth_n12dt = data["F_orth_n12dt"]

    logger.debug(
        f"Restarted at t={molecule.current_time} au; "
        f"loaded propagator='{method}', "
        f"arrays: {list(data.keys())}"
    )