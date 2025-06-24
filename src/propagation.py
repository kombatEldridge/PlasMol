# propagation.py
import logging 
import numpy as np 
from chkfile import update_chkfile
from csv_utils import updateCSV

logger = logging.getLogger("main")

def propagation(params, molecule, field):
    """
    Perform time propagation of the molecular state.

    Propagates the molecule over time using the specified method, recording polarization
    and saving checkpoints as configured.

    Parameters:
    params : object
        Parameters object with simulation settings.
    molecule : object
        Molecule object with current state.
    field : object
        Electric field object with time-dependent field data.

    Returns:
    None
    """
    # Determine which propagation method to be used
    method = params.propagator.lower()
    if method == "step":
        from step import propagate
    elif method == "magnus2":
        from magnus2 import propagate
    elif method == "rk4":
        from rk4 import propagate
    else:
        raise ValueError("Please provide in the molecule input file one of the acceptable Density matrix propagators: " \
        "\nstep, rk4, or magnus2.")

    # for index, current_time in enumerate(field.times):
    #     if current_time < molecule.current_time:
    #         continue

    mu_arr = np.zeros(3)

    propagate(params, molecule, field)
    mu = molecule.calculate_mu()
    for i in [0, 1, 2]:
        mu_arr[i] = float((np.trace(mu[i] @ molecule.D_ao) - np.trace(mu[i] @ molecule.D_ao_0)).real)

    return mu_arr
    # logging.debug(f"At {current_time} au, combined Bohr output is {mu_arr} in au")
    # updateCSV(params.pField_path, current_time, *mu_arr)
    
    # if params.chkfile_path and index % params.chkfile_freq == 0:
    #     update_chkfile(molecule, current_time)
