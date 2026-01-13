# quantum/propagation.py
import logging 
import numpy as np 

logger = logging.getLogger("main")

def propagation(params, molecule, field, propagate):
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
    mu_arr = np.zeros(3)
    propagate(params, molecule, field)
    mu = molecule.calculate_mu()
    for i in [0, 1, 2]:
        mu_arr[i] = float((np.trace(mu[i] @ molecule.D_ao) - np.trace(mu[i] @ molecule.D_ao_0)).real)

    # ------------------------------------ #
    #              Additional              #
    #      custom tracking functions       #
    #           can be added here          #
    #  similar to molecule.calculate_mu()  #
    # ------------------------------------ #

    return mu_arr
