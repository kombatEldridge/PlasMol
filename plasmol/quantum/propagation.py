# quantum/propagation.py
import os 
import logging 
import numpy as np 

logger = logging.getLogger("main")

def propagation(params, molecule, exc, propagator):
    """
    Perform time propagation of the molecular state.

    Propagates the molecule over time using the specified method, recording polarization.

    Parameters:
    params : object
        Parameters object with simulation settings.
    molecule : object
        Molecule object with current state.
    exc : array-like
        Electric field at the current time step.
    propagator : function
        The propagation function to use.

    Returns:
    None
    """
    mu_arr = np.zeros(3)
    
    prop_params = params.copy()
    has_dch = prop_params.pop('has_dch')
    current_time = prop_params.pop('current_time')

    propagator(**prop_params, molecule=molecule, exc=exc)
    mu = molecule.calculate_mu()

    D = molecule.D_ao
    D0 = molecule.D_ao_0
    if D.ndim == 3:
        D = D.sum(axis=0)
        D0 = D0.sum(axis=0)
    
    for i in [0, 1, 2]:
        mu_arr[i] = float((np.trace(mu[i] @ D) - np.trace(mu[i] @ D0)).real)

    # ------------------------------------ #
    #              Additional              #
    #      custom tracking functions       #
    #           can be added here          #
    #  similar to molecule.calculate_mu()  #
    # ------------------------------------ #
    if has_dch:
        molecule.get_mo_occupations(current_time)

    return mu_arr


