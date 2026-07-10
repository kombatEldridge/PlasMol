# quantum/propagation.py
import os 
import logging 
import numpy as np 

from plasmol.utils.csv import init_csv, update_csv
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
    has_dch = params.pop('has_dch', False)
    propagator(**params, molecule=molecule, exc=exc)
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
        filepath = params.dch_mo_occ_filepath
        if molecule.is_open_shell:
            n_alpha = molecule._get_mo_occupations(molecule.D_ao[0])
            n_beta  = molecule._get_mo_occupations(molecule.D_ao[1])
            n_k_total = n_alpha + n_beta
        else:
            n_k_total = molecule._get_mo_occupations(molecule.D_ao)
        
        if not os.path.exists(filepath):
            header = ['Timestamps (au)']
            for inx in params.dch_watch_indices:
                header.append(f'MO index {inx}')
            init_csv(filepath, 
                    f"Time dependent MO occupations for the following MO indices: {params.dch_watch_indices}", 
                    header=header)

        update_csv(filepath, params.current_time, *n_k_total)

    return mu_arr


