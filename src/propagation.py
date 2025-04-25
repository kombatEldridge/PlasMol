# propagation.py
import logging 
import numpy as np 

from csv_utils import updateCSV

logger = logging.getLogger("main")

def propagation(params, molecule, field, polarizability_csv):
    """
    Note here that for Magnus2, E(t+dt) will be the most recent e field from MEEP and 
    E(t) will be the previous electric field
    """
    # Determine which propagation method to be used
    method = molecule.propagator.lower()
    if method == "exponential":
        from exponential import propagate_density
    elif method == "midpoint":
        from midpoint import propagate_density
    elif method == "tr-midpoint":
        from tr_midpoint import propagate_density
    elif method == "magnus2":
        from magnus2 import propagate_density
    elif method == "magnus4":
        from magnus4 import propagate_density
    else:
        raise ValueError("Please provide in the molecule input file one of the acceptable Density matrix propagators: " \
        "\nexponential, midpoint, tr-midpoint, magnus2, or magnus4.")

    # Main loop
    for index, current_time in enumerate(field.times):
        mu_arr = np.zeros(3)
        D_mo_t_plus_dt = propagate_density(params, molecule, field.field[current_time + 1])
        D_ao_t_plus_dt = molecule.transform_D_mo_to_D_ao(D_mo_t_plus_dt, params)

        for i in [0, 1, 2]:
            mu_arr[i] = 2 * float((np.trace(molecule.wfn.mu[i] @ D_ao_t_plus_dt) - np.trace(molecule.wfn.mu[i] @ molecule.wfn.D[0])).real)

        logging.debug(f"At {current_time} fs, combined Bohr output is {mu_arr} in au")
        updateCSV(polarizability_csv, current_time, *mu_arr)

