# propagation.py
import logging 
import numpy as np 
from chkfile import update_chkfile
from csv_utils import updateCSV

logger = logging.getLogger("main")

def propagation(params, molecule, field, polarizability_csv):
    """
    Note here that for Magnus2, E(t+dt) will be the most recent e field from MEEP and 
    E(t) will be the previous electric field
    """
    from magnus2 import propagate
    
    # Determine which propagation method to be used
    # method = molecule.propagator.lower()
    # if method == "exponential": 
    #     from exponential import propagate # TODO: Build exponential method
    # elif method == "midpoint":
    #     from midpoint import propagate # TODO: Build midpoint method
    # elif method == "tr-midpoint":
    #     from tr_midpoint import propagate # TODO: Build tr_midpoint method
    # elif method == "magnus2":
    #     from magnus2 import propagate
    # else:
    #     raise ValueError("Please provide in the molecule input file one of the acceptable Density matrix propagators: " \
    #     "\nexponential, midpoint, tr-midpoint, or magnus2.")

    for index, current_time in enumerate(field.times):
        if current_time < molecule.current_time:
            continue

        mu_arr = np.zeros(3)
        propagate(params, molecule, field.field[index])
        mu = molecule.calculate_mu()
        for i in [0, 1, 2]:
            mu_arr[i] = 2 * float((np.trace(mu[i] @ molecule.D_ao) - np.trace(mu[i] @ molecule.D_ao_0)).real)

        logging.debug(f"At {current_time} au, combined Bohr output is {mu_arr} in au")
        updateCSV(polarizability_csv, current_time, *mu_arr)
        
        if params.chkfile and np.mod(index, params.chkfile_freq) == 0:
            update_chkfile(molecule, current_time)
