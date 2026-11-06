# quantum/propagation.py
import logging 
from induced_mu import induced_dipole

logger = logging.getLogger("main")

def propagation(params, molecule, exc, propagator):
    """
    Perform time propagation of the molecular state.

    Propagates the molecule over time using the specified method, recording polarization.

    The propagator advances the state from current_time to current_time + dt.
    Everything recorded below describes the advanced state, so it is stamped
    with the time the state reached, on the same rounding as the dipole rows.

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
    prop_params = params.copy()
    has_core_hole = prop_params.pop('has_core_hole')
    current_time = prop_params.pop('current_time')
    decimals = getattr(molecule, 'time_rounding_decimals', 12)
    reached_time = round(float(current_time) + float(prop_params['dt']), decimals)

    propagator(**prop_params, molecule=molecule, exc=exc)

    mu_arr = induced_dipole(molecule)

    # ------------------------------------ #
    #              Additional              #
    #      custom tracking functions       #
    #           can be added here          #
    # ------------------------------------ #
    if has_core_hole:
        molecule.get_mo_occupations(reached_time)
    
    return mu_arr


