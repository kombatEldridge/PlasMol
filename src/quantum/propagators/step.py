# step.py
import logging
import numpy as np
from scipy.linalg import expm

logger = logging.getLogger("main")

def propagate(params, molecule, exc):
    """
    Propagate molecular orbitals using the step method.
    
    C'(t+dt) = U(t)C'(t-dt)
    U(t) = exp(-i*2dt*F')

    Parameters:
    params : object
        Parameters object with dt attribute.
    molecule : object
        Molecule object with current state data.
    exc : np.ndarray
        External electric field at the current time step.

    Returns:
    None
    """
    if hasattr(molecule, 'C_orth_ndt'):
        C_orth_ndt = molecule.C_orth_ndt
    else:
        C_orth_ndt = molecule.rotate_coeff_to_orth(molecule.mf.mo_coeff)

    F_orth = molecule.F_orth

    U = expm(-1j * 2 * params.dt * F_orth)
    
    C_orth_pdt = np.matmul(U, C_orth_ndt)
    C_pdt = molecule.rotate_coeff_away_from_orth(C_orth_pdt)
    D_ao_pdt = molecule.mf.make_rdm1(mo_coeff=C_pdt, mo_occ=molecule.occ)
    F_orth_pdt = molecule.get_F_orth(D_ao_pdt, exc)
    
    molecule.mf.mo_coeff = C_pdt
    molecule.C_orth_ndt = molecule.rotate_coeff_to_orth(molecule.mf.mo_coeff)
    molecule.D_ao = D_ao_pdt
    molecule.F_orth = F_orth_pdt
