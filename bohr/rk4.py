import numpy as np
from fock_builder import build_fock

def propagate_density_matrix(dt, D_mo, wfn, dir, exc):
    """
    Propagate the MO density matrix using RK4.

    Args:
        dt: Time step.
        D_mo: Initial density matrix.
        wfn: Wavefunction object containing information about molecule.
        dir: Cartesian direction x, y, or z but in integer form (0, 1, 2)
        exc: External electric field measured in the dir direction.

    Returns:
        Updated density matrix.
    """

    zidt = -1j * dt # Complex time step scalar
    D_ao = wfn.C[0] @ D_mo @ wfn.C[0].T
    
    # Compute F(t) in AO basis and convert to MO basis
    F_ao = build_fock(wfn, D_ao, dir, exc)
    F_mo = wfn.C[0].T @ F_ao @ wfn.C[0]
    
    # k1
    FP = F_mo @ D_mo
    PFH = D_mo @ np.conjugate(F_mo.T)
    k1 = zidt * (FP - PFH)
    
    # k2
    P_k1 = D_mo + 0.5 * k1
    FP = F_mo @ P_k1
    PFH = P_k1 @ np.conjugate(F_mo.T)
    k2 = zidt * (FP - PFH)
    
    # k3
    P_k2 = D_mo + 0.5 * k2
    FP = F_mo @ P_k2
    PFH = P_k2 @ np.conjugate(F_mo.T)
    k3 = zidt * (FP - PFH)
    
    # k4
    P_k3 = D_mo + k3
    FP = F_mo @ P_k3
    PFH = P_k3 @ np.conjugate(F_mo.T)
    k4 = zidt * (FP - PFH)
    
    # Update density matrix
    D_mo = D_mo + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return D_mo
