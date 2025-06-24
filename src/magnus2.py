# magnus2.py
import logging
import numpy as np
from scipy.linalg import expm

logger = logging.getLogger("main")

def propagate(params, molecule, exc):
    """
    Propagate molecular orbitals using the Magnus2 method.

    Implements a predictor-corrector scheme to propagate the density matrix over one time step,
    using extrapolation and iterative refinement until convergence.

    Parameters:
    params : object
        Parameters object with dt, maxiter, and pcconv attributes.
    molecule : object
        Molecule object with current state data.
    exc : np.ndarray
        External electric field at the current time step.

    Returns:
    None
    """
    C_orth = molecule.rotate_coeff_to_orth(molecule.mf.mo_coeff)
    F_orth_p12dt = 2 * molecule.F_orth - molecule.F_orth_n12dt
    C_ao_pdt_old = None

    iteration = 0
    while True:
        iteration += 1
        if iteration > params.maxiter:
            raise RuntimeError(f"Failed to converge within {params.maxiter} iterations")

        # 1) predictor
        U = expm(-1j * params.dt * F_orth_p12dt)
        C_orth_pdt = np.matmul(U, C_orth)
        C_pdt = molecule.rotate_coeff_away_from_orth(C_orth_pdt)
        
        # 2) compute new Fock
        D_ao_pdt = molecule.mf.make_rdm1(mo_coeff=C_pdt, mo_occ=molecule.occ)
        F_orth_pdt = molecule.get_F_orth(D_ao_pdt, exc)
        
        # 3) only check convergence if we have a previous value
        if C_ao_pdt_old is not None and np.linalg.norm(C_pdt - C_ao_pdt_old) < params.pc_convergence:
            molecule.mf.mo_coeff = C_pdt
            molecule.D_ao = D_ao_pdt
            molecule.F_orth = F_orth_pdt
            molecule.F_orth_n12dt = F_orth_p12dt
            logger.debug(f'Magnus2 converged in {iteration} iterations.')
            break

        # 4) update history for next iteration
        F_orth_p12dt = 0.5 * (molecule.F_orth + F_orth_pdt)
        C_ao_pdt_old = C_pdt
        molecule.mf.mo_coeff = C_pdt
        molecule.D_ao = D_ao_pdt
