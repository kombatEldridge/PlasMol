# magnus2.py
import logging
import numpy as np
from scipy.linalg import expm

logger = logging.getLogger("main")

def propagate(params, molecule, exc):
    '''
    C'(t+dt) = U(t+0.5dt)C'(t)
    U(t+0.5dt) = exp(-i*dt*F')

    1. Extrapolate F'(t+0.5dt)
    2. Propagate
    3. Build new F'(t+dt), interpolate new F'(t+0.5dt)
    4. Repeat propagation and interpolation until convergence
    '''
    C_orth = molecule.rotate_coeff_to_orth(molecule.C)
    F_orth_p12dt = 2 * molecule.F_orth - molecule.F_orth_n12dt

    max_iterations = params.max_iter
    iteration = 0
    converged = False
    while not converged:
        iteration += 1
        if iteration > max_iterations:
            raise RuntimeError(f"Failed to converge within {max_iterations} iterations")

        U = expm(-1j * params.dt * F_orth_p12dt)
        C_orth_pdt = np.matmul(U, C_orth)
        C_pdt = molecule.rotate_coeff_away_from_orth(C_orth_pdt)
        D_ao_pdt = molecule.scf.make_rdm1(mo_coeff=C_pdt, mo_occ=molecule.occ)
        F_orth_pdt = molecule.get_F_orth(D_ao_pdt, exc)

        if (iteration > 1 and abs(np.linalg.norm(C_pdt) - np.linalg.norm(C_ao_pdt_old)) < params.pcconv):
            molecule.C = C_pdt
            molecule.D_ao = D_ao_pdt
            molecule.F_orth = F_orth_pdt
            molecule.F_orth_n12dt = F_orth_p12dt
            converged = True

        F_orth_p12dt = 0.5 * (molecule.F_orth + F_orth_pdt)
        C_ao_pdt_old = C_pdt
        molecule.C = C_pdt
        molecule.D_ao = D_ao_pdt
