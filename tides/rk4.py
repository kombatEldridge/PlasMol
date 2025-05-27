# rk4.py
import logging
import numpy as np
from scipy.linalg import expm

logger = logging.getLogger("main")

def propagate(params, molecule, exc):
    '''
    Needed to resume: 
        molecule.scf.mo_coeff
        molecule.F_orth
    
    C'(t + dt) = C'(t) + (k1/6 + k2/3 + k3/3 + k4/6)
    dC' = -i * dt * (F'C')
    '''
    C_orth = molecule.rotate_coeff_to_orth(molecule.scf.mo_coeff)
    F_orth = molecule.F_orth

    # k1
    k1 = -1j * params.dt * (np.matmul(F_orth, C_orth))
    C_orth_1 = C_orth + 1/2 * k1

    # k2
    k2 = -1j * params.dt * (np.matmul(F_orth, C_orth_1))
    C_orth_2 = C_orth + 1/2 * k2

    # k3
    k3 = -1j * params.dt * (np.matmul(F_orth, C_orth_2))
    C_orth_3 = C_orth + k3

    # k4
    k4 = -1j * params.dt * (np.matmul(F_orth, C_orth_3))

    C_orth_pdt = C_orth + (k1/6 + k2/3 + k3/3 + k4/6)
    C_pdt = molecule.rotate_coeff_away_from_orth(C_orth_pdt)
    D_ao_pdt = molecule.scf.make_rdm1(mo_coeff=C_pdt, mo_occ=molecule.occ)
    F_orth_pdt = molecule.get_F_orth(D_ao_pdt, exc)

    molecule.scf.mo_coeff = C_pdt
    molecule.D_ao = D_ao_pdt
    molecule.F_orth = F_orth_pdt
