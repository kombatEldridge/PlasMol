import numpy as np
import matrix_handler as mh
from scipy.linalg import expm
import logging

# Here, when appended to a vars name
    # t means time (two half steps behind dt)
    # t_plus_half_dt refers to t + dt/2 (one half step behind dt)
    # t_plus_dt refers to t + dt
    # dt is the time step between frames in MEEP for this Magnus implementation

int_to_char = {0: 'x', 1: 'y', 2: 'z'}

def JK(wfn, D_ao):
    pot = wfn.jk.get_veff(wfn.ints_factory, 2*D_ao)
    Fa = wfn.T + wfn.Vne + pot
    return Fa


def build_fock(wfn, D_ao, exc, dir):
    # Repisky2015.pdf Eq. 20
    ext = wfn.mu[dir] * exc[dir]
    F_ao = JK(wfn, D_ao) - ext
    return F_ao


def extrapolate(F_mo_t, F_mo_t_minus_half_dt):
    # Repisky2015.pdf Eq. 18
    F_mo_t_plus_dt2 = 2 * F_mo_t - F_mo_t_minus_half_dt
    return F_mo_t_plus_dt2


def interpolate(F_mo_t, F_mo_t_plus_dt):
    # Repisky2015.pdf Eq. 19
    F_mo_t_plus_dt2 = 0.5 * F_mo_t + 0.5 * F_mo_t_plus_dt
    return F_mo_t_plus_dt2


def euclidean_norm_difference(matrix1, matrix2):
    difference = matrix1 - matrix2
    norm_difference = np.linalg.norm(difference, 'fro')
    
    return norm_difference


def construct_U_t_plus_dt(F_mo_t_plus_half_dt, dt, U_t):
    # Repisky2015.pdf Eq. 16
    if not np.allclose(F_mo_t_plus_half_dt, np.conjugate(F_mo_t_plus_half_dt.T), atol=1e-8):
        raise ValueError("F_mo_t_plus_half_dt is not Hermitian.")
    
    U_new = expm(-1j * F_mo_t_plus_half_dt * dt)
    U_t_plus_dt = U_new if U_t is None else U_new @ U_t

    # Check to see if matrix is Unitary
    unitary = np.conjugate(U_t_plus_dt.T) @ U_t_plus_dt
    if not (U_t_plus_dt.shape[0] == U_t_plus_dt.shape[1] and np.allclose(unitary, np.eye(U_t_plus_dt.shape[0]), atol=1e-10)):
        raise ValueError(f"U^+ @ U is not a unitary matrix. Instead: {unitary}.")

    return U_t_plus_dt


def propagate_density_matrix(dt, wfn, exc, D_mo_0, dir):
    F_mo_t_minus_half_dt = mh.get_F_mo_t_minus_half_dt(dir)
    F_mo_t = mh.get_F_mo_t(dir)
    F_mo_t_plus_half_dt = extrapolate(F_mo_t, F_mo_t_minus_half_dt)
    U_t = mh.get_U_t(dir)

    D_mo_t_plus_dt_guess = None
    for limit in range(0, 10000):
        U_t_plus_dt = construct_U_t_plus_dt(F_mo_t_plus_half_dt, dt, U_t)
        D_mo_t_plus_dt = U_t_plus_dt @ D_mo_0 @ np.conjugate(U_t_plus_dt.T)
        D_ao_t_plus_dt = wfn.C[0] @ D_mo_t_plus_dt @ wfn.C[0].T
        F_ao_t_plus_dt = build_fock(wfn, D_ao_t_plus_dt, exc, dir)
        F_mo_t_plus_dt = wfn.C[0].T @ F_ao_t_plus_dt @ wfn.C[0]

        # Predictor-Corrector finish condition
        if D_mo_t_plus_dt_guess is not None:
            if euclidean_norm_difference(D_mo_t_plus_dt, D_mo_t_plus_dt_guess) < 1e-12:
                logging.debug(f"Predictor-Corrector scheme finished for the {int_to_char[dir]} direction in {limit} iterations.")
                mh.set_F_mo_t_minus_half_dt(F_mo_t_plus_half_dt, dir)
                mh.set_F_mo_t(F_mo_t_plus_dt, dir)
                mh.set_U_t(U_t_plus_dt, dir)
                return D_mo_t_plus_dt
            
        F_mo_t_plus_half_dt = interpolate(F_mo_t, F_mo_t_plus_dt)
        D_mo_t_plus_dt_guess = D_mo_t_plus_dt
