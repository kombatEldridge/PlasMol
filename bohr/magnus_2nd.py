import numpy as np
from fock_builder import build_fock
import matrix_handler as mh
import logging 
from scipy.linalg import expm


def extrapolate(F_mo, F_mo_ndt2):
    F_mo_dt2 = 2 * F_mo - F_mo_ndt2
    return F_mo_dt2


def interpolate(F_mo, F_mo_dt):
    F_mo_dt2 = 0.5 * F_mo + 0.5 * F_mo_dt
    return F_mo_dt2


def euclidean_norm_difference(matrix1, matrix2):
    """
    Compute the Euclidean norm (Frobenius norm) difference between two matrices.
    Input order does not matter.

    Parameters:
        matrix1 (ndarray): First matrix.
        matrix2 (ndarray): Second matrix.

    Returns:
        float: Euclidean norm difference.
    """
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same shape")
    
    difference = matrix1 - matrix2
    norm_difference = np.linalg.norm(difference, 'fro')
    
    return norm_difference
 

def construct_D_mo_dt(wfn, dt, exc):
    D_mo_0 = mh.get_D_mo_0()

    trace = np.trace(D_mo_0)
    n = wfn.nel[0]
    if np.isclose(trace, n):
        logging.debug("Previous Density matrix used.")
    else:
        raise ValueError(f"Trace of the matrix is not {n} (instead {trace}).")

    F_mo_ct = mh.get_F_mo_ct()
    F_mo_ndt2 = mh.get_F_mo_ndt2()
    D_mo_dt_prev = None
    F_mo_dt2 = extrapolate(F_mo_ct, F_mo_ndt2)
    U_ct = mh.get_U_ct()

    for limit in range(0, 10000):
        U_dt = construct_U_dt(F_mo_dt2, dt, U_ct)
        D_mo_dt = U_dt @ D_mo_0 @ np.conjugate(U_dt.T)
        D_ao_dt = wfn.C[0] @ D_mo_dt @ wfn.C[0].T
        F_ao_dt = build_fock(wfn, D_ao_dt, exc)
        F_mo_dt = wfn.C[0].T @ F_ao_dt @ wfn.C[0]

        if D_mo_dt_prev is not None:
            if euclidean_norm_difference(D_mo_dt, D_mo_dt_prev) < 1e-10:
                logging.debug(f"Iterations before Predictor-Corrector scheme finished: {limit}")
                return U_dt, F_mo_dt, F_mo_dt2, D_mo_dt
            
        F_mo_dt2 = interpolate(F_mo_ct, F_mo_dt)
        D_mo_dt_prev = D_mo_dt


def construct_U_dt(F_mo_dt2, dt, U_ct):
    U_new = expm(-1j * F_mo_dt2 * dt)
    U_dt = U_new if U_ct is None else U_new @ U_ct

    # Check to see if matrix is Unitary
    unitary = np.conjugate(U_dt.T) @ U_dt
    if not (U_dt.shape[0] == U_dt.shape[1] and np.allclose(unitary, np.eye(U_dt.shape[0]))):
        raise ValueError(f"U^+ @ U is not a unitary matrix. Instead: {unitary}.")

    return U_dt


def propagate_density_matrix(dt, wfn, exc):
    # Here, when appended to a vars name
    # ct means current time
    # dt refers to ct + dt 
    # dt2 refers to ct + dt/2
    # ndt2 refers to ct - dt/2

    U_dt, F_mo_dt, F_mo_dt2, D_mo_dt = construct_D_mo_dt(wfn, dt, exc)

    # Everything is successful, so we save state
    mh.set_U_ct(U_dt)
    mh.set_F_mo_ct(F_mo_dt)
    mh.set_F_mo_ndt2(F_mo_dt2)
    mh.set_D_mo_ct(D_mo_dt)

    return D_mo_dt