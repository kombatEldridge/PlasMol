import numpy as np
from fock_builder import build_fock
import matrix_handler as mh
from scipy.linalg import expm

# Here, when appended to a vars name
    # ct means current time
    # dt refers to ct + dt 
    # dt2 refers to ct + dt/2


def construct_U_dt(F_mo_dt2, dt, U_ct):
    U_new = expm(-1j * F_mo_dt2 * dt)
    U_dt = U_new if U_ct is None else U_new @ U_ct

    # Check to see if matrix is Unitary
    unitary = np.conjugate(U_dt.T) @ U_dt
    if not (U_dt.shape[0] == U_dt.shape[1] and np.allclose(unitary, np.eye(U_dt.shape[0]))):
        raise ValueError(f"U^+ @ U is not a unitary matrix. Instead: {unitary}.")

    return U_dt


def construct_D_mo_dt(wfn, dt, exc, D_mo_0):
    F_mo_dt2 = mh.get_F_mo_dt2()
    U_ct = mh.get_U_ct()
    U_dt = construct_U_dt(F_mo_dt2, dt, U_ct)
    D_mo_dt = U_dt @ D_mo_0 @ np.conjugate(U_dt.T)

    D_ao_dt = wfn.C[0] @ D_mo_dt @ wfn.C[0].T
    F_ao_dt = build_fock(wfn, D_ao_dt, exc)
    F_mo_dt = wfn.C[0].T @ F_ao_dt @ wfn.C[0]

    return U_dt, F_mo_dt, D_mo_dt


def propagate_density_matrix(dt, wfn, exc, D_mo_0):
    U_dt, F_mo_dt, D_mo_dt = construct_D_mo_dt(wfn, dt, exc, D_mo_0)

    # Everything is successful, so we save state
    mh.set_F_mo_ct(mh.get_F_mo_dt2())
    mh.set_F_mo_dt2(F_mo_dt)
    
    mh.set_D_mo_ct(mh.get_D_mo_dt2())
    mh.set_D_mo_dt2(D_mo_dt)
    
    mh.set_U_ct(mh.get_U_dt2())
    mh.set_U_dt2(U_dt)

    return D_mo_dt