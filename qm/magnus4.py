# magnus4.py
from scipy.linalg import expm
import numpy as np
import logging


def euclidean_norm_difference(matrix1, matrix2):
    return np.linalg.norm(matrix1 - matrix2, 'fro')


def qextrapolate(exc_t, exc_t_minus_dt, exc_t_minus_2dt, dt):
    dE_dt = (exc_t - exc_t_minus_dt) / dt
    d2E_dt2 = ((exc_t - exc_t_minus_dt) - (exc_t_minus_dt - exc_t_minus_2dt)) / dt**2
    E_tpdt = exc_t + dE_dt * dt + 0.5 * d2E_dt2 * dt**2
    return E_tpdt


def extrapolate(D1, D0):
    return 2 * D1 - D0


def interpolate(D1, D2, tn):
    return D1 + (0.5 + ((-1)**tn) * np.sqrt(3)/6)*(D2 - D1)


def construct_modified_Hamiltonian(wfn, V_ks_t_plus_t1, V_ks_t_plus_t2, dt):
    # ------------------ Hbar for Magnus 4 ---------------------- #
    #           Hbar = T + 0.5*(V_KS(t_1) + V_KS(t_2))            #
    # ----------------------------------------------------------- #
    Hbar = wfn.T + 0.5*(V_ks_t_plus_t1 + V_ks_t_plus_t2)
    
    # ------------------ dVbar for Magnus 4 --------------------- #
    #     dVbar = np.sqrt(3)/12 * dt * (V_KS(t_2) - V_KS(t_1))    #
    # ----------------------------------------------------------- #
    dVbar = (np.sqrt(3) / 12)*dt*(V_ks_t_plus_t1 - V_ks_t_plus_t2)

    # ------------- V_nonlocal_ext for Magnus 4 ----------------- #
    #             V_nonlocal_ext = V_nuc - V_local_nuc            #
    #    (Only used for Effective Core Potential Calculations)    #
    # ----------------------------------------------------------- #
    V_nonlocal_ext = 0 # Not using Effective Core Potential
    com1 = wfn.T + V_nonlocal_ext

    # ------------------ H_M4 for Magnus 4 ---------------------- #
    #          H_M4 = Hbar + i*[T + V_nonlocal_ext, dVbar]         #
    # ----------------------------------------------------------- #
    Hm4 = Hbar + 1j*(com1 @ dVbar - dVbar @ com1)
    return Hm4


def construct_V_KS(exc, D_ao_t_plus_tn, wfn):
    # ---------------- V_local_ext for Magnus 4 ----------------- #
    #             V_local_ext = Vne - sum(mu_i * E_i)             #
    # ----------------------------------------------------------- #
    V_local_ext = wfn.Vne - sum(wfn.mu[dir] * exc[dir] for dir in range(3))

    # ------------- V_nonlocal_ext for Magnus 4 ----------------- #
    #             V_nonlocal_ext = V_nuc - V_local_nuc            #
    #    (Only used for Effective Core Potential Calculations)    #
    # ----------------------------------------------------------- #
    V_nonlocal_ext = 0 # Not using Effective Core Potential

    # ------------------ V_eff_t for Magnus 4 ------------------- #
    #           V_eff_t = wfn.jk.get_veff(mol, 2*P_t)             #
    # ----------------------------------------------------------- #
    V_eff_t = wfn.jk.get_veff(wfn.ints_factory, 2 * D_ao_t_plus_tn)

    # ------------------ V_KS for Magnus 4 ---------------------- #
    #        V_KS = V_local_ext + V_nonlocal_ext + V_eff_t        #
    # ----------------------------------------------------------- #
    V_ks_t_plus_tn = V_local_ext + V_nonlocal_ext + V_eff_t
    return V_ks_t_plus_tn

def propagate_density_matrix(dt, molecule, field):
    D_ao_t_minus_dt = molecule.get_D_ao_t_minus_dt()
    D_ao_t = molecule.get_D_ao_t()

    exc_t_plus_dt = qextrapolate(field.get_exc_t(), field.get_exc_t_minus_dt(), field.get_exc_t_minus_2dt(), dt) 
    exc_t_plus_t1 = interpolate(field.get_exc_t(), exc_t_plus_dt, 1)
    exc_t_plus_t2 = interpolate(field.get_exc_t(), exc_t_plus_dt, 2)

    D_ao_t_plus_dt = extrapolate(D_ao_t, D_ao_t_minus_dt)
    D_ao_t_plus_dt_guess = None

    for limit in range(0, 10000):
        if limit == 9999:
            raise RuntimeError(f"Predictor-corrector failed to converge in 10000 iterations")

        D_ao_t_plus_t1 = interpolate(D_ao_t, D_ao_t_plus_dt, 1)
        D_ao_t_plus_t2 = interpolate(D_ao_t, D_ao_t_plus_dt, 2)

        V_ks_t_plus_t1 = construct_V_KS(exc_t_plus_t1, D_ao_t_plus_t1, molecule.wfn)
        V_ks_t_plus_t2 = construct_V_KS(exc_t_plus_t2, D_ao_t_plus_t2, molecule.wfn)

        Hm4_ao = construct_modified_Hamiltonian(molecule.wfn, V_ks_t_plus_t1, V_ks_t_plus_t2, dt)

        # ------------------- U for Magnus 4 ------------------------ #
        #                 U_M4 = exp(-i * dt * H_M4)                  #
        # ----------------------------------------------------------- #
        U_t_plus_dt = expm(-1j * Hm4_ao * dt)
        unitary = np.conjugate(U_t_plus_dt.T) @ U_t_plus_dt
        if not (U_t_plus_dt.shape[0] == U_t_plus_dt.shape[1] and np.allclose(unitary, np.eye(U_t_plus_dt.shape[0]), atol=1e-10)):
            raise ValueError(f"U^+ @ U is not a unitary matrix. Instead: {unitary}.")

        D_ao_t_plus_dt = U_t_plus_dt @ D_ao_t @ np.conjugate(U_t_plus_dt.T)

        # Check norm between most recent D_ao_t_plus_dt and previous D_ao_t_plus_dt_guess
        if D_ao_t_plus_dt_guess is not None:
            difference = euclidean_norm_difference(D_ao_t_plus_dt, D_ao_t_plus_dt_guess)
            if difference < molecule.pcconv:
                # Success! Now save state and get out.
                logging.debug(f"Predictor-Corrector scheme finished in {limit} iterations.")
                molecule.set_D_ao_t_minus_dt(D_ao_t)
                molecule.set_D_ao_t(D_ao_t_plus_dt)
                return D_ao_t_plus_dt
            
        # If they are too different, we run the process again
        D_ao_t_plus_dt_guess = D_ao_t_plus_dt
