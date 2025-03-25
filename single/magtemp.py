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


def interpolate(D1, D2):
    return 0.5 * D1 + 0.5 * D2


def propagate_density_matrix(dt, molecule, field):
    D_ao_t_minus_dt = molecule.get_D_ao_t_minus_dt()
    D_ao_t = molecule.get_D_ao_t()

    exc_t_plus_dt = qextrapolate(field.get_exc_t(), field.get_exc_t_minus_dt(), field.get_exc_t_minus_2dt(), dt) 

    D_ao_t_plus_dt = extrapolate(D_ao_t, D_ao_t_minus_dt)
    D_ao_t_plus_dt_guess = None

    for limit in range(0, 10000):
        if limit == 9999:
            raise RuntimeError(f"Predictor-corrector failed to converge in 10000 iterations")

        D_ao_t_plus_half_dt = interpolate(D_ao_t, D_ao_t_plus_dt)

        # ------------------- U for Magnus 2 ------------------------ #
        #          U_M4 = exp(-i * dt * F_ao_t_plus_half_dt)          #
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
