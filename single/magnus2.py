# magnus.py
import numpy as np
from scipy.linalg import expm
import logging

def JK(wfn, D_ao):
    """
    Computes the effective potential and returns the Fock matrix component.

    Parameters:
        wfn: Wavefunction object containing molecular integrals.
        D_ao: Density matrix in atomic orbital basis.

    Returns:
        np.ndarray: The computed Fock matrix.
    """
    pot = wfn.jk.get_veff(wfn.ints_factory, 2 * D_ao)
    Fa = wfn.T + wfn.Vne + pot
    return Fa

def build_fock(wfn, D_ao, exc):
    """
    Builds the Fock matrix by including the external field.

    Parameters:
        wfn: Wavefunction object.
        D_ao (np.ndarray): Density matrix in atomic orbital basis.
        exc (array-like): External electric field components.

    Returns:
        np.ndarray: The computed Fock matrix.
    """
    ext = sum(wfn.mu[dir] * exc[dir] for dir in range(3))
    F_ao = JK(wfn, D_ao) - ext
    return F_ao

def extrapolate(F_mo_t, F_mo_t_minus_half_dt):
    """
    Extrapolates the Fock matrix at t + dt/2 to t + dt using Repisky2015 Eq. 18.

    Parameters:
        F_mo_t (np.ndarray): Fock matrix at time t.
        F_mo_t_minus_half_dt (np.ndarray): Fock matrix at time t - dt/2.

    Returns:
        np.ndarray: Extrapolated Fock matrix at t + dt/2.
    """
    return 2 * F_mo_t - F_mo_t_minus_half_dt

def interpolate(F_mo_t, F_mo_t_plus_dt):
    """
    Interpolates the Fock matrix using Repisky2015 Eq. 19.

    Parameters:
        F_mo_t (np.ndarray): Fock matrix at time t.
        F_mo_t_plus_dt (np.ndarray): Fock matrix at time t + dt.

    Returns:
        np.ndarray: Interpolated Fock matrix.
    """
    return 0.5 * F_mo_t + 0.5 * F_mo_t_plus_dt

def euclidean_norm_difference(matrix1, matrix2):
    """
    Computes the Frobenius norm of the difference between two matrices.

    Parameters:
        matrix1 (np.ndarray): First matrix.
        matrix2 (np.ndarray): Second matrix.

    Returns:
        float: The Frobenius norm of the difference.
    """
    return np.linalg.norm(matrix1 - matrix2, 'fro')

def construct_U_t_plus_dt(F_mo_t_plus_half_dt, dt):
    """
    Constructs the unitary propagator U(t + dt) using the matrix exponential.

    Parameters:
        F_mo_t_plus_half_dt (np.ndarray): Fock matrix at time t + dt/2.
        dt (float): Time step.

    Returns:
        np.ndarray: Unitary propagator matrix.

    Raises:
        ValueError: If the input matrix is not Hermitian or the resulting matrix is not unitary.
    """
    if not np.allclose(F_mo_t_plus_half_dt, np.conjugate(F_mo_t_plus_half_dt.T), atol=1e-8):
        raise ValueError("F_mo_t_plus_half_dt is not Hermitian.")
    
    U_t_plus_dt = expm(-1j * F_mo_t_plus_half_dt * dt)
    unitary = np.conjugate(U_t_plus_dt.T) @ U_t_plus_dt
    if not (U_t_plus_dt.shape[0] == U_t_plus_dt.shape[1] and np.allclose(unitary, np.eye(U_t_plus_dt.shape[0]), atol=1e-10)):
        raise ValueError(f"U^+ @ U is not a unitary matrix. Instead: {unitary}.")
    return U_t_plus_dt

def propagate_density_matrix(dt, molecule, exc):
    """
    Propagates the density matrix for one time step using a predictor-corrector scheme.

    Parameters:
        dt (float): Time step.
        molecule (MOLECULE): The molecule instance.
        exc (array-like): External electric field components.

    Returns:
        np.ndarray: Updated density matrix in atomic orbital basis at t + dt.

    Raises:
        RuntimeError: If the predictor-corrector scheme fails to converge.
    """
    F_mo_t_minus_half_dt = molecule.get_F_mo_t_minus_half_dt()
    F_mo_t = molecule.get_F_mo_t()
    F_mo_t_plus_half_dt = extrapolate(F_mo_t, F_mo_t_minus_half_dt)
    D_mo_t_plus_dt_guess = None

    for limit in range(0, 10000):
        if limit == 9999:
            raise RuntimeError(f"Predictor-corrector failed to converge in 10000 iterations")
        U_t_plus_dt = construct_U_t_plus_dt(F_mo_t_plus_half_dt, dt)
        D_mo_t = molecule.get_D_mo_t()
        D_mo_t_plus_dt = U_t_plus_dt @ D_mo_t @ np.conjugate(U_t_plus_dt.T)
        D_ao_t_plus_dt = molecule.wfn.C[0] @ D_mo_t_plus_dt @ molecule.wfn.C[0].T
        F_ao_t_plus_dt = build_fock(molecule.wfn, D_ao_t_plus_dt, exc)
        F_mo_t_plus_dt = molecule.wfn.C[0].T @ F_ao_t_plus_dt @ molecule.wfn.C[0]

        if D_mo_t_plus_dt_guess is not None:
            if euclidean_norm_difference(D_mo_t_plus_dt, D_mo_t_plus_dt_guess) < molecule.pcconv:
                logging.debug(f"Predictor-Corrector scheme finished in {limit} iterations.")
                molecule.set_F_mo_t_minus_half_dt(F_mo_t_plus_half_dt)
                molecule.set_F_mo_t(F_mo_t_plus_dt)
                molecule.set_D_mo_t(D_mo_t_plus_dt)
                return D_ao_t_plus_dt
        F_mo_t_plus_half_dt = interpolate(F_mo_t, F_mo_t_plus_dt)
        D_mo_t_plus_dt_guess = D_mo_t_plus_dt
