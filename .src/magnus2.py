# magnus2.py
import logging
import numpy as np
from scipy.linalg import expm

logger = logging.getLogger("main")

def propagate_density(params, molecule, exc):
    """
    Propagate complex density matrix forward using second-order Magnus expansion with
    interpolated and converged Fock matrix. Works for closed-shell.

    Parameters:
    -----------
    params : object
        Parameter object containing dt (time step), tol_interpol (convergence tolerance),
        terms_interpol (number of identical iterations for convergence)
    tt : float
        Current time.
    tdfock : callable
        External function to compute Fock matrices in AO basis from density matrices.
        Signature: F_ao, energies = tdfock(params, time, D_ao)
    F_mo_tm12dt : list of ndarray
        Fock matrices at t - dt/2 in MO basis, updated at end for next step.
    F_mo : list of ndarray
        Fock matrices at t in MO basis, updated at end for next step.
    energies : object
        Object to store energy values, updated by tdfock.
    D_mo : list of ndarray
        Density matrices in MO basis, input as P'(t), output as P'(t+dt).

    Raises:
    -------
    RuntimeError
        If convergence is not achieved within max iterations.
    """
    # Grab all important matrices
    F_mo_tm12dt = molecule.F_mo_tm12dt
    F_mo_t = molecule.F_mo_t
    D_mo_t = molecule.D_mo_t

    # Step 1: Extrapolate F'(t + dt/2) = 2 * F'(t) - F'(t - dt/2)
    F_mo_tp12dt = 2 * F_mo_t - F_mo_tm12dt

    # Step 2: Initial propagation of P'(t) to P'(t + dt)
    D_mo_tpdt = D_mo_t.copy()
    D_mo_tpdt = unitary_propagation(params, molecule, F_mo_tp12dt, D_mo_tpdt)

    # Interpolation loop for self-consistency
    converged = False
    iinter = 0
    num_same = 0
    max_iterations = params.max_iter
    while not converged:
        iinter += 1
        if iinter > max_iterations:
            raise RuntimeError(f"Failed to converge within {max_iterations} iterations")

        # Store previous P'(t + dt) for convergence check
        D_mo_pdt_old = D_mo_tpdt.copy()

        # Step 3: Convert P'(t + dt) to AO basis and build F(t + dt)
        D_ao_tpdt = molecule.transform_D_mo_to_D_ao(D_mo_tpdt)
        F_ao_tpdt = molecule.tdfock(exc, D_ao_tpdt)

        # Step 4: Convert F(t + dt) to MO basis and interpolate new F'(t + dt/2)
        F_mo_pdt = molecule.transform_F_ao_to_F_mo(F_ao_tpdt)
        F_mo_tp12dt = 0.5 * F_mo_pdt + 0.5 * F_mo_t

        # Step 5: Propagate P'(t) to P'(t + dt) with new F'(t + dt/2)
        D_mo_tpdt = D_mo_t.copy()
        D_mo_tpdt = unitary_propagation(params, molecule, F_mo_tp12dt, D_mo_tpdt)

        # Step 6: Check convergence
        if iinter > 1:
            diff = np.linalg.norm(D_mo_tpdt - D_mo_pdt_old, 'fro')
            logging.debug(f"Density matrix change after propagation: {diff}")
            if diff <= params.pcconv:
                num_same += 1

            if num_same >= params.terms_interpol:
                converged = True


    # Update input/output matrices for the next time step
    molecule.F_mo_tm12dt = F_mo_tp12dt
    molecule.F_mo_t = F_mo_pdt
    molecule.D_mo_t = D_mo_tpdt

    return D_mo_tpdt

def unitary_propagation(params, molecule, F_mo_tp12dt, D_mo_t):
    """
    Propagate density matrix forward by dt using second-order Magnus expansion in MO basis.
    Computes P'(t+dt) = e^W P'(t) (e^W)^+, where W = -i dt F'(t+dt/2).

    Parameters:
    -----------
    params : object
        Parameter object containing:
        - ns_mo: Number of molecular orbitals.
        - tol_zero: Tolerance for numerical checks (e.g., Hermitian, unitary).
        - checklvl: Level of matrix property checks (0, 1, 2+).
        - exp_method: Method for matrix exponentiation (1 for series, 2 for diagonalization).
    F_mo_mid : ndarray
        Complex Fock matrix in MO basis at t + dt/2, shape (ns_mo, ns_mo).
    D_mo : ndarray
        Complex density matrix in MO basis, input as P'(t), updated in place to P'(t+dt),
        shape (ns_mo, ns_mo).

    Raises:
    -------
    RuntimeError
        If F_mo_mid is not Hermitian or e^W is not unitary when checks are enabled.
    """
    # Compute W = -i dt F'(t+dt/2)
    zdt = complex(params.dt, 0.0)
    W = -1.0j * zdt * F_mo_tp12dt

    # Choose exponentiation method
    exp_method = params.exp_method
    if exp_method == 1:
        expW = exp_pseries(W)
    elif exp_method == 2:
        expW = exp_diag(W)
    else:
        raise ValueError(f"Invalid exp_method: {exp_method}")

    # Matrix property checks
    tol_zero = params.tol_zero
    doublecheck = params.doublecheck
    if doublecheck:
        if not molecule.is_hermitian(F_mo_tp12dt, tol_zero):
            raise ValueError("Fock matrix at t + dt/2 is not Hermitian")
        if not molecule.is_unitary(expW, tol_zero):
            raise ValueError("e^W is not unitary")
    
    # Ensure D_mo_t is complex
    if not np.iscomplexobj(D_mo_t):
        D_mo_t = D_mo_t.astype(complex)

    # Compute P'(t+dt) = e^W P'(t) (e^W)^+
    expW_dag = expW.conj().T  # Hermitian conjugate of e^W
    D_mo_t_plus_dt = expW @ D_mo_t @ expW_dag
    return D_mo_t_plus_dt

def exp_pseries(W):
    """
    Compute matrix exponential e^W using a power series expansion (placeholder).

    Parameters:
    -----------
    W : ndarray
        Matrix to exponentiate, shape (ns_mo, ns_mo).

    Returns:
    --------
    expW : ndarray
        Matrix exponential e^W.
    """
    # Placeholder: Use SciPy's expm for now
    return expm(W)

def exp_diag(W):
    """
    Compute matrix exponential e^W using diagonalization.

    Parameters:
    -----------
    W : ndarray
        Matrix to exponentiate, shape (ns_mo, ns_mo).

    Returns:
    --------
    expW : ndarray
        Matrix exponential e^W.
    """
    # Diagonalize W = V @ diag(evals) @ V^(-1)
    evals, V = np.linalg.eig(W)
    # Compute e^W = V @ diag(exp(evals)) @ V^(-1)
    expW = V @ np.diag(np.exp(evals)) @ np.linalg.inv(V)
    return expW
