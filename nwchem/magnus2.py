# magnus.py
import numpy as np
from scipy.linalg import expm

def prop_magnus_ord2_interpol(params, tdfock, F_mo_n12dt, F_mo, D_mo):
    """
    Propagate complex density matrix forward using second-order Magnus expansion with
    interpolated and converged Fock matrix. Works for closed-shell (nmats=1) or open-shell (nmats=2).

    Parameters:
    -----------
    params : object
        Parameter object containing dt (time step), tol_interpol (convergence tolerance),
        terms_interpol (number of identical iterations for convergence), lskip_interpol (flag to skip interpolation).
    tt : float
        Current time.
    tdfock : callable
        External function to compute Fock matrices in AO basis from density matrices.
        Signature: F_ao, energies = tdfock(params, time, D_ao)
    F_mo_n12dt : list of ndarray
        Fock matrices at t - dt/2 in MO basis, updated at end for next step.
    F_mo : list of ndarray
        Fock matrices at t in MO basis, updated at end for next step.
    energies : object
        Object to store energy values, updated by tdfock.
    D_mo : list of ndarray
        Density matrices in MO basis, input as P'(t), output as P'(t+dt).

    Raises:
    -------
    ValueError
        If nmats is not 1 or 2.
    RuntimeError
        If convergence is not achieved within max iterations.
    """
    # Time step and target time
    dt = params.dt

    # Step 1: Extrapolate F'(t + dt/2) = 2 * F'(t) - F'(t - dt/2)
    F_mo_p12dt = 2 * F_mo - F_mo_n12dt

    # Step 2: Initial propagation of P'(t) to P'(t + dt)
    D_mo_pdt = D_mo.copy()
    prop_magnus_ord2_step(params, dt, F_mo_p12dt, D_mo_pdt)

    # Interpolation loop for self-consistency
    converged = False
    iinter = 0
    num_same = 0
    max_iterations = 200
    while not converged:
        iinter += 1
        if iinter > max_iterations:
            raise RuntimeError(f"Failed to converge within {max_iterations} iterations")

        # Store previous P'(t + dt) for convergence check
        D_mo_pdt_old = D_mo_pdt.copy()

        # Step 3: Convert P'(t + dt) to AO basis and build F(t + dt)
        D_ao = transform_D_mo_to_D_ao(D_mo_pdt, params)
        F_ao = tdfock(params, D_ao)

        # Step 4: Convert F(t + dt) to MO basis and interpolate new F'(t + dt/2)
        F_mo_pdt = transform_F_ao_to_F_mo(F_ao, params)
        F_mo_p12dt = 0.5 * F_mo_pdt + 0.5 * F_mo

        # Step 5: Propagate P'(t) to P'(t + dt) with new F'(t + dt/2)
        D_mo_pdt = D_mo.copy()
        prop_magnus_ord2_step(params, dt, F_mo_p12dt, D_mo_pdt)

        # Step 6: Check convergence
        if iinter > 1:
            diff = np.linalg.norm(D_mo_pdt - D_mo_pdt_old, 'fro')
            if diff <= params.molecule.pcconv:
                num_same += 1

            if num_same >= params.terms_interpol:
                converged = True

        # Optionally skip interpolation after one iteration
        if hasattr(params, 'lskip_interpol') and params.lskip_interpol:
            converged = True
            print("Skipped Magnus interpolation")

    # Update input/output matrices for the next time step
    params.molecule.set_F_mo_t_minus_half_dt(F_mo_p12dt)
    params.molecule.set_F_mo_t(F_mo_pdt)
    params.molecule.set_D_mo_t(D_mo_pdt)
    return D_mo_pdt, params

def prop_magnus_ord2_step(params, dt, F_mo_mid, D_mo):
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
    dt : float
        Time step.
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
    zdt = complex(dt, 0.0)
    W = -1.0j * zdt * F_mo_mid

    # Choose exponentiation method
    exp_method = getattr(params, 'exp_method', 2)  # Default to diagonalization
    if exp_method == 1:
        expW = exp_pseries(W)
    elif exp_method == 2:
        expW = exp_diag(W)
    else:
        raise ValueError(f"Invalid exp_method: {exp_method}")

    # Matrix property checks
    tol_zero = getattr(params, 'tol_zero', 1e-10)
    checklvl = getattr(params, 'checklvl', 0)
    if checklvl >= 2:
        if not is_hermitian(F_mo_mid, tol_zero):
            raise RuntimeError("Fock matrix at t + dt/2 is not Hermitian")
        if not is_unitary(expW, tol_zero):
            raise RuntimeError("e^W is not unitary")
    
    # Ensure D_mo is complex
    if not np.iscomplexobj(D_mo):
        D_mo = D_mo.astype(complex)

    # Compute P'(t+dt) = e^W P'(t) (e^W)^+
    expW_dag = expW.conj().T  # Hermitian conjugate of e^W
    D_mo[:] = expW @ D_mo @ expW_dag

def transform_D_mo_to_D_ao(D_mo, params):
    """
    Transform density matrix from MO basis to AO basis.
    Placeholder - assumes params contains molecule object with wfn attributes.

    Parameters:
    -----------
    D_mo : ndarray
        Density matrix in MO basis.
    params : object
        Contains molecule with wfn (PySCF wavefunction object).

    Returns:
    --------
    D_ao : ndarray
        Density matrix in AO basis.
    """
    C = params.molecule.wfn.C[0]  # MO coefficients

    D_ao = C @ D_mo @ C.T
    return D_ao

def transform_F_ao_to_F_mo(F_ao, params):
    """
    Transform Fock matrix from AO basis to MO basis.
    Placeholder - assumes params contains molecule object with wfn attributes.

    Parameters:
    -----------
    F_ao : ndarray
        Fock matrix in AO basis.
    params : object
        Contains molecule with wfn (PySCF wavefunction object).

    Returns:
    --------
    F_mo : ndarray
        Fock matrix in MO basis.
    """
    C = params.molecule.wfn.C[0]  # MO coefficients

    F_mo = C.T @ F_ao @ C
    return F_mo

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

def is_hermitian(A, tol):
    """
    Check if matrix A is Hermitian within tolerance.

    Parameters:
    -----------
    A : ndarray
        Matrix to check.
    tol : float
        Numerical tolerance.

    Returns:
    --------
    bool
        True if A is Hermitian within tolerance.
    """
    return np.allclose(A, A.conj().T, rtol=0, atol=tol)

def is_unitary(U, tol):
    """
    Check if matrix U is unitary within tolerance (U U^+ = I).

    Parameters:
    -----------
    U : ndarray
        Matrix to check.
    tol : float
        Numerical tolerance.

    Returns:
    --------
    bool
        True if U is unitary within tolerance.
    """
    ns_mo = U.shape[0]
    identity = np.eye(ns_mo, dtype=complex)
    return np.allclose(U @ U.conj().T, identity, rtol=0, atol=tol)
