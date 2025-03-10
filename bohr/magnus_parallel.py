# propagation.py
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

def build_fock(wfn, D_ao, exc, dir):
    """
    Builds the Fock matrix for a given direction by including the external field.

    Parameters:
        wfn: Wavefunction object.
        D_ao (np.ndarray): Density matrix in atomic orbital basis.
        exc (array-like): External electric field components.
        dir (int): Spatial direction (0: x, 1: y, 2: z).

    Returns:
        np.ndarray: The computed Fock matrix.
    """
    ext = wfn.mu[dir] * exc[dir]
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

def propagate_density_matrix(dt, molecule, exc, dir):
    """
    Propagates the density matrix for one time step using a predictor-corrector scheme.

    Parameters:
        dt (float): Time step.
        molecule (MOLECULE): The molecule instance.
        exc (array-like): External electric field components.
        dir (int): Spatial direction (0: x, 1: y, 2: z).

    Returns:
        np.ndarray: Updated density matrix in atomic orbital basis at t + dt.

    Raises:
        RuntimeError: If the predictor-corrector scheme fails to converge.
    """
    F_mo_t_minus_half_dt = molecule.get_F_mo_t_minus_half_dt(dir)
    F_mo_t = molecule.get_F_mo_t(dir)
    F_mo_t_plus_half_dt = extrapolate(F_mo_t, F_mo_t_minus_half_dt)
    D_mo_t_plus_dt_guess = None

    for limit in range(0, 10000):
        if limit == 9999:
            raise RuntimeError(f"Predictor-corrector failed to converge in 10000 iterations for dir {dir}")
        U_t_plus_dt = construct_U_t_plus_dt(F_mo_t_plus_half_dt, dt)
        D_mo_t = molecule.get_D_mo_t(dir)
        D_mo_t_plus_dt = U_t_plus_dt @ D_mo_t @ np.conjugate(U_t_plus_dt.T)
        D_ao_t_plus_dt = molecule.wfn.C[0] @ D_mo_t_plus_dt @ molecule.wfn.C[0].T
        F_ao_t_plus_dt = build_fock(molecule.wfn, D_ao_t_plus_dt, exc, dir)
        F_mo_t_plus_dt = molecule.wfn.C[0].T @ F_ao_t_plus_dt @ molecule.wfn.C[0]

        if D_mo_t_plus_dt_guess is not None:
            if euclidean_norm_difference(D_mo_t_plus_dt, D_mo_t_plus_dt_guess) < 1e-12:
                logging.debug(f"Predictor-Corrector scheme finished for the {'xyz'[dir]} direction in {limit} iterations.")
                molecule.set_F_mo_t_minus_half_dt(F_mo_t_plus_half_dt, dir)
                molecule.set_F_mo_t(F_mo_t_plus_dt, dir)
                molecule.set_D_mo_t(D_mo_t_plus_dt, dir)
                return D_ao_t_plus_dt
        F_mo_t_plus_half_dt = interpolate(F_mo_t, F_mo_t_plus_dt)
        D_mo_t_plus_dt_guess = D_mo_t_plus_dt

def propagate_direction_worker(propagator, direction, time_steps, fields, dt, molecule, output_queue):
    """
    Worker thread function that propagates the molecule in one spatial direction
    for each time step and pushes the result into a shared queue.

    Parameters:
        propagator (function): The propagation function.
        direction (int): Spatial direction (0: x, 1: y, 2: z).
        time_steps (iterable): List of time steps.
        fields (np.ndarray): Array of electric field data corresponding to each time step.
        dt (float): Time step interval.
        molecule (MOLECULE): The molecule instance.
        output_queue (queue.Queue): Thread-safe queue for outputting results.
    """
    for idx, t in enumerate(time_steps):
        try:
            result = propagator(dt, molecule, fields[idx], direction)
            output_queue.put((idx, result))
        except Exception as e:
            logging.error(f"Error in direction {'xyz'[direction]} at time {t} fs: {e}")
            output_queue.put((idx, e))
    output_queue.put((None, None))

def combine_propagation_results(time_steps, molecule, result_queues, polarizability_csv, volume):
    """
    Coordinator function that waits for complete propagation results from all three directions
    for each time step, computes the induced dipole, and writes to a CSV file.
    Aborts processing if an error is encountered.

    Parameters:
        time_steps (iterable): List of time steps.
        molecule (MOLECULE): The molecule instance.
        result_queues (dict): Dictionary mapping each direction to its result queue.
        polarizability_csv (str): CSV file path for polarizability output.
        volume (float): Volume of the molecule in atomic units.
    """
    results_buffer = {}
    current_index = 0
    completed = [False, False, False]

    while True:
        if all(completed) and current_index not in results_buffer:
            break

        for d in range(3):
            if not completed[d]:
                try:
                    idx, res = result_queues[d].get(timeout=0.1)
                    if idx is None:
                        completed[d] = True
                    else:
                        if isinstance(res, Exception):
                            msg = f"Worker error in direction {'xyz'[d]} at time step {time_steps[idx]} fs: {res}"
                            logging.error(msg)
                            raise RuntimeError(msg)
                        if idx not in results_buffer:
                            results_buffer[idx] = {}
                        results_buffer[idx][d] = res
                except Exception:
                    continue

        if current_index in results_buffer and len(results_buffer[current_index]) == 3:
            res_x = results_buffer[current_index][0]
            res_y = results_buffer[current_index][1]
            res_z = results_buffer[current_index][2]

            induced_matrix = np.zeros((3, 3))
            for i in range(3):
                mu_x = np.trace(molecule.wfn.mu[i] @ res_x) - np.trace(molecule.wfn.mu[i] @ molecule.wfn.D[0])
                mu_y = np.trace(molecule.wfn.mu[i] @ res_y) - np.trace(molecule.wfn.mu[i] @ molecule.wfn.D[0])
                mu_z = np.trace(molecule.wfn.mu[i] @ res_z) - np.trace(molecule.wfn.mu[i] @ molecule.wfn.D[0])
                induced_matrix[i, 0] = mu_x.real
                induced_matrix[i, 1] = mu_y.real
                induced_matrix[i, 2] = mu_z.real

            collapsed_output = np.sum(induced_matrix, axis=1) / volume
            current_time = time_steps[current_index]
            logging.debug(f"At {current_time} fs, combined Bohr output is {collapsed_output} in AU")
            from csv_utils import updateCSV  # Import here to avoid circular dependency
            updateCSV(polarizability_csv, current_time, *collapsed_output)
            
            del results_buffer[current_index]
            current_index += 1