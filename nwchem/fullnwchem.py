import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from field import FIELD
from molecule import MOLECULE
from logging_utils import PrintLogger
from electric_field_generator import ElectricFieldGenerator
from csv_utils import initCSV, updateCSV
from plotting import show_eField_pField
from cli import parse_arguments
from volume import get_volume

# magnus.py
from scipy.linalg import expm

def prop_magnus_ord2_interpol(params, tdfock, F_mo_tm12dt, F_mo_t, D_mo_t):
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
    ValueError
        If nmats is not 1 or 2.
    RuntimeError
        If convergence is not achieved within max iterations.
    """
    # Time step and target time
    dt = params.dt

    # Step 1: Extrapolate F'(t + dt/2) = 2 * F'(t) - F'(t - dt/2)
    F_mo_tp12dt = 2 * F_mo_t - F_mo_tm12dt

    # Step 2: Initial propagation of P'(t) to P'(t + dt)
    D_mo_tpdt = D_mo_t.copy()
    D_mo_tpdt = prop_magnus_ord2_step(params, dt, F_mo_tp12dt, D_mo_tpdt)

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
        D_mo_pdt_old = D_mo_tpdt.copy()

        # Step 3: Convert P'(t + dt) to AO basis and build F(t + dt)
        D_ao_tpdt = transform_D_mo_to_D_ao(D_mo_tpdt, params)
        F_ao_tpdt = tdfock(params, D_ao_tpdt)

        # Step 4: Convert F(t + dt) to MO basis and interpolate new F'(t + dt/2)
        F_mo_pdt = transform_F_ao_to_F_mo(F_ao_tpdt, params)
        F_mo_tp12dt = 0.5 * F_mo_pdt + 0.5 * F_mo_t

        # Step 5: Propagate P'(t) to P'(t + dt) with new F'(t + dt/2)
        D_mo_tpdt = D_mo_t.copy()
        D_mo_tpdt = prop_magnus_ord2_step(params, dt, F_mo_tp12dt, D_mo_tpdt)

        # Step 6: Check convergence
        if iinter > 1:
            diff = np.linalg.norm(D_mo_tpdt - D_mo_pdt_old, 'fro')
            logging.debug(f"Density matrix change after propagation: {diff}")
            if diff <= params.molecule.pcconv:
                num_same += 1

            if num_same >= params.terms_interpol:
                converged = True

        # Optionally skip interpolation after one iteration
        if hasattr(params, 'lskip_interpol') and params.lskip_interpol:
            converged = True
            print("Skipped Magnus interpolation")

    # Update input/output matrices for the next time step
    params.molecule.set_F_mo_t_minus_half_dt(F_mo_tp12dt)
    params.molecule.set_F_mo_t(F_mo_pdt)
    params.molecule.set_D_mo_t(D_mo_tpdt)
    return D_mo_tpdt, params

def prop_magnus_ord2_step(params, dt, F_mo_tp12dt, D_mo_t):
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
    W = -1.0j * zdt * F_mo_tp12dt

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
        if not is_hermitian(F_mo_tp12dt, tol_zero):
            raise RuntimeError("Fock matrix at t + dt/2 is not Hermitian")
        if not is_unitary(expW, tol_zero):
            raise RuntimeError("e^W is not unitary")
    
    # Ensure D_mo_t is complex
    if not np.iscomplexobj(D_mo_t):
        D_mo_t = D_mo_t.astype(complex)

    # Compute P'(t+dt) = e^W P'(t) (e^W)^+
    expW_dag = expW.conj().T  # Hermitian conjugate of e^W
    D_mo_t_plus_dt = expW @ D_mo_t @ expW_dag
    return D_mo_t_plus_dt

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fourier(polarizability_csv, output_image='spectrum.png'):
    # Load CSV (assumes columns: 'Timestamps (fs)', 'X Values', 'Y Values', 'Z Values')
    df = pd.read_csv(polarizability_csv, comment='#')
    
    time = df['Timestamps (fs)'].values * 1e-15  # Convert fs to seconds
    signal_x = df['X Values'].values
    signal_y = df['Y Values'].values
    signal_z = df['Z Values'].values

    # Time step (assumes uniform spacing)
    dt = time[1] - time[0]

    # Perform FFT
    fft_vals_x = np.fft.fft(signal_x)
    fft_vals_y = np.fft.fft(signal_y)
    fft_vals_z = np.fft.fft(signal_z)
    freqs = np.fft.fftfreq(len(time), d=dt)  # In Hz

    # Keep only positive frequencies
    mask = freqs > 0
    freqs = freqs[mask]
    fft_vals_x = fft_vals_x[mask]
    fft_vals_y = fft_vals_y[mask]
    fft_vals_z = fft_vals_z[mask]

    # Convert frequencies to wavelengths in nm
    c = 2.99792458e8  # speed of light in m/s
    wavelengths_nm = (c / freqs) * 1e9  # Convert to nm

    # Sort by wavelength for better plotting (descending wavelength = ascending freq)
    sort_indices = np.argsort(wavelengths_nm)
    wavelengths_nm = wavelengths_nm[sort_indices]
    fft_x = np.abs(fft_vals_x[sort_indices])
    fft_y = np.abs(fft_vals_y[sort_indices])
    fft_z = np.abs(fft_vals_z[sort_indices])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths_nm, fft_x, label='X polarization')
    plt.plot(wavelengths_nm, fft_y, label='Y polarization')
    plt.plot(wavelengths_nm, fft_z, label='Z polarization')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Magnitude')
    plt.title('Absorption Spectrum')
    plt.legend()
    plt.grid(True)
    plt.xlim(wavelengths_nm.min(), wavelengths_nm.max())  # Optional, ensures full range shown
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"Saved spectrum to {output_image}")

# main.py
if __name__ == "__main__":
    try:
        # Set up logging
        log_format = '%(levelname)s: %(message)s'
        args = parse_arguments()
        logger = logging.getLogger()
        # Clear any pre-existing handlers.
        if logger.hasHandlers():
            logger.handlers.clear()

        # Set log level based on verbosity.
        if args.verbose >= 2:
            logger.setLevel(logging.DEBUG)
        elif args.verbose == 1:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        # Use FileHandler if a log file is specified; otherwise, use StreamHandler.
        if args.log:
            handler = logging.FileHandler(args.log, mode='w')
        else:
            handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(handler)

        # Optional: Prevent propagation to avoid duplicate logging
        logger.propagate = False
    
        sys.stdout = PrintLogger(logger, logging.INFO)
        logging.getLogger("h5py").setLevel(logging.INFO)
    
        logger.debug(f"Arguments given and parsed correctly: {args}")
    
        # Initialize the molecule and extract parameters
        molecule = MOLECULE(args.bohr, args.pcconv)
    
        # Parameters for the electric field
        wavelength = 250          # In nanometers
        peak_time = 5 * args.dt  # time of pulse peak in fs
        width = 0.005                # width parameter in fs^-2 
        t_start = 0               # start time in fs
        t_end = 50.0              # end time in fs
        dt_fs = args.dt           # time step in fs
        kappa = 0.005             # intensity in au

        # Create time array
        time_values = np.arange(t_start, t_end + dt_fs, dt_fs)

        # Initialize the electric field generator
        field_generator = ElectricFieldGenerator(wavelength, peak_time, width, shape='kick', smoothing=False, dt_fs=dt_fs, kappa=kappa)

        # Create an interpolator for the electric field components
        logger.debug("Building an interpolation profile with ElectricFieldInterpolator")
    
        # Generate a new time grid with adjusted resolution (args.mult is a multiplier)
        interpolated_times = np.linspace(0, time_values[-1], int(len(time_values) * args.mult))
        time_step_fs = interpolated_times[1] - interpolated_times[0]
        interpolated_fields = field_generator.get_field_at(interpolated_times, args.dir)

        # Convert time in fs to au
        dt = dt_fs * 41.3413733
        logger.info(f"The timestep for this simulation is {dt} in au or {dt_fs} in fs.")

        # Initialize the interpolated electric field CSV file using initCSV
        interpolated_e_field_csv = "interpolated-E-Field.csv"
        initCSV(interpolated_e_field_csv, "Interpolated Electric Field measured in au.")
    
        # Append the interpolated data rows to the CSV file
        with open(interpolated_e_field_csv, 'a', newline='') as csvfile:
            import csv
            writer = csv.writer(csvfile)
            for t, field in zip(interpolated_times, interpolated_fields):
                writer.writerow([t, field[0], field[1], field[2]])
    
        class Params:
            tol_zero = args.pcconv # previously tol_interpol
            checklvl = 2 # ensure hermitian and unitary
            exp_method = 1  # scipy.expm()
            dt = dt_fs # time difference
            terms_interpol = 2
            lskip_interpol = False
            molecule = MOLECULE(args.bohr, args.pcconv)
            field = FIELD()
            nmats = 1

        params = Params()

        # Initialize CSV file for the polarizability field output
        polarizability_csv = "magnus-P-Field.csv"

        initCSV(polarizability_csv, "Molecule's Polarizability Field measured in au.")
    
        # Log non-comment lines from the Bohr input file
        bohr_input_path = os.path.abspath(args.bohr)
        with open(bohr_input_path, 'r') as bohr_file:
            for line in bohr_file:
                if not line.strip().startswith(('#', '--', '%')):
                    logger.info('\t%s', line.rstrip('\n'))
        
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

        def tdfock(params, D_ao):
            """
            Builds the Fock matrix by including the external field.

            Parameters:
                wfn: Wavefunction object.
                D_ao (np.ndarray): Density matrix in atomic orbital basis.
                exc (array-like): External electric field components.

            Returns:
                np.ndarray: The computed Fock matrix.
            """
            wfn = params.molecule.wfn
            exc = params.field.get_exc_t_plus_dt()
            
            logging.debug(f"Electric field at t + dt: {exc}")
            ext = sum(wfn.mu[dir] * exc[dir] for dir in range(3))
            logging.debug(f"Dipole interaction term: {np.linalg.norm(ext)}")
            F_ao = JK(wfn, D_ao) - ext
            return F_ao
            
        # Main loop
        for index, current_time in enumerate(interpolated_times):
            F_mo_tm12dt = params.molecule.get_F_mo_t_minus_half_dt()
            F_mo = params.molecule.get_F_mo_t()
            D_mo = params.molecule.get_D_mo_t()

            params.field.set_exc_t(interpolated_fields[index]) # This would be the previous electric field in meep
            params.field.set_exc_t_plus_dt(interpolated_fields[index + 1] if index + 1 < len(interpolated_fields) else interpolated_fields[index]) # this would be the most recent e field
            
            # mu.py
            mu_arr = np.zeros(3)
            D_mo_t_plus_dt, params = prop_magnus_ord2_interpol(params, tdfock, F_mo_tm12dt, F_mo, D_mo)
            D_ao_t_plus_dt = transform_D_mo_to_D_ao(D_mo_t_plus_dt, params)
            volume = get_volume(molecule.molecule["coords"])

            for i in [0, 1, 2]:
                mu_arr[i] = 2 * float((np.trace(params.molecule.wfn.mu[i] @ D_ao_t_plus_dt) - np.trace(params.molecule.wfn.mu[i] @ params.molecule.wfn.D[0])).real)

            mu_arr /= volume

            logging.debug(f"At {current_time} fs, combined Bohr output is {mu_arr} in au")
            updateCSV(polarizability_csv, current_time, *mu_arr)

        # Plot the results using the interpolated electric field data
        show_eField_pField(interpolated_e_field_csv, polarizability_csv)

        fourier(polarizability_csv)
        logging.info("Simulation completed successfully.")
    except Exception as err:
        logging.exception(f"Simulation aborted due to an error: {err}")

