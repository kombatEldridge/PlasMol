import os
import sys
import csv
import math
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm

from pyscf import gto
from bohr_internals import input_parser
from bohr_internals import wavefunction

C_NM_FS = 299.792458  # speed of light in nm/fs

# electric_field_generator.py
class ElectricFieldGenerator:
    """
    Generates electric field components over time using a Gaussian-enveloped oscillatory function.
    """
    def __init__(self, wavelength, peak_time, width, smoothing=True, shape='pulse', dt_fs=None, kappa=None):
        """
        Initialize the generator with waveform parameters.

        Parameters:
            wavelength (float): Wavelength in nanometers
            peak_time (float): Time of pulse peak in femtoseconds
            width (float): Width parameter of Gaussian envelope in fs^{-2}
        """
        self.wavelength = wavelength
        self.peak_time = peak_time
        self.width = width
        self.frequency = C_NM_FS / wavelength
        self.smoothing = smoothing
        self.shape = shape
        self.dt_fs = dt_fs
        self.kappa = kappa


    def get_field_at(self, query_times, dir):
        """
        Returns the electric field components at the specified times.

        Parameters:
            query_times (array-like): Time values in femtoseconds to evaluate the field at
            dir (str): The direction ('x', 'y', or 'z') for the active component

        Returns:
            np.ndarray: A 2D array with columns [x, y, z] representing the field
        """
        # Convert query_times to a NumPy array
        t = np.asarray(query_times)
        
        # Compute the complex electric field
        if self.shape == 'pulse':
            complex_field = np.exp(1j * 2 * np.pi * self.frequency * (t - self.peak_time)) * \
                            np.exp(-self.width * (t - self.peak_time) ** 2)
        elif self.shape == 'kick':
            complex_field = self.kappa * np.exp(-((t - self.peak_time)**2) / (2 * self.width**2))
        
        # Extract the active component and apply threshold
        active_component = np.real(complex_field)
        active_component[abs(active_component) < 1e-20] = 0
        
        # Define a smoothing window at the start
        if self.smoothing:
            ramp_duration = 10  # Adjust this (in fs) to control the smoothing length
            t_start = t[0]  # Assume t is sorted and starts at the beginning
            window = np.ones_like(t)
            mask = t < (t_start + ramp_duration)
            window[mask] = 0.5 * (1 - np.cos(np.pi * (t[mask] - t_start) / ramp_duration))
        
            # Apply the window to the active component
            active_component *= window

        # Initialize a 2D array for the field components (all zeros initially)
        field = np.zeros((len(t), 3))
        
        # Map direction to column index and assign active component
        dir = dir.lower()  # Make direction case-insensitive
        if dir == 'x':
            field[:, 0] = active_component  # x-component
        elif dir == 'y':
            field[:, 1] = active_component  # y-component
        elif dir == 'z':
            field[:, 2] = active_component  # z-component
        else:
            raise ValueError("Invalid direction. Must be 'x', 'y', or 'z'.")
        
        # Return the field as a 2D array with columns [x, y, z]
        return field

# field.py
class FIELD():
    def __init__(self):
        self.exc_store = {}
        self.empty = np.array([0.0, 0.0, 0.0])

    def get_exc_t_plus_dt(self):
        key = 'exc_t_plus_dt'
        return self.exc_store.get(key, self.empty)

    def set_exc_t_plus_dt(self, exc_t_plus_dt):
        self.exc_store['exc_t_plus_dt'] = exc_t_plus_dt

    def get_exc_t(self):
        key = 'exc_t'
        return self.exc_store.get(key, self.empty)

    def set_exc_t(self, exc_t):
        self.exc_store['exc_t'] = exc_t

    def get_exc_t_minus_dt(self):
        key = 'exc_t_minus_dt'
        return self.exc_store.get(key, self.empty)

    def set_exc_t_minus_dt(self, exc_t_minus_dt):
        self.exc_store['exc_t_minus_dt'] = exc_t_minus_dt

    def get_exc_t_minus_2dt(self):
        key = 'exc_t_minus_2dt'
        return self.exc_store.get(key, self.empty)

    def set_exc_t_minus_2dt(self, exc_t_minus_2dt):
        self.exc_store['exc_t_minus_2dt'] = exc_t_minus_2dt

# logging_utils.py
class PrintLogger:
    """
    Redirects standard print statements to a logger.
    """
    def __init__(self, logger, level=logging.INFO):
        """
        Initialize the PrintLogger.

        Parameters:
            logger (logging.Logger): The logger instance.
            level (int): Logging level.
        """
        self.logger = logger
        self.level = level

    def write(self, message):
        """
        Write a message to the logger.

        Parameters:
            message (str): The message to log.
        """
        message = message.rstrip()
        if message:
            self.logger.log(self.level, message)

    def flush(self):
        """Dummy flush method for compatibility."""
        pass

# molecule.py
class MOLECULE():
    """
    Represents a molecule and its electronic structure, initializing the molecule using PySCF.

    Attributes:
        molecule (dict): Dictionary containing molecule parameters.
        method (dict): Dictionary with method-related options.
        matrix_store (dict): Dictionary storing matrices for each direction (x, y, z) and initial matrices.
        wfn (wavefunction.RKS): Wavefunction object computed from the molecule.
    """
    def __init__(self, inputfile, pcconv):
        """
        Initializes the molecule from an input file.

        Parameters:
            inputfile (str): Path to the input file containing molecule data.
        """
        from bohr_internals import options
        options = options.OPTIONS()
        self.molecule, self.method, basis = input_parser.read_input(inputfile, options)
        options.molecule = self.molecule

        self.pcconv = pcconv

        # Format molecule string as required by PySCF
        atoms = self.molecule["atoms"]
        pyscf_molecule = ""
        for index, atom in enumerate(atoms):
            pyscf_molecule += " " + atom
            pyscf_molecule += " " + str(self.molecule["coords"][atom+str(index+1)][0])
            pyscf_molecule += " " + str(self.molecule["coords"][atom+str(index+1)][1])
            pyscf_molecule += " " + str(self.molecule["coords"][atom+str(index+1)][2])
            if index != (len(atoms)-1):
                pyscf_molecule += ";"

        # Create PySCF molecule
        pyscf_mol = gto.M(atom=pyscf_molecule,
                          basis=basis["name"],
                          unit='B',
                          charge=int(options.charge),
                          spin=int(options.spin),
                          cart=options.cartesian)
        pyscf_mol.set_common_origin(self.molecule["com"])
        pyscf_mol.verbose = 0
        pyscf_mol.max_memory = options.memory
        pyscf_mol.build()

        # Initialize matrices and wavefunction
        self.matrix_store = {}
        self.wfn = wavefunction.RKS(pyscf_mol)
        rks_energy = self.wfn.compute(options)

        F_ao_0 = self.wfn.F[0]
        F_mo_0 = self.wfn.C[0].T @ F_ao_0 @ self.wfn.C[0]
        self.matrix_store['F_mo_0'] = F_mo_0

        self.D_ao_0 = self.wfn.D[0]
        self.D_mo_0 = self.wfn.C[0].T @ self.wfn.S @ self.D_ao_0 @ self.wfn.S @ self.wfn.C[0]
        self.matrix_store['D_mo_0'] = self.D_mo_0

        trace = np.trace(self.D_mo_0)
        n = self.wfn.nel[0]
        if not np.isclose(trace, n):
            raise ValueError(f"Trace of the matrix is not {n} (instead {trace}).")

    def get_F_mo_t(self):
        """
        Gets the Fock matrix at time t.

        Returns:
            np.ndarray: Fock matrix.
        """
        key = 'F_mo_t'
        return self.matrix_store.get(key, self.get_F_mo_0())

    def set_F_mo_t(self, F_mo_t):
        """
        Sets the Fock matrix at time t.

        Parameters:
            F_mo_t (np.ndarray): Fock matrix.
        """
        self.matrix_store['F_mo_t'] = F_mo_t
        # print("F_mo_t successfully set!")

    def get_F_mo_t_minus_half_dt(self):
        """
        Gets the Fock matrix at time t - dt/2.

        Returns:
            np.ndarray: Fock matrix.
        """
        key = 'F_mo_t_minus_half_dt'
        return self.matrix_store.get(key, self.get_F_mo_0())

    def set_F_mo_t_minus_half_dt(self, F_mo_t_minus_half_dt):
        """
        Sets the Fock matrix at time t - dt/2 for a given direction.

        Parameters:
            F_mo_t_minus_half_dt (np.ndarray): Fock matrix.
        """
        self.matrix_store['F_mo_t_minus_half_dt'] = F_mo_t_minus_half_dt
        # print("F_mo_t_minus_half_dt successfully set!")

    def get_F_mo_t_minus_dt(self):
        """
        Gets the Fock matrix at time t - dt.

        Returns:
            np.ndarray: Fock matrix.
        """
        key = 'F_mo_t_minus_dt'
        return self.matrix_store.get(key, self.get_F_mo_0())

    def set_F_mo_t_minus_dt(self, F_mo_t_minus_dt):
        """
        Sets the Fock matrix at time t - dt for a given direction.

        Parameters:
            F_mo_t_minus_dt (np.ndarray): Fock matrix.
        """
        self.matrix_store['F_mo_t_minus_dt'] = F_mo_t_minus_dt
        # print("F_mo_t_minus_dt successfully set!")

    def get_F_mo_0(self):
        """
        Returns the initial Fock matrix in the molecular orbital basis.

        Returns:
            np.ndarray: Initial Fock matrix.
        """
        return self.matrix_store['F_mo_0']
    
    def get_D_mo_t_minus_dt(self):
        """
        Gets the Density matrix at time t - dt.

        Returns:
            np.ndarray: Density matrix.
        """
        key = 'D_mo_t_minus_dt'
        return self.matrix_store.get(key, self.get_D_mo_0())

    def set_D_mo_t_minus_dt(self, D_mo_t_minus_dt):
        """
        Sets the Density matrix at time t - dt for a given direction.

        Parameters:
            F_mo_t_minus_dt (np.ndarray): Density matrix.
        """
        self.matrix_store['D_mo_t_minus_dt'] = D_mo_t_minus_dt
        # print("D_mo_t_minus_dt successfully set!")

    def get_D_mo_t(self):
        """
        Gets the density matrix at time t.

        Returns:
            np.ndarray: Density matrix.
        """
        key = 'D_mo_t'
        return self.matrix_store.get(key, self.get_D_mo_0())

    def set_D_mo_t(self, D_mo_t):
        """
        Sets the density matrix at time t.

        Parameters:
            D_mo_t (np.ndarray): Density matrix.
        """
        self.matrix_store['D_mo_t'] = D_mo_t
        # print("D_mo_t successfully set!")

    def get_D_mo_0(self):
        """
        Returns the initial density matrix.

        Returns:
            np.ndarray: Initial density matrix.
        """
        return self.matrix_store['D_mo_0']
    
    def get_D_ao_t_minus_dt(self):
        """
        Gets the Density matrix at time t - dt.

        Returns:
            np.ndarray: Density matrix.
        """
        key = 'D_ao_t_minus_dt'
        return self.matrix_store.get(key, self.get_D_ao_0())

    def set_D_ao_t_minus_dt(self, D_ao_t_minus_dt):
        """
        Sets the Density matrix at time t - dt for a given direction.

        Parameters:
            F_ao_t_minus_dt (np.ndarray): Density matrix.
        """
        self.matrix_store['D_ao_t_minus_dt'] = D_ao_t_minus_dt
        # print("D_ao_t_minus_dt successfully set!")

    def get_D_ao_t(self):
        """
        Gets the density matrix at time t.

        Returns:
            np.ndarray: Density matrix.
        """
        key = 'D_ao_t'
        return self.matrix_store.get(key, self.get_D_ao_0())

    def set_D_ao_t(self, D_ao_t):
        """
        Sets the density matrix at time t.

        Parameters:
            D_ao_t (np.ndarray): Density matrix.
        """
        self.matrix_store['D_ao_t'] = D_ao_t
        # print("D_ao_t successfully set!")

    def get_D_ao_0(self):
        """
        Returns the initial density matrix.

        Returns:
            np.ndarray: Initial density matrix.
        """
        return self.matrix_store['D_ao_0']

# cli.py
def parse_arguments():
    """
    Parses command-line arguments for the simulation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Meep simulation with Bohr dipole moment calculation."
    )
    parser.add_argument('-b', '--bohr', type=str, help="Path to the Bohr input file.")
    parser.add_argument('-e', '--csv', default="", type=str, help="(Optional) Path to the electric field CSV file.")
    parser.add_argument('-m', '--mult', type=float, default=1, help="(Optional) Multiplier for the electric field interpolator resolution.")
    parser.add_argument('-l', '--log', help="(Optional) Log file name")
    parser.add_argument('-v', '--verbose', action='count', default=0, help="(Optional) Increase verbosity.")
    parser.add_argument('-i', '--pcconv', type=float, default=1e-12, help="(Optional) Iteration convergence for Predictor-Corrector scheme in Magnus propagator.")
    parser.add_argument('-dt_fs', type=float, help="(Optional) The time step used in the simulation in fs.")
    parser.add_argument('-dt_au', type=float, help="(Optional) The time step used in the simulation in au.")
    parser.add_argument('-d', '--dir', type=str, default='z', help="(Optional) Direction string (x, y, or z) for the excited electric field.")

    args = parser.parse_args()
    if args.log and args.verbose == 0:
        args.verbose = 1

    return args

# csv_utils.py
def initCSV(filename, comment):
    """
    Initializes a CSV file with a header and comment lines.

    Parameters:
        filename (str): Path to the CSV file.
        comment (str): Comment to include at the beginning of the file.
    """
    with open(filename, 'w', newline='') as file:
        for line in comment.splitlines():
            file.write(f"# {line}\n")
        file.write("\n")
        writer = csv.writer(file)
        header = ['Timestamps (fs)', 'X Values', 'Y Values', 'Z Values']
        writer.writerow(header)

def updateCSV(filename, timestamp, x_value=None, y_value=None, z_value=None):
    """
    Appends a row of data to a CSV file.

    Parameters:
        filename (str): Path to the CSV file.
        timestamp (float): Time stamp.
        x_value (float, optional): Value for x component.
        y_value (float, optional): Value for y component.
        z_value (float, optional): Value for z component.
    """
    file_exists = os.path.exists(filename)
    row = [timestamp, x_value if x_value is not None else 0,
           y_value if y_value is not None else 0,
           z_value if z_value is not None else 0]
    
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "X", "Y", "Z"])
        writer.writerow(row)

def read_electric_field_csv(file_path):
    """
    Reads electric field values from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: Four lists corresponding to time values, and electric field components (x, y, z).
    """
    time_values, electric_x, electric_y, electric_z = [], [], [], []
    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row and row[0].startswith("Timestamps"):
                break
        for row in reader:
            if len(row) < 4:
                continue
            time_values.append(float(row[0]))
            electric_x.append(float(row[1]))
            electric_y.append(float(row[2]))
            electric_z.append(float(row[3]))
    return time_values, electric_x, electric_y, electric_z

# plotting.py
def show_eField_pField(eFieldFileName, pFieldFileName=None, matplotlibLocationIMG=None, matplotlibOutput=None):
    """
    Plots the electric field and, optionally, the polarizability field from CSV files.

    Parameters:
        eFieldFileName (str): CSV file path for the electric field.
        pFieldFileName (str, optional): CSV file path for the polarizability field.
        matplotlibLocationIMG (str, optional): Location for saving the image.
        matplotlibOutput (str, optional): Output name for the image.
    """
    logging.getLogger('matplotlib').setLevel(logging.INFO)

    if pFieldFileName is not None:
        logging.debug(f"Reading CSV files: {eFieldFileName} and {pFieldFileName}")
    else:
        logging.debug(f"Reading CSV file: {eFieldFileName}")

    def sort_csv_by_first_column(filename):
        """
        Sorts a CSV file by the first column (timestamps) while preserving comment lines.

        Parameters:
            filename (str): Path to the CSV file.
        """
        with open(filename, 'r') as file:
            lines = file.readlines()
        comments = [line for line in lines if line.startswith('#')]
        header = next(line for line in lines if not line.startswith('#'))
        data_lines = [line for line in lines if not line.startswith('#') and line != header]
        from io import StringIO
        data = pd.read_csv(StringIO(''.join(data_lines)))
        data_sorted = data.sort_values(by='Timestamps (fs)')
        with open(filename, 'w') as file:
            file.writelines(comments)
            file.write(header)
            data_sorted.to_csv(file, index=False)

    sort_csv_by_first_column(eFieldFileName)
    data1 = pd.read_csv(eFieldFileName, comment='#')
    data1 = data1.sort_values(by='Timestamps (fs)', ascending=True)
    timestamps1 = data1['Timestamps (fs)']
    x_values1 = data1['X Values']
    y_values1 = data1['Y Values']
    z_values1 = data1['Z Values']

    if pFieldFileName is not None:
        sort_csv_by_first_column(pFieldFileName)
        data2 = pd.read_csv(pFieldFileName, comment='#')
        data2 = data2.sort_values(by='Timestamps (fs)', ascending=True)
        timestamps2 = data2['Timestamps (fs)']
        x_values2 = data2['X Values']
        y_values2 = data2['Y Values']
        z_values2 = data2['Z Values']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.plot(timestamps1, x_values1, label='x', marker='o')
        ax1.plot(timestamps1, y_values1, label='y', marker='o')
        ax1.plot(timestamps1, z_values1, label='z', marker='o')
        ax1.set_title('Incident Electric Field')
        ax1.set_xlabel('Timestamps (fs)')
        ax1.set_ylabel('Electric Field Magnitude')
        ax1.legend()

        ax2.plot(timestamps2, x_values2, label='x', marker='o')
        ax2.plot(timestamps2, y_values2, label='y', marker='o')
        ax2.plot(timestamps2, z_values2, label='z', marker='o')
        ax2.set_title("Molecule's Response")
        ax2.set_xlabel('Timestamps (fs)')
        ax2.set_ylabel('Polarization Field Magnitude')
        ax2.legend()
    else:
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax1.plot(timestamps1, x_values1, label='x', marker='o')
        ax1.plot(timestamps1, y_values1, label='y', marker='o')
        ax1.plot(timestamps1, z_values1, label='z', marker='o')
        ax1.set_title('Incident Electric Field')
        ax1.set_xlabel('Timestamps (fs)')
        ax1.set_ylabel('Electric Field Magnitude')
        ax1.legend()

    plt.tight_layout()
    if matplotlibLocationIMG is None:
        if matplotlibOutput is None:
            plt.savefig('output.png', dpi=1000)
            logging.debug("Matplotlib image written: output.png")
        else:
            plt.savefig(f'{matplotlibOutput}.png', dpi=1000)
            logging.debug(f"Matplotlib image written: {matplotlibOutput}.png")
    elif matplotlibOutput is None:
        plt.savefig(f'{matplotlibLocationIMG}.png', dpi=1000)
        logging.debug(f"Matplotlib image written: {matplotlibLocationIMG}.png")
    else:
        plt.savefig(f'{matplotlibLocationIMG}{matplotlibOutput}.png', dpi=1000)
        logging.debug(f"Matplotlib image written: {matplotlibLocationIMG}{matplotlibOutput}.png")

# volume.py
def get_volume(xyz):
    # Define van der Waals radii (Å)
    vdw_radii = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52}

    volume = 0
    for atom in xyz:
        element = atom[0]
        if element in vdw_radii:
            radius = vdw_radii[element]
            volume += (4 / 3) * math.pi * (radius ** 3)
        else:
            print(f"Warning: No van der Waals radius for {element}")
        
    # Dividing by 0.14818471 Å³ will set the volume to atomic units.
    return volume / 0.14818471

# magnus.py
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
    return D_mo_tpdt

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

    F_mo = C.T @ F_ao @ C.T 
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

# fourier.py
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

        # Convert time in fs to au
        if (args.dt_fs):
            dt_fs = args.dt_fs
            dt_au = dt_fs * 41.3413733
        else:
            dt_au = args.dt_au
            dt_fs = dt_au / 41.3413733
        
        logger.info(f"The timestep for this simulation is {dt_au} in au or {dt_fs} in fs.")

        # Time step check
        if (dt_au < 0.1):
            raise ValueError(f"The timestep for this simulation are too small to elicit physical results.")
        
        # Parameters for the electric field
        wavelength = 250            # in nm
        peak_time = 5 * dt_fs       # time of pulse peak in fs
        width = 0.0005              # width parameter in fs^-2 
        t_start = 0                 # start time in fs
        t_end = 50.0                # end time in fs
        kappa = 5e-5                # intensity in au

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
            tol_zero = args.pcconv 
            checklvl = 2                # ensure hermitian and unitary
            exp_method = 1              # scipy.expm()
            dt = dt_au
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
            params.field.set_exc_t_plus_dt(interpolated_fields[index + 1] if index + 1 < len(interpolated_fields) else np.zeros(3)) # this would be the most recent e field

            # mu.py
            mu_arr = np.zeros(3)
            D_mo_t_plus_dt = prop_magnus_ord2_interpol(params, tdfock, F_mo_tm12dt, F_mo, D_mo)
            D_ao_t_plus_dt = transform_D_mo_to_D_ao(D_mo_t_plus_dt, params)

            # volume = get_volume(params.molecule.molecule["coords"])

            for i in [0, 1, 2]:
                mu_arr[i] = 2 * float((np.trace(params.molecule.wfn.mu[i] @ D_ao_t_plus_dt) - np.trace(params.molecule.wfn.mu[i] @ params.molecule.wfn.D[0])).real)

            # mu_arr /= volume

            logging.debug(f"At {current_time} fs, combined Bohr output is {mu_arr} in au")
            updateCSV(polarizability_csv, current_time, *mu_arr)

        # Plot the results using the interpolated electric field data
        show_eField_pField(interpolated_e_field_csv, polarizability_csv)

        fourier(polarizability_csv)
        logging.info("Simulation completed successfully.")
    except Exception as err:
        logging.exception(f"Simulation aborted due to an error: {err}")
