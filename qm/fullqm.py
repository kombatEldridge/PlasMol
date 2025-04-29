import argparse
import os
import sys
import csv
import pandas as pd
import numpy as np
import math
import logging
import matplotlib.pyplot as plt
from io import StringIO

from pyscf import gto
from scipy.linalg import expm
from bohr_internals import input_parser
from bohr_internals import wavefunction

C_NM_FS = 299.792458  # speed of light in nm/fs


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
    parser.add_argument('-dt', type=float, default=1e-12, help="(Optional) The time step used in the simulation in fs.")
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


# electric_field_generator.py
class ElectricFieldGenerator:
    """
    Generates electric field components over time using a Gaussian-enveloped oscillatory function.
    """
    def __init__(self, wavelength, peak_time, width):
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
        complex_field = np.exp(1j * 2 * np.pi * self.frequency * (t - self.peak_time)) * \
                        np.exp(-self.width * (t - self.peak_time) ** 2)
        
        # Extract the active component and apply threshold
        active_component = np.real(complex_field)
        active_component[abs(active_component) < 1e-20] = 0
        
        # Define a better smoothing window at the start - use sine squared ramp
        ramp_duration = 15  # Longer ramp in fs
        t_start = t[0]  # Assume t is sorted and starts at the beginning
        window = np.ones_like(t)
        mask = t < (t_start + ramp_duration)
        window[mask] = np.sin(np.pi * (t[mask] - t_start) / ramp_duration / 2) ** 2
        
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


    # magnus4.py


# magnus4.py
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
    # Check if field is zero
    if (np.abs(field.get_exc_t()) < 1e-12).all() and \
       (np.abs(field.get_exc_t_minus_dt()) < 1e-12).all() and \
       (np.abs(field.get_exc_t_minus_2dt()) < 1e-12).all():
        # For zero field, density should be static 
        logging.debug("Zero field detected, skipping propagation")
        molecule.set_D_ao_t_minus_dt(molecule.get_D_ao_t())
        return molecule.get_D_ao_t()

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
        if not np.allclose(Hm4_ao, np.conjugate(Hm4_ao.T), atol=1e-10):
            logging.warning("Hm4_ao is not Hermitian!")


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
                if not np.isclose(np.trace(D_ao_t_plus_dt), np.trace(D_ao_t), atol=1e-10):
                    logging.warning(f"Trace not conserved! Difference: {np.trace(D_ao_t_plus_dt) - np.trace(D_ao_t)}")

                # Success! Now save state and get out.
                logging.debug(f"Predictor-Corrector scheme finished in {limit} iterations.")
                molecule.set_D_ao_t_minus_dt(D_ao_t)
                molecule.set_D_ao_t(D_ao_t_plus_dt)
                return D_ao_t_plus_dt
            
        # If they are too different, we run the process again
        D_ao_t_plus_dt_guess = D_ao_t_plus_dt


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
        F_mo_0 = self.wfn.S @ self.wfn.C[0] @ F_ao_0 @ self.wfn.C[0].T @ self.wfn.S
        self.matrix_store['F_mo_0'] = F_mo_0

        self.D_ao_0 = self.wfn.D[0]
        self.D_mo_0 = self.wfn.C[0].T @ self.wfn.S @ self.D_ao_0 @ self.wfn.S @ self.wfn.C[0]
        trace = np.trace(self.D_mo_0)
        n = self.wfn.nel[0]
        if not np.isclose(trace, n):
            raise ValueError(f"Trace of the matrix is not {n} (instead {trace}).")
        self.matrix_store['D_mo_0'] = self.D_mo_0
        self.matrix_store['D_ao_0'] = self.D_ao_0

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

    def get_D_ao_0(self):
        """
        Returns the initial density matrix.

        Returns:
            np.ndarray: Initial density matrix.
        """
        return self.matrix_store['D_ao_0']
    
    # mu.py


# mu.py
def calculate_ind_dipole(propagate_density_matrix, dt, molecule, field):
    output = np.zeros((3, 3))

    for j in [0, 1, 2]: 
        D_ao_t_plus_dt = propagate_density_matrix(dt, molecule, field)

        for i in [0, 1, 2]:
            # Repisky2015.pdf Eq. 22
            output[i][j] = 2 * float((np.trace(molecule.wfn.mu[i] @ D_ao_t_plus_dt) - np.trace(molecule.wfn.mu[i] @ molecule.wfn.D[0])).real)

    return output


def run(dt, molecule, field):
    volume = get_volume(molecule.molecule["coords"])
    induced_dipole_matrix = calculate_ind_dipole(propagate_density_matrix, dt, molecule, field) / volume
    collapsed_output = np.sum(induced_dipole_matrix, axis=1)

    # Should be [p_x, p_y, p_z] where p is the dipole moment in au
    return collapsed_output


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
        wavelength = 250    # In nanometers
        peak_time = 25      # time of pulse peak in fs
        width = 0.05        # width parameter in fs^-2
        t_start = 0         # start time in fs
        t_end = 50.0        # end time in fs
        dt_fs = args.dt     # time step in fs

        # Create time array
        time_values = np.arange(t_start, t_end + dt_fs, dt_fs)

        # Initialize the electric field generator
        field_generator = ElectricFieldGenerator(wavelength, peak_time, width)

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
            writer = csv.writer(csvfile)
            for t, field in zip(interpolated_times, interpolated_fields):
                writer.writerow([t, field[0], field[1], field[2]])
    
        # Initialize Field object
        field = FIELD()

        # Initialize CSV file for the polarizability field output
        polarizability_csv = "magnus-P-Field.csv"
        initCSV(polarizability_csv, "Molecule's Polarizability Field measured in au.")
    
        # Log non-comment lines from the Bohr input file
        bohr_input_path = os.path.abspath(args.bohr)
        with open(bohr_input_path, 'r') as bohr_file:
            for line in bohr_file:
                if not line.strip().startswith(('#', '--', '%')):
                    logger.info('\t%s', line.rstrip('\n'))
        
        for index, current_time in enumerate(interpolated_times):
            if index >= 2: field.set_exc_t_minus_2dt(interpolated_fields[index-2])
            if index >= 1: field.set_exc_t_minus_dt(interpolated_fields[index-1])
            field.set_exc_t(interpolated_fields[index])

            mu_arr = run(dt, molecule, field)
            logging.debug(f"At {current_time} fs, combined Bohr output is {mu_arr} in au")
            updateCSV(polarizability_csv, current_time, *mu_arr)

        # Plot the results using the interpolated electric field data
        show_eField_pField(interpolated_e_field_csv, polarizability_csv)
        logging.info("Simulation completed successfully.")
    except Exception as err:
        logging.exception(f"Simulation aborted due to an error: {err}")