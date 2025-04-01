# cli.py
import argparse

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
    parser.add_argument('-e', '--csv', type=str, help="Path to the electric field CSV file.")
    parser.add_argument('-m', '--mult', type=float, default=1, help="(Optional) Multiplier for the electric field interpolator resolution.")
    parser.add_argument('-l', '--log', help="(Optional) Log file name")
    parser.add_argument('-v', '--verbose', action='count', default=0, help="(Optional) Increase verbosity.")
    parser.add_argument('-i', '--pcconv', type=float, default=1e-12, help="(Optional) Iteration convergence for Predictor-Corrector scheme in Magnus propagator.")
    
    args = parser.parse_args()
    if args.log and args.verbose == 0:
        args.verbose = 1

    return args

# plotting.py
import matplotlib.pyplot as plt
import pandas as pd
import logging

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

# csv_utils.py
import os
import csv
import pandas as pd

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
            if euclidean_norm_difference(D_mo_t_plus_dt, D_mo_t_plus_dt_guess) < molecule.pcconv:
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

# volume.py
import math

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

# interpolation.py
import numpy as np
from scipy.interpolate import interp1d

class ElectricFieldInterpolator:
    """
    Interpolates electric field components over time using cubic interpolation.
    """
    def __init__(self, time_values, electric_x, electric_y, electric_z):
        """
        Initialize the interpolator with time and electric field data.

        Parameters:
            time_values (list or array-like): Time values.
            electric_x (list or array-like): Electric field component in x.
            electric_y (list or array-like): Electric field component in y.
            electric_z (list or array-like): Electric field component in z.
        """
        self.interp_x = interp1d(time_values, electric_x, kind='cubic', fill_value="extrapolate")
        self.interp_y = interp1d(time_values, electric_y, kind='cubic', fill_value="extrapolate")
        self.interp_z = interp1d(time_values, electric_z, kind='cubic', fill_value="extrapolate")

    def get_field_at(self, query_times):
        """
        Returns the interpolated electric field components at the specified times.

        Parameters:
            query_times (array-like): Time values to interpolate.

        Returns:
            np.ndarray: A 2D array with columns [x, y, z] representing the interpolated field.
        """
        x_interp = self.interp_x(query_times)
        y_interp = self.interp_y(query_times)
        z_interp = self.interp_z(query_times)
        return np.column_stack((x_interp, y_interp, z_interp))

# logging_utils.py
import logging

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


# bohr_internals/wavefunction
from pyscf import dft,solvent,tools
import numpy as np
import scipy as sp

from . import diis_routine
from . import io_utils
from . import constants

class RKS():
    def __init__(self,mol):
      self.ints_factory   = mol
      self.nbf            = mol.nao 
      self.nel            = mol.nelec #returns [nalpha, nbeta]
      self.mult           = mol.multiplicity 
      self.charge         = mol.charge 
      self.nbf            = mol.nao
      self.e_nuc          = mol.energy_nuc()
      self.reference      = "rks"
      self.molden_reorder = tools.molden.order_ao_index(self.ints_factory)
      self.triplet        = False

    def compute(self,options):
      print("", end ="") # print("    Alpha electrons  : %4i"%(self.nel[0]),flush=True)
      print("", end ="") # print("    Beta  electrons  : %5i"%(self.nel[1]),flush=True)
    
      self.options = options

      self.S   = self.ints_factory.intor('int1e_ovlp') 
      self.T   = self.ints_factory.intor('int1e_kin')
      self.Vne = self.ints_factory.intor('int1e_nuc')
      self.mu  = self.ints_factory.intor('int1e_r')

      options.xctype  = dft.libxc.xc_type(self.options.xc)
      options.xcalpha = dft.libxc.hybrid_coeff(self.options.xc)
      print("", end ="") # print("\n    Exchange-Correlation Functional:",options.xc)
      print("", end ="") # print("\n    Exchange-Correlation Functional Type:",options.xctype)
      print("", end ="") # print("\n    Hybrid alpha parameter: %f"%options.xcalpha)
      #dft grid parameters
      self.jk    = dft.RKS(self.ints_factory,options.xc) #jk object from pyscf
      #self.jk.multiplicity=2

      eps_scaling = np.zeros((self.nbf,self.nbf))
      self.eps = np.zeros((self.nbf))

      # if self.options.relativistic == "zora":
      #   zora = relativistic.ZORA(self)
      #   zora.get_zora_correction()
      #   self.T =  zora.T[0]
      #   self.H_so = zora.H_so 
      #   eps_scaling = zora.eps_scal_ao
      #   if self.options.so_scf is True:
      #     exit("    CAN'T ADD H_SO TO RKS")

      if options.guess_mos_provided is True:
        print("", end ="") # print("    Reading MOs from File...")
        Cmo, nel = io_utils.read_real_mos(options.guess_mos)
        #sanity check
        if Cmo.shape != (2,self.nbf,self.nbf):
          exit("Incompatible MOs dimension")
        elif nel[0] != nel[1]:
          exit("Incompatible number of electrons, Na = %i, Nb = %i"%(nel[0],nel[1]))
        D = np.matmul(Cmo[0][:,:nel[0]],np.conjugate(Cmo[0].T[:nel[0],:]))
      else:
        self.jk.conv_tol	= options.e_conv
        self.jk.conv_tol_grad = options.d_conv
        self.jk.verbose = 4
        energy_pyscf = self.jk.kernel()
        Cmo = np.zeros((self.nbf,self.nbf))
        print("", end ="") # print("    Computing MOs from PySCF....")
        #D =  self.jk.init_guess_by_atom()
        C =  self.jk.mo_coeff
        D = C[:,:self.nel[0]]@C.T[:self.nel[0],:]

      F = self.T + self.Vne

      #orthogonalization matrix
      Shalf = sp.linalg.sqrtm(np.linalg.inv(self.S))
      Forth = np.matmul(Shalf,np.matmul(F,Shalf))
      newE  = np.einsum("mn,mn->",D,(F+F)) #+ Exc 
      newD  = np.zeros((self.nbf,self.nbf))

      if self.options.cosmo is True: 
        cosmo = solvent.ddcosmo.DDCOSMO(self.ints_factory)
        cosmo.eps = self.options.cosmo_epsilon 
        print("", end ="") # print("    COSMO solvent enabled. Dielectric constant %f"%(self.options.cosmo_epsilon))

      energy = 0.
      print("", end ="") # print("\n                   Total Energy    |Delta E|    RMS |[F,D]| ",flush=True)
      if options.diis is True:
        diis = diis_routine.DIIS(self)
        err_vec = np.zeros((1,1,1))
        Fdiis = np.zeros((1,1,1))
  
      for it in range(options.maxiter):
          ##THIS WORKS
          pot = self.jk.get_veff(self.ints_factory,2.*D) #pyscf effective potential
          if self.options.cosmo is True:
            e_cosmo, v_cosmo = cosmo.kernel(dm=2.*D)
            F = self.T + self.Vne + pot + v_cosmo #2.*J - 0*K + Vxc
          else:
            F = self.T + self.Vne + pot  #2.*J - 0*K + Vxc
          ###
          if options.diis is True:
            Faorth, e_rms = diis.do_diis(F,D,self.S,Shalf)
          else:
            diis = False
            Faorth = np.matmul(Shalf.T,np.matmul(F,Shalf))
          evals, evecs = sp.linalg.eigh(Faorth)
          C    = np.matmul(Shalf,evecs).real
          if options.noscf is True:
            print("", end ="") # print("    NOSCF flag enabled. Guess orbitals will be used without optimization")
            dE = options.e_conv/10.
            dD = options.d_conv/10.
            oneE  = 2.*np.trace(np.matmul(D,self.T+self.Vne)) 
            if self.options.cosmo is True:
              oneE  += e_cosmo 
            twoE  = pot.ecoul + pot.exc 
            newE  = oneE + twoE
            energy = newE
          else:
            newD = np.matmul(C[:self.nbf,:self.nel[0]],C.T[:self.nel[0],:self.nbf])

            if self.options.cosmo is True:
              oneE = 2. * np.trace(np.matmul((newD),(self.T+self.Vne))) + e_cosmo
            else:
              oneE = 2. * np.trace(np.matmul((newD),(self.T+self.Vne)))
            twoE = pot.ecoul + pot.exc #1. * np.trace(np.matmul(newD,(2.*J - 0.*K))) + Exc
            newE = oneE + twoE
            dE   = abs(newE - energy)
            if options.diis is True:
                dD = e_rms
            else:
                dD = abs(np.sqrt(np.trace((newD-D)**2)))
            energy = newE
            D = 1.*newD
            if it > 0:
                print("", end ="") # print("    @SCF iter %3i: % 12.8f % 10.6e % 10.6e %6s"%(it,energy+self.e_nuc,dE,dD,diis.is_diis),flush=True)

          if (dE < options.e_conv) and (dD < options.d_conv):
            # if self.options.relativistic == "zora":
            #   Ci = np.zeros((1,self.nbf),dtype=complex)
            #   zora_scal = 0.
            #   for i in range(self.nel[0]):
            #     Ci[0] = C.T[i]
            #     Di = np.matmul(np.conjugate(Ci.T),Ci).real
            #     eps_scal = np.trace(np.matmul(eps_scaling,Di))
            #     self.eps[i] = evals[i]/(1.+eps_scal)
            #     zora_scal -= eps_scal*self.eps[i]
             
            #   self.eps[self.nel[0]:] = evals[self.nel[0]:]
            # else:
            self.eps = evals  

            print("", end ="") # print("    Iterations have converged!")
            print("", end ="") # print("")
            print("", end ="") # print("    Nuclear Repulsion Energy:     %20.12f" %(self.e_nuc))
            print("", end ="") # print("    One-electron Energy:          %20.12f" %(oneE))
            print("", end ="") # print("    Two-electron Energy:          %20.12f" %(twoE))
            print("", end ="") # print("    Total  Energy:                %20.12f\n" %(twoE+oneE+self.e_nuc))
            print("", end ="") # print("")
            print("", end ="") # print("    Orbital Energies [Eh]")
            print("", end ="") # print("    ---------------------")
            print("", end ="") # print("")
            print("", end ="") # print("    Alpha occupied:")
            for i in range(self.nel[0]):
                print("", end ="") # print("    %4i: %12.5f "%(i+1,self.eps[i]),end="")
                if (i+1)%3 == 0: print("", end ="") # print("")
            print("", end ="") # print("")
            print("", end ="") # print("    Alpha Virtual:")
            for a in range(self.nbf-self.nel[0]):
                print("", end ="") # print("    %4i: %12.5f "%(self.nel[0]+a+1,self.eps[self.nel[0]+a]),end="")
                if (a+1)%3 == 0: print("", end ="") # print("")
            print("", end ="") # print("")
            break
          if (it == options.maxiter-1):
              print("", end ="") # print("SCF iterations did not converge!")
              exit(0)
          it += 1
      print("", end ="") # print("")

      print("", end ="") # print("    Molecular Orbital Analysis")
      print("", end ="") # print("    --------------------------\n")
  
      ao_labels = self.ints_factory.ao_labels()
      for p in range(self.nbf):
        print("", end ="") # print("    Vector %5i:    Occupation = %1i    Energy = %8.5f "%(p+1, (2 if p < self.nel[0] else 0),self.eps[p]))
        print("", end ="") # print("    ----------------------------------------------------------------------------")
        sort_idx = np.argsort(-np.abs(C.T[p]))
        for idx, i in enumerate(sort_idx[:6]):
          print("", end ="") # print("    %-12s: % 8.5f "%(ao_labels[i],C[i][p]),end="")
          if ((idx+1)%3 == 0): print("", end ="") # print("")
        print("", end ="") # print("\n")

      self.F = [F,F]
      self.C = [C,C]
      self.D = [D,D]

      self.eps = [self.eps,self.eps]
      self.scf_energy = twoE + oneE + self.e_nuc
      mos_filename = self.options.inputfile.split(".")[0]+".mos"
      io_utils.write_mos([C,C],self.nel,mos_filename)
      filename = self.options.inputfile.split(".")[0]
      io_utils.write_molden(self,filename)

      print("", end ="") # print("\n")
      print("", end ="") # print("    Mulliken Population Analysis (q_A = Z_A - Q_A)")
      print("", end ="") # print("    --------------------------------------------\n")
      Q = 2.*np.diag(D@self.S)
      natoms = int(ao_labels[-1].split()[0])+1
      qA = np.zeros(natoms)
      QA = np.zeros(natoms)
      ZA = np.zeros(natoms)
      A = [None]*natoms
      
      for p in range(self.nbf):
          atom_index = ao_labels[p].split()[0]
          atom_label = ao_labels[p].split()[1]
          ZA[int(atom_index)]  = constants.Z[atom_label.upper()]
          QA[int(atom_index)] += Q[p] 
          A[int(atom_index)] = atom_index + " " + atom_label
      for a in range(natoms):
          print("", end ="") # print("    %-12s: % 8.5f "%(A[a],ZA[a]-QA[a]))
      print("", end ="") # print("\n")
      print("", end ="") # print("\n    Energetics Summary")
      print("", end ="") # print("    ------------------\n")
      print("", end ="") # print("    Nuclear Repulsion Energy:     %20.12f" %(self.e_nuc))
      print("", end ="") # print("    One-electron Energy:          %20.12f" %(oneE))
      print("", end ="") # print("    Two-electron Energy:          %20.12f" %(twoE))
      print("", end ="") # print("    @RKS Final Energy:            %20.12f" %(self.scf_energy))

      #quick spin analysis 
      N = np.trace((D@self.S))
      self.S2 = N - np.trace((D@self.S)@(D@self.S))
      print("", end ="") # print("")
      print("", end ="") # print("    Computed <S2> : %12.8f "%(self.S2))
      print("", end ="") # print("    Expected <S2> : %12.8f "%(0.0))

      return self.scf_energy


# bohr_internals/input_parser
import os
import sys
import numpy as np
import bohr_internals.constants as constants

# I replaced all prints with blanks to not clog up runtime messages.

def read_input(inputfile,options):
    #read input file and separates it into three classes: molecule, method, and basis
    molecule = {}
    method = {}
    basis = {} 
    
    options.inputfile = inputfile
    #set default parameters
    method["e_conv"] = 1e-6 
    method["d_conv"] = 1e-6
    method["maxiter"] = 500 
    method["diis"] = True 
    method["zora"] = False 
    method["so_correction"] = False 
    method["so_scf"] = False 
    method["in_core"] = False 
    method["reference"] = "restricted" 
    method["occupied"] = []
    method["nroots"] = 5
    method["do_cis"] = False  
    method["do_tdhf"] = False  
    method["do_cpp"] = False  
    method["resplimit"] = 1e-20
    method["propagator"] = "rk4"

    molecule["charge"] = 0
    molecule["spin"] = 0
    molecule["units"] = "angstroms"
    molecule["nocom"] = False

    basis["n_radial"]  = 49
    basis["n_angular"] = 35


    read_mol = False
    count = 0
    atoms = []
    coords = {}
    f = open(inputfile,"r")
    for line in f:
        if "#" in line:
           continue
        if "end molecule" in line:
            read_mol = False
    
        if (len(line.split()) > 1) and (read_mol is True):
            atoms.append(line.split()[0])
            coords[str(atoms[count])+str(count+1)] = \
            np.array((float(line.split()[1]),float(line.split()[2]),\
            float(line.split()[3])))
            count += 1
    
        if "start molecule" in line:
            read_mol = True
            count = 0
    
        if "basis" in line:
            if "library" in line:
              basis["name"] = os.path.abspath(os.path.dirname(__file__))+"/basis/"+str(line.split()[2])
            else: 
              basis["name"] = str(line.split()[1])

        if "n_radial" in line:
            basis["n_radial"] = int(line.split()[1])

        if "n_angular" in line:
            basis["n_angular"] = int(line.split()[1])
    
        if "method" in line:
            method["name"] = str(line.split()[1])
            options.method = str(line.split()[1])
    
        if "reference" in line:
            method["reference"] = str(line.split()[1])

        if "xc" in line:
            if len(line.split()) == 3: #read exchange and correlation separately
              method["xc"] = str(line.split()[1])+","+str(line.split()[2])
              options.xc = str(line.split()[1])+","+str(line.split()[2])
            elif len(line.split()) == 2: #read alias to xc functional
              options.xc = str(line.split()[1]) #+","
        if "e_conv" in line:
            method["e_conv"] = float(line.split()[1])
            options.e_conv   = float(line.split()[1])

        if "d_conv" in line:
            method["d_conv"] = float(line.split()[1])
            options.d_conv   = float(line.split()[1])

        if "ft_gamma" in line:
            options.gamma   = float(line.split()[1])

        if "maxiter" in line:
            method["maxiter"] = int(line.split()[1])
            options.maxiter   = int(line.split()[1])

        if "nroots" in line:
            method["nroots"] = int(line.split()[1])
            options.nroots = int(line.split()[1])
        if "grid_level" in line:
            options.grid_level = int(line.split()[1])
        
        if "batch_size" in line:
            options.batch_size = int(line.split()[1])

        if "guess_mos" in line:
            options.guess_mos = str(line.split()[1])
            options.guess_mos_provided = True
        if "cosmo" in line:
            if len(line.split()) > 1:
              options.cosmo_epsilon = float(line.split()[1])
            options.cosmo = True

            

        if "occupied" in line:
            method["occupied"] = [int(line.split()[1]), int(line.split()[2])]
            options.occupied = [int(line.split()[1]), int(line.split()[2])]
            options.cvs = True

        if "virtual" in line:
            options.virtual = [int(line.split()[1]), int(line.split()[2])]
            options.reduced_virtual = True

        if "couple_states" in line:
            options.occupied1 = [int(line.split()[1]), int(line.split()[2])]
            options.occupied2 = [int(line.split()[3]), int(line.split()[4])]
            options.couple_states = True

        if "frequencies" in line:
            if "eV" in line:
              options.frequencies = np.asarray([float(line.split()[1]), float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])*constants.ev2au
            else: 
              options.frequencies = np.asarray([float(line.split()[1]), float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])

        if "so_scf" in line:
            method["so_scf"] = True
            options.so_scf = True

        if "akonly" in line:
            options.akonly = True
            options.nofxc = True
        if "fonly" in line:
            options.fonly = True
            options.nofxc = True
        if "jkonly" in line:
            options.jkonly = True
            options.nofxc = True
        if "jonly" in line:
            options.jonly = True
            options.nofxc = True
        if "nofxc" in line:
            options.nofxc = True

        if "cartesian" in line:
            options.cartesian = True
        if "spherical" in line:
            options.cartesian = False
        if "tddft_plus_tb" in line:
            options.plus_tb = True

        if "noscf" in line:
            method["noscf"] = True
            options.noscf = True

        if "tdscf_in_core" in line:
            options.tdscf_in_core = True

        if "direct_diagonalization" in line:
            options.direct_diagonalization = True

        if "in_core" in line:
            method["in_core"] = True

        if "do_tda" in line:
            method["do_tda"] = True
            options.do_tda = True

        if "do_cis" in line:
            method["do_cis"] = True
            options.do_cis = True

        if "do_tdhf" in line:
            method["do_tdhf"] = True
            options.do_tdhf = True

        if "do_cpp" in line:
            method["do_cpp"] = True
            options.do_cpp   = True

        if "charge" in line:
            molecule["charge"] = float(line.split()[1])
            options.charge = float(line.split()[1])

        if "spin" in line:
            molecule["spin"] = int(line.split()[1]) 
            options.spin = int(line.split()[1])

        if "mult" in line:
            options.mult = int(line.split()[1])
            molecule["spin"] = int(line.split()[1])-1 
            options.spin = int(line.split()[1])-1 
            

        if "units" in line:
            molecule["units"] = str(line.split()[1])
        
        if "nocom" in line:
            molecule["nocom"] = True

        if "sflip" in line:
            options.spin_flip = True

        if "swap_mos" in line:
           orig_str = line.split(",")[0].split()[1:]
           swap_str = line.split(",")[1].split()
           if len(orig_str) != len(swap_str):
             exit("Incorrect dimensions for orbital swap!")
           orig = [int(i)-1 for i in orig_str]
           swap = [int(i)-1 for i in swap_str]
  
           options.swap_mos = [orig,swap]

        if "diis" in line:
            if line.split()[1] == "true":
                method["diis"] = True
                options.diis   = True
            elif line.split()[1] == "false":
                method["diis"] = False
                options.diis   = False

        if "relativistic" in line:
            options.relativistic = line.split()[1]

        if "memory" in line:
            options.memory = float(line.split()[1])

        if "B_field_amplitude" in line:
            options.B_field_amplitude = float(line.split()[1])
        if "E_field_amplitude" in line:
            options.E_field_amplitude = float(line.split()[1])
        if "B_field_polarization" in line:
            options.B_field_polarization = int(line.split()[1])
        if "E_field_polarization" in line:
            options.E_field_polarization = int(line.split()[1])

        if "so_correction" in line:
            if line.split()[1] == "true":
                method["so_correction"] = True 

        if "roots_lookup_table" in line:
            low = int(line.split()[1])
            high = int(line.split()[2])+1
            options.roots_lookup_table = np.arange(low,high,1)

        if "resplimit" in line:
            method["resplimit"] = float(line.split()[1])

        if "propagator" in line:
            method["propagator"] = str(line.split()[1])

    molecule["atoms"] = atoms
    molecule["coords"] = get_coords(atoms,coords,molecule)

    #check/get multiplicity
    molecule["n_electrons"] = get_nelectrons(molecule["atoms"]) - molecule["charge"]
    molecule["e_nuclear"] = get_enuc(molecule)
    molecule["tb_gamma"] = get_tbgamma_w(molecule)[0]
    molecule["tb_w"] = get_tbgamma_w(molecule)[1]

    print_molecule_info(molecule)
    print_method_info(method)
    print_basis_info(basis)

    return molecule, method, basis

def get_coords(atoms,coords,mol):
    if (mol["units"] == "angstroms"):
        for Ai, A in enumerate(atoms):
            coords[A+str(Ai+1)] *= constants.angs2bohr

    #center of mass
    mass = np.zeros((len(atoms)))
    com  = np.zeros((3))
    for Ai, A in enumerate(atoms):
        xyz = coords[A+str(Ai+1)]
        mass[Ai] = constants.masses[A.upper()]
        com += xyz * mass[Ai] 

    mol["com"] = com
        
    if (mol["nocom"] is False):    
        for Ai, A in enumerate(atoms):
            coords[A+str(Ai+1)] -= com/np.sum(mass) 

     
    return coords    


def get_enuc(mol):
    atoms = mol["atoms"]
    coords = mol["coords"]
    natoms = len(atoms)
    enuc = 0.
    for a in range(natoms):
        Za = constants.Z[atoms[a].upper()]
        atom = atoms[a]+str(a+1)
        RA = np.array([coords[atom][0],coords[atom][1],coords[atom][2]])
        for b in range(0,a):
            Zb = constants.Z[atoms[b].upper()]
            atom = atoms[b]+str(b+1)
            RB = np.array([coords[atom][0],coords[atom][1],coords[atom][2]])
            RAB = np.linalg.norm(RA-RB)
            enuc += Za*Zb/RAB
    return enuc

def get_tbgamma_w(mol):
    atoms = mol["atoms"]
    coords = mol["coords"]
    natoms = len(atoms)
    enuc = 0.
    tbgamma = np.zeros((len(atoms),len(atoms)))
    tbw = np.zeros((len(atoms),len(atoms)))
    for a in range(natoms):
        try:
          eta_a = constants.eta[atoms[a].upper()]/27.21138 
        except:
          eta_a = 0.  
        try: 
          tbw_a = constants.tbw[atoms[a].upper()] 
        except:
          tbw_a = 0.  
        tbw[a][a] = tbw_a
        atom = atoms[a]+str(a+1)
        RA = np.array([coords[atom][0],coords[atom][1],coords[atom][2]])
        for b in range(0,a):
            try:
              eta_b = constants.eta[atoms[b].upper()]/27.21138
            except:
              eta_b = 0. 
            atom = atoms[b]+str(b+1)
            RB = np.array([coords[atom][0],coords[atom][1],coords[atom][2]])
            RAB = np.linalg.norm(RA-RB)
            if RAB == 0:
              S = 5.* eta_a/16. 
            elif (RAB != 0) and abs(eta_a-eta_b) < 1e-12:
              S = np.exp(-eta_a*RAB) * (48. + 33.*eta_a*RAB + 9.*eta_a**2 * RAB**2 + eta_a**3 * RAB**3)/(48.*RAB)
            else:
              S = np.exp(-eta_a*RAB)*(eta_b**4 * eta_a/(2.*(eta_a**2-eta_b**2)**2) - (eta_b**6 - 3.*eta_b**4 * eta_a**2)/((eta_a**2 - eta_b**2)**3 *RAB))
              S += np.exp(-eta_b*RAB)*(eta_a**4 * eta_b/(2.*(eta_b**2-eta_a**2)**2) - (eta_a**6 - 3.*eta_a**4 * eta_b**2)/((eta_b**2 - eta_a**2)**3 *RAB))
            tbgamma[a][b] += 1./RAB - S
    return tbgamma, tbw

def get_nelectrons(atoms):
    nel = 0
    for atom in atoms: 
        nel += constants.Z[atom.upper()]
    return nel

def print_molecule_info(mol):
    atoms = mol["atoms"]
    coords = mol["coords"]
    nel = mol["n_electrons"]
    mult = mol["spin"]+1
    charge = mol["charge"]
    natoms = len(atoms)
    print("", end ="") # print("    Molecule Info")
    print("", end ="") # print("    -------------")
    print("", end ="") # print("")
    print("", end ="") # print("    Number of atoms    : %i"%(natoms))
    print("", end ="") # print("    Number of electrons: %i"%(nel))
    print("", end ="") # print("    Charge             : %i"%(charge))
    print("", end ="") # print("    Multiplicity       : %i"%(mult))
    print("", end ="") # print("    Geometry [a0]:")
    for a in range(natoms):
        atom = atoms[a]+str(a+1)
        print("", end ="") # print("    %5s %20.12f %20.12f %20.12f "%(atoms[a],coords[atom][0],coords[atom][1],coords[atom][2]))
    print("", end ="") # print("")
    return None

def print_method_info(method):
    print("", end ="") # print("    Method: %s "%method["name"])
    print("", end ="") # print("    E_conv: %e "%method["e_conv"])
    print("", end ="") # print("    D_conv: %e "%method["d_conv"])
    print("", end ="") # print("    Maxiter: %i "%method["maxiter"])
    print("", end ="") # print("")
    return None
def print_basis_info(basis):
    print("", end ="") # print("    Basis set: %s "%basis["name"])
    return None


# molecule.py
import numpy as np
from pyscf import gto
from bohr_internals import input_parser
from bohr_internals import wavefunction

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
        self.matrix_store = {0: {}, 1: {}, 2: {}}
        self.wfn = wavefunction.RKS(pyscf_mol)
        rks_energy = self.wfn.compute(options)

        F_ao_0 = self.wfn.F[0]
        F_mo_0 = self.wfn.S @ self.wfn.C[0] @ F_ao_0 @ self.wfn.C[0].T @ self.wfn.S
        self.matrix_store['F_mo_0'] = F_mo_0

        D_ao_0 = self.wfn.D[0]
        self.D_mo_0 = self.wfn.C[0].T @ self.wfn.S @ D_ao_0 @ self.wfn.S @ self.wfn.C[0]
        trace = np.trace(self.D_mo_0)
        n = self.wfn.nel[0]
        if not np.isclose(trace, n):
            raise ValueError(f"Trace of the matrix is not {n} (instead {trace}).")
        self.matrix_store['D_mo_0'] = self.D_mo_0

    def get_F_mo_t(self, dir):
        """
        Gets the Fock matrix at time t for a given direction.

        Parameters:
            dir (int): Direction (0: x, 1: y, 2: z).

        Returns:
            np.ndarray: Fock matrix.
        """
        key = f'F_mo_t_{"xyz"[dir]}'
        return self.matrix_store[dir].get(key, self.get_F_mo_0())

    def set_F_mo_t(self, F_mo_t, dir):
        """
        Sets the Fock matrix at time t for a given direction.

        Parameters:
            F_mo_t (np.ndarray): Fock matrix.
            dir (int): Direction (0: x, 1: y, 2: z).
        """
        self.matrix_store[dir][f'F_mo_t_{"xyz"[dir]}'] = F_mo_t

    def get_F_mo_t_minus_half_dt(self, dir):
        """
        Gets the Fock matrix at time t - dt/2 for a given direction.

        Parameters:
            dir (int): Direction.

        Returns:
            np.ndarray: Fock matrix.
        """
        key = f'F_mo_t_minus_half_dt_{"xyz"[dir]}'
        return self.matrix_store[dir].get(key, self.get_F_mo_0())

    def set_F_mo_t_minus_half_dt(self, F_mo_t_minus_half_dt, dir):
        """
        Sets the Fock matrix at time t - dt/2 for a given direction.

        Parameters:
            F_mo_t_minus_half_dt (np.ndarray): Fock matrix.
            dir (int): Direction.
        """
        self.matrix_store[dir][f'F_mo_t_minus_half_dt_{"xyz"[dir]}'] = F_mo_t_minus_half_dt

    def get_F_mo_0(self):
        """
        Returns the initial Fock matrix in the molecular orbital basis.

        Returns:
            np.ndarray: Initial Fock matrix.
        """
        return self.matrix_store['F_mo_0']

    def get_D_mo_t(self, dir):
        """
        Gets the density matrix at time t for a given direction.

        Parameters:
            dir (int): Direction.

        Returns:
            np.ndarray: Density matrix.
        """
        key = f'D_mo_t_{"xyz"[dir]}'
        return self.matrix_store[dir].get(key, self.get_D_mo_0())

    def set_D_mo_t(self, D_mo_t, dir):
        """
        Sets the density matrix at time t for a given direction.

        Parameters:
            D_mo_t (np.ndarray): Density matrix.
            dir (int): Direction.
        """
        self.matrix_store[dir][f'D_mo_t_{"xyz"[dir]}'] = D_mo_t

    def get_D_mo_0(self):
        """
        Returns the initial density matrix.

        Returns:
            np.ndarray: Initial density matrix.
        """
        return self.matrix_store['D_mo_0']

# main.py
import os
import sys
import logging
import numpy as np
import meep as mp  # type: ignore
import threading
import queue

from molecule import MOLECULE
from logging_utils import PrintLogger
from interpolation import ElectricFieldInterpolator
from volume import get_volume
from magnus2 import propagate_direction_worker, combine_propagation_results
from csv_utils import initCSV, read_electric_field_csv
from plotting import show_eField_pField
from cli import parse_arguments

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
    
        mp.verbosity(min(3, args.verbose))
        sys.stdout = PrintLogger(logger, logging.INFO)
        logging.getLogger("h5py").setLevel(logging.INFO)
    
        logger.debug(f"Arguments given and parsed correctly: {args}")
        # Initialize the molecule and extract parameters
        molecule = MOLECULE(args.bohr, args.pcconv)
    
        # Read the original electric field CSV file
        time_values, electric_x, electric_y, electric_z = read_electric_field_csv(args.csv)
    
        # Create an interpolator for the electric field components
        logger.debug("Building an interpolation profile with ElectricFieldInterpolator")
        field_interpolator = ElectricFieldInterpolator(time_values, electric_x, electric_y, electric_z)
    
        # Generate a new time grid with adjusted resolution (args.mult is a multiplier)
        interpolated_times = np.linspace(0, time_values[-1], int(len(time_values) * args.mult))
        time_step = interpolated_times[1] - interpolated_times[0]
        interpolated_fields = field_interpolator.get_field_at(interpolated_times)
    
        # Initialize the interpolated electric field CSV file using initCSV
        interpolated_e_field_csv = "interpolated-E-Field.csv"
        initCSV(interpolated_e_field_csv, "Interpolated Electric Field measured in atomic units.")
    
        # Append the interpolated data rows to the CSV file
        with open(interpolated_e_field_csv, 'a', newline='') as csvfile:
            import csv
            writer = csv.writer(csvfile)
            for t, field in zip(interpolated_times, interpolated_fields):
                writer.writerow([t, field[0], field[1], field[2]])
    
        # Initialize CSV file for the polarizability field output
        polarizability_csv = "magnus-P-Field.csv"
        initCSV(polarizability_csv, "Molecule's Polarizability Field measured in atomic units.")
    
        # Log non-comment lines from the Bohr input file
        bohr_input_path = os.path.abspath(args.bohr)
        with open(bohr_input_path, 'r') as bohr_file:
            for line in bohr_file:
                if not line.strip().startswith(('#', '--', '%')):
                    logger.info('\t%s', line.rstrip('\n'))
    
        # Create a thread-safe queue for each spatial direction: 0 -> x, 1 -> y, 2 -> z.
        result_queues = {d: queue.Queue() for d in range(3)}
        threads = []
    
        # Launch one worker thread per direction.
        for d in range(3):
            t = threading.Thread(target=propagate_direction_worker,
                                 args=(None, d, interpolated_times,
                                       interpolated_fields, time_step, molecule, result_queues[d]),
                                 kwargs={'propagator': None})
            # We pass the propagate_density_matrix function via kwargs; here, we override the placeholder.
            # Alternatively, you can import it from propagation and pass it directly.
            from magnus2 import propagate_density_matrix
            t = threading.Thread(target=propagate_direction_worker,
                                 args=(propagate_density_matrix, d, interpolated_times,
                                       interpolated_fields, time_step, molecule, result_queues[d]))
            t.start()
            threads.append(t)
    
        volume = get_volume(molecule.molecule["coords"])
    
        # In the main thread, combine results as soon as a complete set is available.
        combine_propagation_results(interpolated_times, molecule, result_queues, polarizability_csv, volume)
    
        # Wait for all worker threads to finish.
        for t in threads:
            t.join()
    
        # Plot the results using the interpolated electric field data
        show_eField_pField(interpolated_e_field_csv, polarizability_csv)
        logging.info("Simulation completed successfully.")
    except Exception as err:
        logging.exception(f"Simulation aborted due to an error: {err}")