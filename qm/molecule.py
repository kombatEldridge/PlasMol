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