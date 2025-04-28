# molecule.py
import logging
import numpy as np
from pyscf import gto
from bohr_internals import input_parser
from bohr_internals import wavefunction

logger = logging.getLogger("main")

class MOLECULE():
    """
    Represents a molecule and its electronic structure, initializing the molecule using PySCF.

    Attributes:
        molecule (dict): Dictionary containing molecule parameters.
        method (dict): Dictionary with method-related options.
        matrix_store (dict): Dictionary storing matrices for each direction (x, y, z) and initial matrices.
        wfn (wavefunction.RKS): Wavefunction object computed from the molecule.
    """
    def __init__(self, inputfile):
        """
        Initializes the molecule from an input file.

        Parameters:
            inputfile (str): Path to the input file containing molecule data.
        """
        from bohr_internals import options
        options = options.OPTIONS()
        self.molecule, self.method, basis = input_parser.read_input(inputfile, options)
        options.molecule = self.molecule

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

        self.propagator = self.method["propagator"]

        self.F_ao_0 = self.wfn.F[0]
        if not self.is_hermitian(self.F_ao_0, tol=1e-12):
            raise ValueError("Initial fock matrix in AO is not Hermitian")

        self.F_mo_0 = self.transform_F_ao_to_F_mo(self.F_ao_0)
        if not self.is_hermitian(self.F_mo_0, tol=1e-12):
            raise ValueError("Initial fock matrix in MO is not Hermitian")

        self.D_ao_0 = self.wfn.D[0]
        if not self.is_hermitian(self.D_ao_0, tol=1e-12):
            raise ValueError("Initial density matrix in AO is not Hermitian")

        self.D_mo_0 = self.wfn.C[0].T @ self.wfn.S @ self.D_ao_0 @ self.wfn.S @ self.wfn.C[0]
        if not self.is_hermitian(self.D_mo_0, tol=1e-12):
            raise ValueError("Initial density matrix in MO is not Hermitian")

        trace = np.trace(self.D_mo_0)
        n = self.wfn.nel[0]
        if not np.isclose(trace, n):
            raise ValueError(f"Trace of the matrix is not {n} (instead {trace}).")
        
        self.F_mo_tm12dt = self.F_mo_0
        self.F_mo_t = self.F_mo_0
        self.D_mo_t = self.D_mo_0

    def JK(self, D_ao):
        """
        Computes the effective potential and returns the Fock matrix component.

        Parameters:
            wfn: Wavefunction object containing molecular integrals.
            D_ao: Density matrix in atomic orbital basis.

        Returns:
            np.ndarray: The computed Fock matrix.
        """

        pot = self.wfn.jk.get_veff(self.wfn.ints_factory, 2 * D_ao)
        Fa = self.wfn.T + self.wfn.Vne + pot
        return Fa

    def tdfock(self, exc, D_ao):
        """
        Builds the Fock matrix by including the external field.

        Parameters:
            exc (array-like): External electric field components.
            D_ao (np.ndarray): Density matrix in atomic orbital basis.

        Returns:
            np.ndarray: The computed Fock matrix.
        """
        wfn = self.wfn
        
        logging.debug(f"Electric field at t + dt: {exc}")
        ext = sum(wfn.mu[dir] * exc[dir] for dir in range(3))
        logging.debug(f"Dipole interaction term: {np.linalg.norm(ext)}")
        F_ao = self.JK(D_ao) - ext
        return F_ao

    def transform_D_mo_to_D_ao(self, D_mo):
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
        C = self.wfn.C[0]  # MO coefficients

        D_ao = C @ D_mo @ C.T
        return D_ao

    def transform_F_ao_to_F_mo(self, F_ao):
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
        C = self.wfn.C[0]  # MO coefficients

        F_mo = C.T @ F_ao @ C
        return F_mo

    def is_hermitian(self, A, tol):
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

    def is_unitary(self, U, tol):
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
