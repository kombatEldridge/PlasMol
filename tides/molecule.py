# molecule.py
import logging
import numpy as np
from pyscf import gto, scf
from scipy.linalg import inv
from bohr_internals import input_parser
import wavefunction

logger = logging.getLogger("main")

class MOLECULE():
    """
    Represents a molecule and its electronic structure, initializing the molecule using PySCF.

    Attributes:
        molecule (dict): Dictionary containing molecule parameters.
        method (dict): Dictionary with method-related options.
        wfn (wavefunction.RKS): Wavefunction object computed from the molecule.
    """
    def __init__(self, inputfile, params):
        """
        Initializes the molecule from an input file.

        Parameters:
            inputfile (str): Path to the input file containing molecule data.
        """
        if params.chkfile:
            self.chkfile = params.chkfile_path
        else:
            self.chkfile = None

        from bohr_internals import options
        options = options.OPTIONS()
        self.molecule, self.method, basis = input_parser.read_input(inputfile, options)
        options.molecule = self.molecule

        # Format molecule string as required by PySCF
        atoms = self.molecule["atoms"]
        molecule_coords = ""
        for index, atom in enumerate(atoms):
            molecule_coords += " " + atom
            molecule_coords += " " + str(self.molecule["coords"][atom+str(index+1)][0])
            molecule_coords += " " + str(self.molecule["coords"][atom+str(index+1)][1])
            molecule_coords += " " + str(self.molecule["coords"][atom+str(index+1)][2])
            if index != (len(atoms)-1):
                molecule_coords += ";"

        # From TIDES
        mol = gto.M(atom=molecule_coords,
                          basis=basis["name"],
                          unit='B',
                          charge=int(options.charge),
                          spin=int(options.spin),
                          cart=options.cartesian)
        rhf = scf.RHF(mol)
        rhf.kernel()

        # charges = rhf.mol.atom_charges()
        # coords = rhf.mol.atom_coords()
        # nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
        # rhf.mol.set_common_orig_(nuc_charge_center)

        # Initialize matrices and wavefunction
        self.wfn = wavefunction.SCF(rhf, chkfile=self.chkfile)

        # NEED TO DO SOMETHING WITH THIS
        self.current_time = self.wfn.start_time

        # HERE RIGHT NOW
        self.propagator = self.method["propagator"]

        self.D_ao = self.wfn.D_ao_0
        if not self.is_hermitian(self.D_ao, tol=1e-12):
            raise ValueError("Initial density matrix in AO is not Hermitian")

        self.F_orth = self.get_F_orth(self.D_ao)
        self.F_orth_n12dt = self.F_orth

    def get_F_orth(self, D_ao, exc=None):
        F_ao = self.wfn._scf.get_fock(dm=D_ao).astype(np.complex128)
        if exc is not None:
            F_ao += self.calculate_potential(exc)
        return np.matmul(self.wfn.X.conj().T, np.matmul(F_ao, self.wfn.X))

    def rotate_coeff_to_orth(self, coeff_ao):
        return np.matmul(inv(self.wfn.X), coeff_ao)

    def rotate_coeff_away_from_orth(self, coeff_orth):
        return np.matmul(self.wfn.X, coeff_orth)
    
    def calculate_mu(self):
        mol = self.wfn._scf.mol
        charges = mol.atom_charges()
        coords = mol.atom_coords()
        nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
        mol.set_common_orig_(nuc_charge_center)
        mu = -1 * mol.intor('int1e_r', comp=3)
        return mu

    def calculate_potential(self, exc):
        mu = self.calculate_mu()
        return -1 * np.einsum('xij,x->ij', mu, exc)



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
