# molecule.py
import os
import sys
import logging
import numpy as np
from pyscf import gto, scf
from pyscf.scf import addons
from scipy.linalg import inv

import input_parser
from chkfile import restart_from_chkfile

logger = logging.getLogger("main")

class MOLECULE():
    """
    Represents a molecule and its electronic structure, initializing the molecule using PySCF.

    Attributes:
        molecule (dict): Dictionary containing molecule parameters.
        method (dict): Dictionary with method-related options.
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

        import options
        options = options.OPTIONS()
        self.molecule, self.method, basis = input_parser.read_input(inputfile, options)
        options.molecule = self.molecule

        self.propagator = self.method["propagator"]

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
        self.scf = scf.RHF(mol)
        self.scf.kernel()

        # Initialize matrices and wavefunction
        self.S = self.scf.get_ovlp()
        self.X = addons.canonical_orth_(self.S)
        self.occ = self.scf.get_occ()
        self.D_ao_0 = self.scf.make_rdm1(mo_occ=self.occ)

        if len(np.shape(self.D_ao_0)) == 3:
            self.nmat = 2
            sys.exit("nmat == 2")
        else:
            self.nmat = 1

        self.current_time = 0

        if self.chkfile is not None and os.path.exists(self.chkfile):
            restart_from_chkfile(self)
            self.D_ao = self.scf.make_rdm1(mo_occ=self.occ)
            self.F_orth = self.get_F_orth(self.D_ao)
        else: 
            self.D_ao = self.D_ao_0
            self.F_orth = self.get_F_orth(self.D_ao)
            self.F_orth_n12dt = self.F_orth

        if not self.is_hermitian(self.D_ao, tol=1e-12):
            raise ValueError("Initial density matrix in AO is not Hermitian")


    def get_F_orth(self, D_ao, exc=None):
        F_ao = self.scf.get_fock(dm=D_ao).astype(np.complex128)
        if exc is not None:
            F_ao += self.calculate_potential(exc)
        return np.matmul(self.X.conj().T, np.matmul(F_ao, self.X))

    def rotate_coeff_to_orth(self, coeff_ao):
        return np.matmul(inv(self.X), coeff_ao)

    def rotate_coeff_away_from_orth(self, coeff_orth):
        return np.matmul(self.X, coeff_orth)
    
    def calculate_mu(self):
        mol = self.scf.mol
        charges = mol.atom_charges()
        coords = mol.atom_coords()
        nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
        mol.set_common_orig_(nuc_charge_center)
        mu = -1 * mol.intor('int1e_r', comp=3)
        return mu

    def calculate_potential(self, exc):
        mu = self.calculate_mu()
        return -1 * np.einsum('xij,x->ij', mu, exc)

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
