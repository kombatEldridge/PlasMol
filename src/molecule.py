# molecule.py
import os
import sys
import logging
import numpy as np
from pyscf import gto, dft
from pyscf.scf import addons

import molecule_parser
from chkfile import restart_from_chkfile

logger = logging.getLogger("main")

class MOLECULE():
    """
    Represents a molecule in the RT-TDDFT simulation.

    Manages quantum mechanical properties, SCF calculations, and time propagation state.
    """
    def __init__(self, params):
        """
        Initialize the MOLECULE object with input file and parameters.

        Sets up the molecule from the input file, performs initial SCF calculation,
        and loads from checkpoint if available.

        Parameters:
        params : object
            Parameters object with simulation settings.

        Returns:
        None
        """
        # Format molecule string as required by PySCF
        mol = gto.M(atom=params.molecule_coords,
                    basis=params.basis,
                    unit='B',
                    charge=int(params.charge),
                    spin=int(params.spin))
        self.mf = dft.RKS(mol)
        self.mf.xc = params.xc
        self.mf.kernel()

        charges = self.mf.mol.atom_charges()
        coords = self.mf.mol.atom_coords()
        nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
        self.mf.mol.set_common_orig_(nuc_charge_center)

        # Initialize matrices and wavefunction
        self.S = self.mf.get_ovlp()
        self.X = addons.canonical_orth_(self.S)

        if not self.is_hermitian(np.dot(self.X.conj().T, self.X), tol=params.check_tolerance):
            logger.warning("Orthogonalization matrix X may not be unitary")
            
        self.occ = self.mf.get_occ()
        self.D_ao_0 = self.mf.make_rdm1(mo_occ=self.occ)

        if len(np.shape(self.D_ao_0)) == 3:
            self.nmat = 2
            sys.exit("nmat == 2")
        else:
            self.nmat = 1

        self.current_time = 0

        if params.chkfile_path is not None and os.path.exists(params.chkfile_path):
            restart_from_chkfile(self)
            self.D_ao = self.mf.make_rdm1(mo_occ=self.occ)
            self.F_orth = self.get_F_orth(self.D_ao) # Should this include exc? at what time?
        else: 
            self.D_ao = self.D_ao_0
            self.F_orth = self.get_F_orth(self.D_ao)
            self.F_orth_n12dt = self.F_orth

        if not self.is_hermitian(self.D_ao, tol=1e-12):
            raise ValueError("Initial density matrix in AO is not Hermitian")


    def get_F_orth(self, D_ao, exc=None):
        """
        Compute the Fock matrix in the orthogonal basis.

        Includes the effect of an external field if provided.

        Parameters:
        D_ao : np.ndarray
            Density matrix in atomic orbital basis.
        exc : np.ndarray, optional
            External electric field components [x, y, z].

        Returns:
        np.ndarray
            Fock matrix in orthogonal basis.
        """
        F_ao = self.mf.get_fock(dm=D_ao).astype(np.complex128)
        if exc is not None:
            F_ao += self.calculate_potential(exc)
        return np.matmul(self.X.conj().T, np.matmul(F_ao, self.X))

    def rotate_coeff_to_orth(self, coeff_ao):
        """
        Transform molecular orbital coefficients to the orthogonal basis.

        Parameters:
        coeff_ao : np.ndarray
            Coefficients in atomic orbital basis.

        Returns:
        np.ndarray
            Coefficients in orthogonal basis.
        """
        return np.matmul(np.linalg.inv(self.X), coeff_ao)

    def rotate_coeff_away_from_orth(self, coeff_orth):
        """
        Transform molecular orbital coefficients from orthogonal to atomic orbital basis.

        Parameters:
        coeff_orth : np.ndarray
            Coefficients in orthogonal basis.

        Returns:
        np.ndarray
            Coefficients in atomic orbital basis.
        """
        return np.matmul(self.X, coeff_orth)
    
    def calculate_mu(self):
        """
        Calculate the dipole moment integrals for the molecule.

        Sets the origin to the nuclear charge center and computes dipole integrals.

        Parameters:
        None

        Returns:
        np.ndarray
            Dipole moment integrals with shape (3, nao, nao) for x, y, z components.
        """
        mu = -1 * self.mf.mol.intor('int1e_r', comp=3)
        return mu

    def calculate_potential(self, exc):
        """
        Calculate the potential contribution from an external electric field.

        Uses dipole moment integrals to compute the field-induced potential.

        Parameters:
        exc : np.ndarray
            External electric field components [x, y, z] in atomic units.

        Returns:
        np.ndarray
            Potential matrix in atomic orbital basis.
        """
        mu = self.calculate_mu()
        return -1 * np.einsum('xij,x->ij', mu, exc)

    def is_hermitian(self, A, tol):
        """
        Check if a matrix is Hermitian within a tolerance.

        Parameters:
        A : np.ndarray
            Matrix to check.
        tol : float
            Numerical tolerance for Hermitian property.

        Returns:
        bool
            True if the matrix is Hermitian within tolerance, False otherwise.
        """
        return np.allclose(A, A.conj().T, rtol=0, atol=tol)
