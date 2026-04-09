# quantum/molecule.py
import os
import sys
import math
import logging
import numpy as np
import scipy.linalg
import scipy.optimize as opt
from pyscf import gto, dft
from pyscf import scf, tdscf, lib
from pyscf.scf import addons
from pyscf.dft import libxc
from scipy.interpolate import interp1d
from scipy.linalg import inv

from plasmol import constants

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
        # set all key values that are in params as key values for self
        for key, value in params.__dict__.items():
            setattr(self, key, value)

        self.mol = gto.M(atom=self.molecule_coords,
                    basis=self.molecule_basis,
                    unit='B',
                    charge=self.molecule_charge,
                    spin=self.molecule_spin)
        self.mol.verbose = 0
        self.mf = dft.RKS(self.mol)
        self.mf.xc = self.molecule_xc
        if hasattr(self, 'molecule_lrc_parameter'):
            self.mf.omega = self.molecule_lrc_parameter
        self.mf.kernel()

        logger.debug(f"Number of Occupied MOs: {np.sum(self.mf.mo_occ > 0)}")
        logger.debug(f"E_HOMO energy: {np.round(self.mf.mo_energy[self.mf.mo_occ > 0][-1], 6)} Ha, {np.round(self.mf.mo_energy[self.mf.mo_occ > 0][-1] * 27.2114, 6)} eV")

        charges = self.mf.mol.atom_charges()
        coords = self.mf.mol.atom_coords()
        nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
        self.mf.mol.set_common_orig_(nuc_charge_center)

        # Initialize matrices and wavefunction
        self.S = self.mf.get_ovlp()
        self.X = addons.canonical_orth_(self.S)

        if not self.is_hermitian(np.dot(self.X.conj().T, self.X), tol=self.molecule_hermiticity_tolerance):
            logger.warning("Orthogonalization matrix X may not be unitary")
        
        self.occ = self.mf.get_occ()
        
        skip_checkpoint = False
        if self.resumed_from_checkpoint:
            dir_component = getattr(params, 'molecule_source_dict', {}).get('component') if self.has_fourier else None
            if dir_component in params.not_checkpointed_dirs:
                skip_checkpoint = True

        if self.resumed_from_checkpoint and not skip_checkpoint:
            suffix = f"_{dir_component}" if self.has_fourier and dir_component else ""
            self.D_ao_0 = self.checkpoint_dict[f"D_ao_0{suffix}"]
            self.mf.mo_coeff = self.checkpoint_dict[f"mo_coeff{suffix}"]
            self.D_ao = self.mf.make_rdm1(mo_occ=self.occ)
            self.F_orth = self.get_F_orth(self.D_ao) # Should this include exc? at what time?
            if self.molecule_propagator_str == 'magnus2':
                self.F_orth_n12dt = self.checkpoint_dict[f"F_orth_n12dt{suffix}"]
            elif self.molecule_propagator_str == 'step':
                self.C_orth_ndt = self.checkpoint_dict[f"C_orth_ndt{suffix}"]
        else:
            self.D_ao_0 = self.mf.make_rdm1(mo_occ=self.occ)
            self.D_ao = self.D_ao_0
            self.F_orth = self.get_F_orth(self.D_ao)
            if self.molecule_propagator_str == 'magnus2':
                self.F_orth_n12dt = self.F_orth
            elif self.molecule_propagator_str == 'step':
                self.C_orth_ndt = self.rotate_coeff_to_orth(self.mf.mo_coeff)

        # TODO: Add open shell support
        if len(np.shape(self.D_ao_0)) == 3:
            self.nmat = 2
            sys.exit("nmat == 2")
        else:
            self.nmat = 1

        if not self.is_hermitian(self.D_ao, tol=1e-12):
            raise ValueError("Initial density matrix in AO is not Hermitian")
        
        self.volume = self.get_volume(self.molecule_atoms)

    # # TODO: Test extensively
    # def xc_tuning(self, tol=1e-3):
    #     """Tune mu for LC functionals to minimize |E_cat - E_neut + ε_HOMO|."""

    # # TODO: Test extensively
    # def compute_vacuum_level(self, nstates=20, diffuse_basis='d-aug-cc-pvqz'):
    #     """Estimate ε_0 by interpolating KS ε vs. EA_k (paper eq 12-13)."""


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
        if self.has_broadening:
            if not hasattr(self, 'Gamma_ao_0'):
                self.Gamma_ao_0 = self.get_gamma_ao(**self.broadening_dict)
            if self.broadening_type == 'static':
                F_ao -= 1j * self.Gamma_ao_0
            elif self.broadening_type == 'dynamic':
                F_ao -= 1j * self.get_gamma_ao(**self.broadening_dict, D_ao=D_ao)
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

    def get_volume(self, coords):
        volume = 0
        for atom in coords:
            element = atom[0]
            if element in constants.vdw_radii:
                radius = constants.vdw_radii[element]
                volume += (4 / 3) * math.pi * (radius ** 3)
            else:
                print(f"Warning: No van der Waals radius for {element}")
            
        # Dividing by 0.14818471 Å³ will set the volume to atomic units.
        return volume / constants.V_AU_AA3
    
    # ------------------------------------ #
    #              Additional              #
    #      measurables can be defined      #
    #    here and added mid-simulation     #
    #      in quantum/propagation.py       #
    # ------------------------------------ #

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

    def get_gamma_ao(self, gam0, xi, eps0, clamp, D_ao=None):
        """
        Construct a diagonal damping matrix (Gamma(t) in the AO basis)
        according to Lopata2013: https://doi.org/10.1021/ct400569s

        Parameters:
        None

        Returns:
        np.ndarray
            Gamma(t) matrix.
        """
        if D_ao is None:
            D_ao = self.D_ao_0
            
        F_ao = self.mf.get_fock(dm=D_ao).astype(np.complex128)
        F_orth = self.X.conj().T @ F_ao @ self.X
        eps, C_prime = np.linalg.eigh(F_orth)

        M = len(eps)
        gamma = np.zeros(M, dtype=float)
        for i in range(M):
            e_tilde = eps[i] - eps0
            if e_tilde > 0:
                gamma_i = gam0 * (np.exp(xi * e_tilde) - 1.0)
                gamma[i] = min(clamp, gamma_i)

        Lambda = np.diag(gamma)
        Gamma_orth = C_prime @ Lambda @ C_prime.conj().T
        # Gamma_orth = self.X.conj().T @ G_ao @ self.X 
        # ==> inv(self.X.conj().T) @ Gamma_orth = G_ao @ self.X 
        # ==> inv(self.X.conj().T) @ Gamma_orth @ inv(self.X) = G_ao

        # TODO: double check that inv(self.X.T).T == inv(self.X)
        Gamma_ao = inv(self.X.conj().T) @ Gamma_orth @ inv(self.X)
        return Gamma_ao
    