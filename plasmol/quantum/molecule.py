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
from scipy.optimize import minimize_scalar

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
        self.is_open_shell = (self.molecule_spin != 0)
        if self.is_open_shell:
            self.mf = dft.UKS(self.mol)
        else:
            self.mf = dft.RKS(self.mol)
        self.mf.xc = self.molecule_xc
        if hasattr(self, 'molecule_lrc_parameter'):
            if self.molecule_lrc_parameter == "tune":
                self.molecule_lrc_parameter = self.xc_tuning()
                logger.info(f"Optimal μ (ω) = {self.molecule_lrc_parameter:.6f}")
            self.mf.omega = self.molecule_lrc_parameter
        if self.has_broadening:
            if self.broadening_dict["eps0"] == "tune":
                eps0 = self.compute_vacuum_level()
                self.broadening_dict["eps0"] = eps0
                logger.info(f"Vacuum level ε₀ = {eps0:.6f} Ha")

        self.mf.kernel()
        self.nmat = 2 if self.is_open_shell else 1

        if self.is_open_shell:
            occ_a, occ_b = self.mf.mo_occ
            n_occ_a = int(np.sum(occ_a > 0))
            n_occ_b = int(np.sum(occ_b > 0))
            logger.debug(f"Number of Occupied MOs: α={n_occ_a}, β={n_occ_b}")
            if n_occ_a > 0:
                e_homo_a = self.mf.mo_energy[0][occ_a > 0][-1]
                logger.debug(f"E_HOMO(α) = {e_homo_a:.6f} Ha ({e_homo_a*27.2114:.6f} eV)")
            if n_occ_b > 0:
                e_homo_b = self.mf.mo_energy[1][occ_b > 0][-1]
                logger.debug(f"E_HOMO(β) = {e_homo_b:.6f} Ha ({e_homo_b*27.2114:.6f} eV)")
        else:
            logger.debug(f"Number of Occupied MOs: {np.sum(self.mf.mo_occ > 0)}")
            logger.debug(f"E_HOMO energy: {self.mf.mo_energy[self.mf.mo_occ > 0][-1]:.6f} Ha, "
                        f"{self.mf.mo_energy[self.mf.mo_occ > 0][-1]*27.2114:.6f} eV")

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

        if not self.is_hermitian(self.D_ao, tol=1e-12):
            raise ValueError("Initial density matrix in AO is not Hermitian")
        
        self.volume = self.get_volume(self.molecule_atoms)

    def xc_tuning(self):
        """
        Tune μ for LC functionals to minimize |E_cat - E_neut + ε_HOMO|.
        """
        mol_neutral = gto.M(atom=self.molecule_coords, basis=self.molecule_basis, charge=self.molecule_charge, spin=self.molecule_spin)
        mol_cation = gto.M(atom=self.molecule_coords, basis=self.molecule_basis, charge=self.molecule_charge + 1, spin=abs(self.molecule_spin - 1))

        mol_neutral.verbose = 0
        mol_cation.verbose = 0

        def get_homo(mf):
            """Return the highest occupied orbital energy (works for RKS and UKS)."""
            if isinstance(mf.mo_energy, list): 
                occ_a = mf.mo_occ[0] > 0.5
                homo_a = mf.mo_energy[0][occ_a][-1] if np.any(occ_a) else -np.inf
                occ_b = mf.mo_occ[1] > 0.5
                homo_b = mf.mo_energy[1][occ_b][-1] if np.any(occ_b) else -np.inf
                return max(homo_a, homo_b)
            else:
                occ = mf.mo_occ > 0.5
                return mf.mo_energy[occ][-1]

        def compute_J(omega):
            """Compute J(ω) = |IP_ΔSCF + ε_HOMO| (in Hartree)."""
            omega = float(omega)

            # Neutral
            if self.molecule_spin == 0:
                mf_n = dft.RKS(mol_neutral, xc=self.molecule_xc)
            else:
                mf_n = dft.UKS(mol_neutral, xc=self.molecule_xc)
            mf_n.omega = omega
            mf_n.conv_tol = 1e-7
            mf_n.max_cycle = 80
            mf_n.kernel()
            E_n = mf_n.e_tot
            eps_homo_n = get_homo(mf_n)

            # Cation
            if abs(self.molecule_spin - 1) == 0:
                mf_c = dft.RKS(mol_cation, xc=self.molecule_xc)
            else:
                mf_c = dft.UKS(mol_cation, xc=self.molecule_xc)
            mf_c.omega = omega
            mf_c.conv_tol = 1e-7
            mf_c.max_cycle = 80
            mf_c.kernel()
            E_c = mf_c.e_tot

            J = abs(E_c - E_n + eps_homo_n)
            return J

        logger.info("Tuning LC-ωPBE...\n")
        res = minimize_scalar(compute_J, bounds=(0.1, 0.6), method="bounded", tol=1e-6)
        logger.info(f"Optimal μ (ω) = {res.x:.6f}")
        return res.x

    def compute_vacuum_level(self):
        """
        Estimate ε_0.
        """
        mol_neutral = gto.M(atom=self.molecule_coords, basis=self.molecule_basis, charge=self.molecule_charge, spin=self.molecule_spin)
        mol_anion = gto.M(atom=self.molecule_coords, basis=self.molecule_basis, charge=self.molecule_charge - 1, spin=abs(self.molecule_spin - 1))

        mol_neutral.verbose = 0
        mol_anion.verbose = 0

        # 1. Ground-state calculations
        if self.molecule_spin == 0:
            mf_n = dft.RKS(mol_neutral, xc=self.molecule_xc)
        else:
            mf_n = dft.UKS(mol_neutral, xc=self.molecule_xc)
        mf_n.omega = self.molecule_lrc_parameter
        mf_n.kernel()
        E_n = mf_n.e_tot

        # Virtual orbital energies from neutral
        if self.molecule_spin == 0: 
            virt_energies = mf_n.mo_energy[mf_n.mo_occ < 0.5]
        else:
            virt_energies = mf_n.mo_energy[0][mf_n.mo_occ[0] < 0.5]

        # Anion: always UKS
        mf_a = dft.UKS(mol_anion, xc=self.molecule_xc)
        mf_a.omega = self.molecule_lrc_parameter
        mf_a.kernel()
        E_a = mf_a.e_tot

        EA1 = (E_a - E_n) * 27.2114

        # 2. LRTDDFT on the anion
        n_virt = len(virt_energies)
        n_states = max(20, n_virt - 1)

        td = tdscf.TDDFT(mf_a)
        td.conv_tol = 1e-6
        td.max_cycle = 100
        td.lindep = 1e-10
        td.kernel(nstates=n_states)
        nu = td.e * 27.2114

        # 3. Estimated EA for each virtual orbital
        estimated_EA = np.zeros_like(virt_energies)
        estimated_EA[0] = EA1
        for k in range(1, len(virt_energies)):
            if k - 1 < len(nu):
                estimated_EA[k] = EA1 + nu[k - 1]
            else:
                estimated_EA[k] = estimated_EA[k - 1] + 2.0

        # 4. Interpolate to find where EA = 0
        virt_eV = virt_energies * 27.2114
        f = interp1d(estimated_EA, virt_eV, kind='linear', fill_value='extrapolate')
        epsilon0_eV = f(0.0)
        epsilon0 = epsilon0_eV / 27.2114

        return epsilon0

    # def xc_tuning_Li2(self):
    #     """
    #     Tune mu for LC functionals to minimize |E_cat - E_neut + ε_HOMO|.
    #     I'll need to come back and try to make this more generizable but this should
    #     work for Li2 right now.
    #     """
    #     mol_neutral = gto.M(atom=self.molecule_coords, basis=self.molecule_basis, charge=self.molecule_charge,   spin=self.molecule_spin)  # Li2 (singlet)
    #     mol_cation  = gto.M(atom=self.molecule_coords, basis=self.molecule_basis, charge=self.molecule_charge+1, spin=self.molecule_spin+1)  # Li2+ (doublet)
    #     mol_neutral.verbose = 0
    #     mol_cation.verbose = 0

    #     def get_homo(mf):
    #         e = mf.mo_energy # orbital energy
    #         occ = mf.mo_occ # orbital occ number
    #         idx = np.where(occ > 0.5)[0][-1] # index of last orbital that is occupied
    #         homo = e[idx]
    #         return homo

    #     def compute_J(omega):
    #         """Compute J(ω) = |IP_ΔSCF + ε_HOMO| (in Hartree)."""
    #         omega = float(omega)

    #         # Neutral Li2 (RKS)
    #         mf_n = dft.RKS(mol_neutral, xc=self.molecule_xc)
    #         mf_n.omega = omega
    #         mf_n.conv_tol = 1e-7
    #         mf_n.max_cycle = 80
    #         mf_n.kernel()
    #         E_n = mf_n.e_tot
    #         eps_homo_n = get_homo(mf_n)

    #         # Cation Li2+ (UKS)
    #         mf_c = dft.UKS(mol_cation, xc=self.molecule_xc)
    #         mf_c.omega = omega
    #         mf_c.conv_tol = 1e-7
    #         mf_c.max_cycle = 80
    #         mf_c.kernel()
    #         E_c = mf_c.e_tot

    #         J = abs(E_c - E_n + eps_homo_n)
    #         return J

    #     logger.info("Tuning LC-ωPBE...\n")
    #     res = minimize_scalar(compute_J, bounds=(0.25, 0.55), method="bounded", tol=1e-6)
    #     return res.x

    # def xc_tuning_Na(self):
    #     """
    #     Tune mu for LC functionals to minimize |E_cat - E_neut + ε_HOMO|.
    #     I'll need to come back and try to make this more generizable but this should
    #     work for Na right now.
    #     """
    #     mol_neutral = gto.M(atom=self.molecule_coords, basis=self.molecule_basis, charge=self.molecule_charge,   spin=self.molecule_spin)  # Na (doublet)
    #     mol_cation  = gto.M(atom=self.molecule_coords, basis=self.molecule_basis, charge=self.molecule_charge+1, spin=self.molecule_spin-1)  # Na+ (singlet)
    #     mol_neutral.verbose = 0
    #     mol_cation.verbose = 0

    #     def get_homo(mf):
    #         e_a = mf.mo_energy[0] # α orbital energy
    #         occ_a = mf.mo_occ[0] # α orbital occ number
    #         idx_a = np.where(occ_a > 0.5)[0][-1] # index of last α orbital that is occupied
    #         homo_a = e_a[idx_a]

    #         e_b = mf.mo_energy[1] # β orbital energy
    #         occ_b = mf.mo_occ[1] # β orbital occ number
    #         idx_b = np.where(occ_b > 0.5)[0][-1] if np.any(occ_b > 0.5) else None # index of last β orbital that is occupied
    #         homo_b = e_b[idx_b] if idx_b is not None else -np.inf
    #         return max(homo_a, homo_b)

    #     def compute_J(omega):
    #         """Compute J(ω) = |IP_ΔSCF + ε_HOMO| (in Hartree)."""
    #         omega = float(omega)

    #         # Neutral Na (UKS)
    #         mf_n = dft.UKS(mol_neutral, xc=self.molecule_xc)
    #         mf_n.omega = omega
    #         mf_n.conv_tol = 1e-7
    #         mf_n.max_cycle = 80
    #         mf_n.kernel()
    #         E_n = mf_n.e_tot
    #         eps_homo_n = get_homo(mf_n)

    #         # Cation Na+ (RKS)
    #         mf_c = dft.RKS(mol_cation, xc=self.molecule_xc)
    #         mf_c.omega = omega
    #         mf_c.conv_tol = 1e-7
    #         mf_c.max_cycle = 80
    #         mf_c.kernel()
    #         E_c = mf_c.e_tot

    #         J = abs(E_c - E_n + eps_homo_n)
    #         return J

    #     logger.info("Tuning LC-ωPBE...\n")
    #     res = minimize_scalar(compute_J, bounds=(0.25, 0.55), method="bounded", tol=1e-6)
    #     return res.x

    # def compute_vacuum_level_Li2(self):
    #     """
    #     Estimate ε_0 for Li₂.
    #     Neutral Li₂ is closed-shell singlet → RKS
    #     Anion Li₂⁻ is doublet → UKS
    #     """
    #     mol_neutral = gto.M(atom=self.molecule_coords,
    #                         basis=self.molecule_basis,
    #                         charge=self.molecule_charge,
    #                         spin=self.molecule_spin)   # Li₂ (singlet)

    #     mol_anion   = gto.M(atom=self.molecule_coords,
    #                         basis=self.molecule_basis,
    #                         charge=self.molecule_charge-1,
    #                         spin=self.molecule_spin+1)  # Li₂⁻ (doublet)

    #     mol_neutral.verbose = 0
    #     mol_anion.verbose = 0

    #     # 1. Ground-state calculations
    #     # Neutral Li₂ → closed-shell singlet → RKS
    #     mf_n = dft.RKS(mol_neutral, xc=self.molecule_xc)
    #     mf_n.omega = self.molecule_lrc_parameter
    #     mf_n.kernel()
    #     E_n = mf_n.e_tot
    #     # For RKS the arrays are 1D
    #     virt_energies = mf_n.mo_energy[mf_n.mo_occ < 0.5]

    #     # Anion Li₂⁻ → doublet → UKS
    #     mf_a = dft.UKS(mol_anion, xc=self.molecule_xc)
    #     mf_a.omega = self.molecule_lrc_parameter
    #     mf_a.kernel()
    #     E_a = mf_a.e_tot

    #     EA1 = (E_a - E_n) * 27.2114

    #     # 2. LRTDDFT on the anion
    #     n_virt = len(virt_energies)
    #     n_states = max(20, n_virt - 1)

    #     td = tdscf.TDDFT(mf_a)          # works for UKS
    #     td.conv_tol = 1e-6
    #     td.max_cycle = 100
    #     td.lindep = 1e-10
    #     td.kernel(nstates=n_states)
    #     nu = td.e * 27.2114

    #     # 3. Estimated EA for each virtual orbital
    #     estimated_EA = np.zeros_like(virt_energies)
    #     estimated_EA[0] = EA1
    #     for k in range(1, len(virt_energies)):
    #         if k-1 < len(nu):
    #             estimated_EA[k] = EA1 + nu[k-1]
    #         else:
    #             estimated_EA[k] = estimated_EA[k-1] + 2.0   # safe fallback

    #     # 4. Interpolate to find where EA = 0
    #     virt_eV = virt_energies * 27.2114
    #     f = interp1d(estimated_EA, virt_eV, kind='linear', fill_value='extrapolate')
    #     epsilon0_eV = f(0.0)
    #     epsilon0 = epsilon0_eV / 27.2114

    #     return epsilon0

    # def compute_vacuum_level_Na(self):
    #     """
    #     Estimate ε_0.
    #     I'll need to come back and try to make this more generizable but this should
    #     work for Na right now.
    #     """
    #     mol_neutral = gto.M(atom=self.molecule_coords, basis=self.molecule_basis, charge=self.molecule_charge,   spin=self.molecule_spin)  # Na (doublet)
    #     mol_anion  = gto.M(atom=self.molecule_coords, basis=self.molecule_basis, charge=self.molecule_charge-1, spin=self.molecule_spin-1)  # Na- (singlet)
    #     mol_neutral.verbose = 0
    #     mol_anion.verbose = 0

    #     # 1. Ground-state calculations
    #     mf_n = dft.UKS(mol_neutral, xc=self.molecule_xc)
    #     mf_n.omega = self.molecule_lrc_parameter
    #     mf_n.kernel()
    #     E_n = mf_n.e_tot
    #     virt_energies = mf_n.mo_energy[0][mf_n.mo_occ[0] < 0.5]   # virtual orbital energies (alpha)

    #     mf_a = dft.RKS(mol_anion, xc=self.molecule_xc)
    #     mf_a.omega = self.molecule_lrc_parameter
    #     mf_a.kernel()
    #     E_a = mf_a.e_tot

    #     EA1 = (E_a - E_n) * 27.2114

    #     # 2. LRTDDFT on the anion
    #     n_virt = len(virt_energies)
    #     n_states = max(20, n_virt - 1)

    #     td = tdscf.TDDFT(mf_a)
    #     td.conv_tol = 1e-6
    #     td.max_cycle = 100
    #     td.lindep = 1e-10
    #     td.kernel(nstates=n_states)
    #     nu = td.e * 27.2114

    #     # 3. Estimated EA for each virtual orbital
    #     estimated_EA = np.zeros_like(virt_energies)
    #     estimated_EA[0] = EA1
    #     for k in range(1, len(virt_energies)):
    #         if k-1 < len(nu):
    #             estimated_EA[k] = EA1 + nu[k-1]
    #         else:
    #             estimated_EA[k] = estimated_EA[k-1] + 2.0   # safe fallback

    #     # 4. Interpolate to find where EA = 0
    #     virt_eV = virt_energies * 27.2114
    #     f = interp1d(estimated_EA, virt_eV, kind='linear', fill_value='extrapolate')
    #     epsilon0_eV = f(0.0)                 # eigenvalue where EA crosses zero
    #     epsilon0 = epsilon0_eV / 27.2114     # back to Hartree

    #     return epsilon0

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
        if A.ndim == 3:
            return all(np.allclose(a, a.conj().T, rtol=0, atol=tol) for a in A)
        return np.allclose(A, A.conj().T, rtol=0, atol=tol)

    def get_volume(self, coords):
        volume = 0
        for atom in coords:
            if atom in constants.vdw_radii:
                radius = constants.vdw_radii[atom]
                volume += (4 / 3) * math.pi * (radius ** 3)
            else:
                logger.warning(f"Warning: No van der Waals radius for {atom}")
            
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

        Gamma_orth = self.X.conj().T @ G_ao @ self.X 
        ==> inv(self.X.conj().T) @ Gamma_orth = G_ao @ self.X 
        ==> inv(self.X.conj().T) @ Gamma_orth @ inv(self.X) = G_ao

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

        def _gamma_one_spin(F_orth_s):
            eps, C_prime = np.linalg.eigh(F_orth_s)
            M = len(eps)
            gamma = np.zeros(M, dtype=float)
            for i in range(M):
                e_tilde = eps[i] - eps0
                if e_tilde > 0:
                    gamma[i] = min(clamp, gam0 * (np.exp(xi * e_tilde) - 1.0))
            Gamma_orth = C_prime @ np.diag(gamma) @ C_prime.conj().T
            Gamma_ao = inv(self.X.conj().T) @ Gamma_orth @ inv(self.X)
            return Gamma_ao
        
        if F_orth.ndim == 3:
            return np.stack([_gamma_one_spin(F_orth[s]) for s in range(F_orth.shape[0])])
        return _gamma_one_spin(F_orth)
