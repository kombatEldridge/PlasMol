# quantum/molecule.py
import os
import sys
import math
import logging
import numpy as np
import scipy.linalg
from pyscf import gto, dft
from pyscf import lib
from pyscf.scf import addons
from pyscf.dft import libxc
from scipy.linalg import inv

from plasmol.utils import constants
from plasmol.utils.csv import init_csv, update_csv

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
        self.is_open_shell = (self.molecule_spin != 0) or getattr(self, 'force_open_shell', False)
        if self.is_open_shell:
            logger.debug("Initializing open-shell calculation.")
            self.mf = dft.UKS(self.mol)
        else:
            logger.debug("Initializing closed-shell calculation.")
            self.mf = dft.RKS(self.mol)
        if hasattr(self, 'molecule_lrc_parameter'):
            if isinstance(self.molecule_lrc_parameter, (int, float)):
                self.mf.omega = self.molecule_lrc_parameter
        self.mf.xc = self.molecule_xc
        self.mf.kernel()
        self.nmat = 2 if self.is_open_shell else 1

        # Neutral MOs / occupations (fixed projection basis for DCH hole dynamics)
        self.C = self.mf.mo_coeff.copy()
        if not self.is_open_shell:
            self.occ_neutral = self.mf.get_occ().copy()
            self.mo_energy = self.mf.mo_energy.copy()
        else:
            self.occ_neutral = (self.mf.mo_occ[0].copy(), self.mf.mo_occ[1].copy())
            self.mo_energy = (self.mf.mo_energy[0].copy(), self.mf.mo_energy[1].copy())

        if self.has_dch:
            self._setup_dch_mo_logging()
            self.remove_core_electrons(self.mo_removal_index_dict)
            self.occ = self.mf.mo_occ
        else:
            self.occ = self.mf.get_occ()

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
                
        skip_checkpoint = False
        if self.resumed_from_checkpoint:
            dir_component = getattr(params, 'molecule_source_component') if self.has_fourier else None
            if dir_component in params.not_checkpointed_dirs:
                skip_checkpoint = True

        if self.resumed_from_checkpoint and not skip_checkpoint:
            suffix = f"_{dir_component}" if self.has_fourier and dir_component else ""
            self.D_ao_0 = self.values_from_checkpoint[f"D_ao_0{suffix}"]
            self.mf.mo_coeff = self.values_from_checkpoint[f"mo_coeff{suffix}"]
            self.D_ao = self.mf.make_rdm1(mo_occ=self.occ)
            self.F_orth = self.get_F_orth(self.D_ao) # Should this include exc? at what time?
            if self.molecule_propagator_str == 'magnus2':
                self.F_orth_n12dt = self.values_from_checkpoint[f"F_orth_n12dt{suffix}"]
            elif self.molecule_propagator_str == 'step':
                self.C_orth_ndt = self.values_from_checkpoint[f"C_orth_ndt{suffix}"]
        else:
            if self.has_dch and getattr(self, '_dch_dm0', None) is not None:
                self.D_ao_0 = np.asarray(self._dch_dm0, dtype=np.complex128)
                self.D_ao = self.D_ao_0.copy()
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


    def get_F_orth(self, D_ao, exc=None):
        """
        Compute the Fock matrix in the orthogonal basis.
        Fully supports RKS and UKS.
        """
        F_ao = self.mf.get_fock(dm=D_ao).astype(np.complex128)
        if exc is not None:
            F_ao = F_ao + self.calculate_potential(exc)
        if self.has_cap:
            cap_kwargs = {k: v for k, v in self.cap_dict.items() if k != "type"}
            if not hasattr(self, 'Gamma_ao_0'):
                self.Gamma_ao_0 = self.get_gamma_ao(**cap_kwargs)
            if self.cap_type == 'static':
                F_ao = F_ao - 1j * self.Gamma_ao_0
            elif self.cap_type == 'dynamic':
                F_ao = F_ao - 1j * self.get_gamma_ao(**cap_kwargs, D_ao=D_ao)

        if self.is_open_shell:
            return np.stack([self.X.conj().T @ F_ao[s] @ self.X for s in range(2)])
        else:
            return self.X.conj().T @ F_ao @ self.X
    
    def rotate_coeff_to_orth(self, coeff_ao):
        """
        Transform MO coefficients from AO → orthogonal basis.
        Supports both RKS (2-D) and UKS (3-D / list).
        """
        Xinv = np.linalg.inv(self.X)
        if self.is_open_shell:
            # coeff_ao is (2, nao, nmo) or list of two arrays
            return np.stack([Xinv @ coeff_ao[s] for s in range(2)])
        else:
            return Xinv @ coeff_ao

    def rotate_coeff_away_from_orth(self, coeff_orth):
        """
        Transform MO coefficients from orthogonal → AO basis.
        Supports both RKS and UKS.
        """
        if self.is_open_shell:
            return np.stack([self.X @ coeff_orth[s] for s in range(2)])
        else:
            return self.X @ coeff_orth
    
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
    
    def remove_core_electrons(self, mo_removal_index_dict):
        """
        Sudden single/double core-hole: remove electrons from selected MOs without
        re-optimizing the SCF. Density is non-stationary; analysis uses neutral MOs.

        Parameters
        ----------
        mo_removal_index_dict : dict
            Map of 0-based MO index → number of electrons to remove (1 or 2).
        """
        mo_gs = self.mf.mo_coeff.copy()
        mf_gs = self.mf.copy()

        nmo = mo_gs.shape[-1]
        for mo_idx in mo_removal_index_dict:
            if mo_idx >= nmo:
                raise ValueError(
                    f"MO index {mo_idx} is out of range (molecule has {nmo} MOs, 0-based)."
                )

        logger.debug("Before removing core electrons:")
        self.print_occ(5)

        original_charge = self.mf.mol.charge
        original_spin = self.mf.mol.spin

        n_holes = sum(mo_removal_index_dict.values())
        if n_holes < 1:
            raise ValueError("mo_removal_index_dict must remove at least one electron.")

        self.mf.mol.charge = original_charge + n_holes
        if n_holes == 2:
            if len(mo_removal_index_dict) == 1:
                self.mf.mol.spin = original_spin  # closed-shell double hole
            else:
                self.mf.mol.spin = original_spin + 2  # one hole on each of two MOs (α,α)
        else:
            self.mf.mol.spin = original_spin + 1

        self.mf.mol.build(False, False)

        setocc = [self.mf.mo_occ[0].copy(), self.mf.mo_occ[1].copy()]
        for mo, n_remove in mo_removal_index_dict.items():
            for spin_idx in range(n_remove):
                setocc[spin_idx][mo] = 0

        dm_sudden = mf_gs.make_rdm1(mo_gs, setocc)
        self.mf = addons.mom_occ(self.mf, mo_gs, setocc)
        self.mf.mo_coeff = mo_gs
        self.mf.mo_occ = setocc
        self._dch_dm0 = dm_sudden

        logger.debug("After sudden core-hole creation (no SCF re-optimization):")
        self.print_occ(5)

    
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
    
        pot = -1.0 * np.einsum('xij,x->ij', mu, exc)

        if self.is_open_shell:
            return np.stack([pot, pot])
        return pot
    
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

    def print_occ(self, n_print=30):
        logger.debug(f"{'MO':>4} | {'Eα (Ha)':>12} | {'occα':>5} | {'Eβ (Ha)':>12} | {'occβ':>5}")
        logger.debug("-" * 55)

        for i in range(min(n_print, len(self.mf.mo_energy[0]))):
            logger.debug(f"{i+1:4d} | {self.mf.mo_energy[0][i]:12.5f} | "
                f"{self.mf.mo_occ[0][i]:5.1f} | "
                f"{self.mf.mo_energy[1][i]:12.5f} | "
                f"{self.mf.mo_occ[1][i]:5.1f}")

    def _neutral_lumo_index(self):
        """
        0-based LUMO index of the neutral molecule (first virtual MO).

        For open-shell (UKS) takes the lower of the α/β first-virtual indices
        so the logged range covers both spin channels through LUMO+1.
        """
        if self.is_open_shell:
            occ_a = np.asarray(self.occ_neutral[0])
            occ_b = np.asarray(self.occ_neutral[1])
            virt_a = np.where(occ_a < 0.5)[0]
            virt_b = np.where(occ_b < 0.5)[0]
            if len(virt_a) == 0 and len(virt_b) == 0:
                return len(occ_a) - 1
            candidates = []
            if len(virt_a):
                candidates.append(int(virt_a[0]))
            if len(virt_b):
                candidates.append(int(virt_b[0]))
            return min(candidates)
        occ = np.asarray(self.occ_neutral)
        virt = np.where(occ < 0.5)[0]
        if len(virt) == 0:
            return len(occ) - 1
        return int(virt[0])

    def _setup_dch_mo_logging(self):
        """
        Log hole occupations for neutral MOs 0 .. LUMO+1 (inclusive).

        Plot selection is separate (``dch_watch_indices`` in the input file).
        """
        nmo = int(np.asarray(self.C).shape[-1])
        lumo = self._neutral_lumo_index()
        max_log = min(lumo + 1, nmo - 1)  # one above LUMO, clipped to last MO
        self.dch_log_indices = list(range(0, max_log + 1))
        logger.debug(f"Neutral LUMO index={lumo}, logging MO indices 0..{max_log} (inclusive), nmo={nmo}.")

        filepath = getattr(self, 'dch_mo_occ_filepath', None)
        if not filepath:
            raise ValueError(
                "DCH driver requires 'dch_mo_occ_filepath' under additional_parameters."
            )

        if self.resumed_from_checkpoint and os.path.exists(filepath):
            logger.debug(f"Resuming DCH MO occupation tracking from existing {filepath}")
            return

        header = ['Timestamps (au)'] + [f'MO index {i}' for i in self.dch_log_indices]
        init_csv(
            filepath,
            f"Time-dependent hole occupations (neutral MO basis) for MO indices: "
            f"{self.dch_log_indices}",
            header=header,
        )
        if self.resumed_from_checkpoint:
            logger.warning(
                "Resumed DCH run but no mo_occ CSV was restored from the checkpoint; "
                "started a fresh occupation file at current time."
            )
        else:
            logger.debug(f"DCH MO occupation file initialized: {filepath}")

    def get_mo_occupations(self, current_time):
        """
        Hole occupations on the neutral MO basis (Fig. 8 convention).
        DCH always runs open-shell (UKS).

            n_k^e(t) = [C_n† S D_AO(t) S C_n]_{kk}
            h_k(t)   = n_k^neutral - n_k^e(t)

        Positive h_k is loss of electronic density; negative is gain.

        All MOs in ``dch_log_indices`` (0 through neutral LUMO+1) are written;
        ``dch_watch_indices`` only selects which series are plotted at the end.
        """
        S = self.S
        C_a, C_b = self.C[0], self.C[1]

        def _electron_occ(D_ao, C):
            D_mo = (C.conj().T @ S) @ D_ao @ (S @ C)
            return D_mo.diagonal().real

        n_e = _electron_occ(self.D_ao[0], C_a) + _electron_occ(self.D_ao[1], C_b)
        n0 = np.asarray(self.occ_neutral[0]) + np.asarray(self.occ_neutral[1])
        values = (n0 - n_e)[self.dch_log_indices]
        update_csv(self.dch_mo_occ_filepath, current_time, None, None, None, *values)
