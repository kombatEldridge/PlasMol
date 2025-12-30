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

from .. import constants
from ..quantum.checkpoint import restart_from_checkpoint

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
        # If basis set is 'd-aug-cc-pvtz', import using bse
        if params.basis.lower() == 'd-aug-cc-pvtz':
            import basis_set_exchange as bse
            basis_h_str = bse.get_basis(params.basis, elements=[1], fmt='nwchem')
            basis_o_str = bse.get_basis(params.basis, elements=[8], fmt='nwchem')
            basis_h = gto.basis.parse(basis_h_str)
            basis_o = gto.basis.parse(basis_o_str)
            params.basis = {'H': basis_h, 'O': basis_o}

        # Format molecule string as required by PySCF
        self.mol = gto.M(atom=params.molecule_coords,
                    basis=params.basis,
                    unit='B', # Units of molecule are changed in parser.py to Bohr
                    charge=int(params.charge),
                    spin=int(params.spin))
        self.mol.verbose = 0
        self.mf = dft.RKS(self.mol)
        self.xc = params.xc
        self.mu = params.mu
        if 'LC' in self.xc or 'CAM' in self.xc:
            if self.mu is None:
                logger.warning("No mu value found in LC PBE functional, running tuning process.")
                self.mu = self.xc_tuning()
            if self.xc == 'LCBLYP' or self.xc == 'CAMBLYP':
                self.xc = f'RSH({self.mu}, 0.0, 1.0) + B88, LYP'
            if self.xc == 'LCwPBE' or self.xc == 'CAMwPBE':
                self.xc = f'RSH({self.mu}, 1.0, -1.0) + wPBEH, PBE'
            if self.xc == 'LCPBE' or self.xc == 'CAMPBE':
                self.xc = f'RSH({self.mu}, 1, -1.0) + PBE, PBE'
            print(f"Using LC functional {self.xc} with mu = {self.mu}")
            self.mf.omega = self.mu
        
        self.mf.xc = self.xc
        self.mf.kernel()

        print(f"Occupied count: {np.sum(self.mf.mo_occ > 0)}")
        print(f"-E_HOMO energy: {-self.mf.mo_energy[self.mf.mo_occ > 0][-1]} Ha,{-self.mf.mo_energy[self.mf.mo_occ > 0][-1] * 27.2114} eV")

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

        self.chkpoint_time = 0
        
        self.damping = params.damping
        self.damping_params = {}
        if self.damping is not None:
            if params.gam0 is not None:
                self.damping_params["gam0"] = params.gam0
            if params.xi is not None:
                self.damping_params["xi"] = params.xi
            if params.eps0 is not None:
                self.damping_params["eps0"] = params.eps0
            else:
                self.damping_params["eps0"] = self.compute_vacuum_level()
            if params.clamp is not None:
                self.damping_params["clamp"] = params.clamp
            self.Gamma_ao_0 = self.get_gamma_ao(**self.damping_params)

        if params.checkpoint_path is not None and os.path.exists(params.checkpoint_path):
            restart_from_checkpoint(self, params)
            self.D_ao = self.mf.make_rdm1(mo_occ=self.occ)
            self.F_orth = self.get_F_orth(self.D_ao) # Should this include exc? at what time?
        else: 
            self.D_ao = self.D_ao_0
            self.F_orth = self.get_F_orth(self.D_ao)
            self.F_orth_n12dt = self.F_orth

        if not self.is_hermitian(self.D_ao, tol=1e-12):
            raise ValueError("Initial density matrix in AO is not Hermitian")
        
        self.volume = self.get_volume(params.molecule_atoms)

    def xc_tuning(self, tol=1e-3):
        """Tune mu for LC functionals to minimize |E_cat - E_neut + ε_HOMO|."""
        if 'lc' not in self.xc.lower() and 'cam' not in self.xc.lower():
            return

        def j_func(mu):
            logger.debug(f"Mu Tuning: Evaluating at mu = {mu}")
            xc = self.xc
            if xc == 'LCBLYP' or xc == 'CAMBLYP':
                xc = f'RSH({mu}, 0.0, 1.0) + B88, LYP'
            neutral_mf = scf.UKS(self.mol)
            neutral_mf.xc = xc
            neutral_mf.omega = mu
            neutral_mf.kernel()
            if not neutral_mf.converged:
                logger.warning(f"Neutral SCF not converged at mu={mu}")

            e_homo_alpha = neutral_mf.mo_energy[0][neutral_mf.mo_occ[0] > 0][-1] if any(neutral_mf.mo_occ[0] > 0) else -np.inf
            e_homo_beta = neutral_mf.mo_energy[1][neutral_mf.mo_occ[1] > 0][-1] if any(neutral_mf.mo_occ[1] > 0) else -np.inf
            e_homo = max(e_homo_alpha, e_homo_beta)

            cation_mol = self.mol.copy()
            cation_mol.charge = self.mol.charge + 1
            cation_mol.spin = 1 if self.mol.spin == 0 else self.mol.spin - 1
            cation_mf = scf.UKS(cation_mol)
            if xc == 'LCBLYP' or xc == 'CAMBLYP':
                xc = f'RSH({mu}, 0.0, 1.0) + B88, LYP'
            cation_mf.xc = xc
            cation_mf.omega = mu
            cation_mf.kernel()
            if not cation_mf.converged:
                logger.warning(f"Cation SCF not converged at mu={mu}")

            ip_scf = cation_mf.energy_tot() - neutral_mf.energy_tot()
            j = np.abs(e_homo + ip_scf)
            logger.debug(f"Mu={mu:.3f}, IP_SCF={ip_scf:.3f} Ha, ε_HOMO={e_homo:.3f} Ha, J={j:.3f}")
            return j

        res = opt.minimize_scalar(j_func, bounds=(0.1, 1.0), tol=tol, method='bounded', options={'maxiter': 100})
        if res.success:
            mu = res.x
            logger.info(f"Tuned mu={mu:.3f}, min J={res.fun:.3f}")
            return mu
        else:
            raise ValueError(f"Tuning failed: {res.message}")

    def compute_vacuum_level(self, nstates=20, diffuse_basis='d-aug-cc-pvqz'):
        """Estimate ε_0 by interpolating KS ε vs. EA_k (paper eq 12-13)."""
        from pyscf import scf, tdscf, lib
        from scipy.interpolate import interp1d

        if diffuse_basis == 'd-aug-cc-pvtz':
            import basis_set_exchange as bse
            basis_h_str = bse.get_basis(diffuse_basis, elements=[1], fmt='nwchem')
            basis_o_str = bse.get_basis(diffuse_basis, elements=[8], fmt='nwchem')

            basis_h = gto.basis.parse(basis_h_str)
            basis_o = gto.basis.parse(basis_o_str)

            diffuse_basis = {'H': basis_h, 'O': basis_o}

        original_basis = self.mol.basis
        self.mol.basis = diffuse_basis
        self.mol.build()

        # Neutral SCF
        if self.mol.spin == 0:
            neutral_mf = scf.RKS(self.mol)
        else:
            neutral_mf = scf.UKS(self.mol)
        neutral_mf.xc = self.xc
        neutral_mf.kernel()
        e_neutral = neutral_mf.energy_tot()
        mo_energy = neutral_mf.mo_energy
        mo_occ = neutral_mf.mo_occ

        # Extract and sort virtual orbital energies
        if self.mol.spin == 0:  # RKS
            virtual_indices = np.where(mo_occ == 0)[0]
            virtual_eps = np.sort(mo_energy[virtual_indices])
        else:  # UKS
            ea, eb = mo_energy
            oa, ob = mo_occ
            virtual_eps_a = ea[oa == 0]
            virtual_eps_b = eb[ob == 0]
            virtual_eps = np.sort(np.concatenate((virtual_eps_a, virtual_eps_b)))
        print('Sorted virtual KS energies (first 11):', virtual_eps[:11])

        # Anion UKS (charge=-1, spin=neutral.spin + 1 for high-spin pairing)
        anion_mol = self.mol.copy()
        anion_mol.charge = self.mol.charge - 1
        anion_mol.spin = self.mol.spin + 1
        anion_mol.build()
        anion_mf = scf.UKS(anion_mol)
        anion_mf.xc = self.xc
        anion_mf.max_cycle = 200

        # Initialize from neutral density matrix
        dm_neutral = neutral_mf.make_rdm1()
        if not isinstance(dm_neutral, tuple):
            dm0 = (dm_neutral / 2, dm_neutral / 2)
        else:
            dm0 = dm_neutral
        anion_mf.kernel(dm0=dm0)
        e_anion = anion_mf.energy_tot()

        ea1 = e_anion - e_neutral
        print('EA1:', ea1)

        # TDDFT on anion for excitations v_k
        td = tdscf.TDDFT(anion_mf)
        td.nstates = nstates
        td.kernel()
        v_k = td.e
        print('TDDFT excitation energies v_k:', v_k)

        # Build EA list: EA1, EA1 + v1, EA1 + v2, ...
        eas = [ea1] + [ea1 + v for v in v_k]
        print('Constructed EAs:', eas)

        # Pair with virtual ε (assume energy order, truncate to len(eas))
        num_pairs = min(len(virtual_eps), len(eas))
        eps_paired = virtual_eps[:num_pairs]
        eas_paired = np.array(eas[:num_pairs])
        print('Paired eps:', eps_paired)
        print('Paired eas:', eas_paired)

        # Interpolate EA(ε) to find ε where EA=0
        if np.all(eas_paired > 0):
            print("All EAs positive, extrapolating left")
            f = interp1d(eas_paired, eps_paired, kind='linear', fill_value='extrapolate')
            epsilon0 = f(0)
        elif np.all(eas_paired < 0):
            print("All EAs negative, extrapolating right")
            f = interp1d(eas_paired, eps_paired, kind='linear', fill_value='extrapolate')
            epsilon0 = f(0)
        else:
            print("Mixed EAs, interpolating")
            f = interp1d(eas_paired, eps_paired, kind='linear')
            epsilon0 = f(0)

        self.mol.basis = original_basis
        self.mol.build()
        logger.info(f"Computed ε_0={epsilon0:.4f} Ha")
        sys.exit(0)
        return epsilon0


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
        if self.damping == 'static':
            F_ao -= 1j * self.Gamma_ao_0
        elif self.damping == 'dynamic':
            F_ao -= 1j * self.get_gamma_ao(**self.damping_params, D_ao=D_ao)
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

    def get_gamma_ao(self, gam0=1.0, xi=0.5, eps0=0.05, clamp=100, D_ao=None):
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
        F_orth = np.matmul(self.X.T, np.matmul(F_ao, self.X))
        # F_orth = np.matmul(self.X.conj().T, np.matmul(F_ao, self.X))
        eps, C_prime = np.linalg.eigh(F_orth)

        M = len(eps)
        gamma = np.zeros(M)
        for i in range(M):
            e_tilde = eps[i] - eps0
            if e_tilde > 0:
                gamma[i] = gam0 * (np.exp(xi * e_tilde) - 1.0)
            gamma[i] = min(clamp, gamma[i])

        Lambda = np.diag(gamma)
        Gamma_mo = np.matmul(C_prime, np.matmul(Lambda, C_prime.conj().T))
        Gamma_ao = np.matmul(inv(self.X.T), np.matmul(Gamma_mo, inv(self.X.T).T))
        # Gamma_ao = np.matmul(scipy.linalg.inv(self.X.conj().T), np.matmul(Gamma_mo, scipy.linalg.inv(self.X).conj().T))
        return Gamma_ao
    
    # define more contractions with the density matrix below...