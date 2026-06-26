# drivers/custom_drivers/tune.py
"""
Tuning driver for automatically determining optimal lrc_parameter (μ) and/or cap eps0.

This is invoked by setting "driver": "tune" in the settings section of the input file
when "tune" (string) or "{TUNE}" appears for the relevant parameters.
It performs the tuning calculations, logs the results, and exits without running
any time propagation.
"""
import logging

import numpy as np
from pyscf import gto, dft, tdscf
from pyscf.dft import libxc
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

logger = logging.getLogger("main")


def _resolve_lrc(params):
    """Return a numeric range-separation parameter for ground-state SCF."""
    lrc = getattr(params, "molecule_lrc_parameter", None)
    if lrc is None or lrc == "tune":
        xc = getattr(params, "molecule_xc", "pbe0")
        derived_omega, _, _ = libxc.rsh_coeff(str(xc).upper())
        if derived_omega <= 0:
            return 0.0
        return derived_omega
    return float(lrc)


def _virt_energies(mf):
    """Return virtual-orbital energies used for ε₀ estimation."""
    en = np.asarray(mf.mo_energy)
    occ = np.asarray(mf.mo_occ)
    if en.ndim == 2:
        # For UKS, the anion adds an alpha electron; use alpha virtuals.
        return en[0][occ[0] < 0.5]
    return en[occ < 0.5]


def _get_homo(mf):
    """Return the HOMO energy used for Koopmans tuning (works for RKS and UKS)."""
    en = np.asarray(mf.mo_energy)
    occ = np.asarray(mf.mo_occ)
    if en.ndim == 2 or isinstance(mf.mo_energy, list):
        occ_a = occ[0] > 0.5
        homo_a = en[0][occ_a][-1] if np.any(occ_a) else -np.inf
        occ_b = occ[1] > 0.5
        homo_b = en[1][occ_b][-1] if np.any(occ_b) else -np.inf
        # For open-shell UKS, ionization removes an alpha electron; use alpha HOMO.
        return homo_a if np.any(occ_a) else max(homo_a, homo_b)
    occ = mf.mo_occ > 0.5
    return mf.mo_energy[occ][-1]


def _tune_lrc_parameter(params):
    """
    Tune μ (lrc_parameter) for LC functionals to minimize |E_cat - E_neut + ε_HOMO|.
    Ported from previous molecule.xc_tuning so that tuning logic lives only in the driver.
    """
    mol_neutral = gto.M(
        atom=params.molecule_coords,
        unit='B',
        basis=params.molecule_basis,
        charge=params.molecule_charge,
        spin=params.molecule_spin,
        verbose=0
    )
    mol_cation = gto.M(
        atom=params.molecule_coords,
        unit='B',
        basis=params.molecule_basis,
        charge=params.molecule_charge + 1,
        spin=abs(params.molecule_spin - 1),
        verbose=0
    )

    xc_template = getattr(params, 'molecule_xc', 'pbe0')

    def compute_J(omega):
        """Compute J(ω) = |IP_ΔSCF + ε_HOMO| (in Hartree)."""
        omega = float(omega)
        xc = xc_template
        if isinstance(xc, str) and "{TUNE}" in xc.upper():
            xc = xc.upper().replace("{TUNE}", str(omega))

        # Neutral
        if params.molecule_spin == 0:
            mf_n = dft.RKS(mol_neutral, xc=xc)
        else:
            mf_n = dft.UKS(mol_neutral, xc=xc)
        mf_n.omega = omega
        mf_n.conv_tol = 1e-9
        mf_n.conv_tol_grad = 1e-7
        mf_n.max_cycle = 120
        mf_n.init_guess = 'minao'
        mf_n.diis_space = 12
        mf_n.kernel()
        E_n = mf_n.e_tot
        eps_homo_n = _get_homo(mf_n)

        # Cation
        if abs(params.molecule_spin - 1) == 0:
            mf_c = dft.RKS(mol_cation, xc=xc)
        else:
            mf_c = dft.UKS(mol_cation, xc=xc)
        mf_c.omega = omega
        mf_c.conv_tol = 1e-9
        mf_c.conv_tol_grad = 1e-7
        mf_c.max_cycle = 120
        mf_c.init_guess = 'minao'
        mf_c.diis_space = 12
        mf_c.kernel()
        E_c = mf_c.e_tot

        J = abs(E_c - E_n + eps_homo_n)
        return J

    logger.info("Calculating μ parameter for xc functional (tune driver)")
    res = minimize_scalar(
        compute_J, bounds=(0.01, 0.8), method="bounded",
        options={'xatol': 1e-7, 'maxiter': 1000}
    )
    logger.info(f"Optimal μ (lrc_parameter) = {res.x:.6f}")
    return res.x


def _tune_eps0(params):
    """
    Estimate ε₀ (vacuum level) for CAP.
    Ported from previous molecule.compute_vacuum_level so that tuning logic lives only in the driver.
    """
    logger.info("Calculating vacuum level ε₀ (tune driver)")

    mol_neutral = gto.M(
        atom=params.molecule_coords,
        unit='B',
        basis=params.molecule_basis,
        charge=params.molecule_charge,
        spin=params.molecule_spin,
        verbose=0
    )
    mol_anion = gto.M(
        atom=params.molecule_coords,
        unit='B',
        basis=params.molecule_basis,
        charge=params.molecule_charge - 1,
        spin=abs(params.molecule_spin - 1),
        verbose=0
    )

    xc = getattr(params, 'molecule_xc', None)
    lrc = _resolve_lrc(params)

    # 1. Ground-state calculations
    if params.molecule_spin == 0:
        mf_n = dft.RKS(mol_neutral, xc=xc)
    else:
        mf_n = dft.UKS(mol_neutral, xc=xc)
    mf_n.omega = lrc
    mf_n.kernel()
    E_n = mf_n.e_tot

    virt_energies = _virt_energies(mf_n)
    if virt_energies.size == 0:
        raise ValueError("No virtual orbitals found for ε₀ tuning.")

    # Anion: always UKS (supports both closed- and open-shell anions)
    mf_a = dft.UKS(mol_anion, xc=xc)
    mf_a.omega = lrc
    mf_a.kernel()
    E_a = mf_a.e_tot

    EA1 = (E_a - E_n) * 27.2114
    virt_eV = virt_energies * 27.2114

    if virt_energies.size == 1:
        # Single virtual: linear extrapolation from (EA1, ε_virt) to EA = 0.
        epsilon0_eV = virt_eV[0] - EA1
        logger.debug(
            "Only one virtual orbital available for ε₀ tuning; "
            "using linear extrapolation from ΔSCF EA."
        )
    else:
        # 2. LRTDDFT on the anion
        n_states = max(20, virt_energies.size - 1)

        td = tdscf.TDA(mf_a)
        td.conv_tol = 1e-5
        td.max_cycle = 100
        td.lindep = 1e-8
        td.kernel(nstates=n_states)
        nu = td.e * 27.2114

        # 3. Estimated EA for each virtual orbital
        estimated_EA = np.zeros_like(virt_energies)
        estimated_EA[0] = EA1
        for k in range(1, virt_energies.size):
            if k - 1 < len(nu):
                estimated_EA[k] = EA1 + nu[k - 1]
            else:
                estimated_EA[k] = estimated_EA[k - 1] + 2.0

        if not np.all(np.diff(estimated_EA) > 0):
            logger.warning(
                "Estimated EA values are not strictly increasing; "
                "ε₀ interpolation may be unreliable."
            )

        # 4. Interpolate to find where EA = 0
        f = interp1d(estimated_EA, virt_eV, kind='linear', fill_value='extrapolate')
        epsilon0_eV = float(f(0.0))

    epsilon0 = epsilon0_eV / 27.2114
    logger.info(f"Vacuum level ε₀ = {epsilon0:.6f} Ha")

    return epsilon0


def run(params):
    """Regular (non-custom) tuning driver entry point."""
    logger.info(" === RUNNING TUNE DRIVER === ")

    # Determine what needs to be tuned
    lrc = getattr(params, 'molecule_lrc_parameter', None)
    xc = getattr(params, 'molecule_xc', "")
    has_lrc_tune = (lrc == "tune")
    has_xc_tune = isinstance(xc, str) and "{TUNE}" in xc.upper()

    do_lrc_tune = has_lrc_tune or has_xc_tune

    do_eps_tune = False
    if getattr(params, 'has_cap', False):
        cap_dict = getattr(params, 'cap_dict', {}) or {}
        if cap_dict.get("eps0") == "tune":
            do_eps_tune = True

    if not do_lrc_tune and not do_eps_tune:
        logger.info("No tuning markers ('tune' or '{TUNE}') found. Nothing to tune.")
        return

    tuned_lrc = None

    if do_lrc_tune:
        tuned_lrc = _tune_lrc_parameter(params)

        # Update params so that a subsequent eps0 tune (in same run) uses the tuned value
        params.molecule_lrc_parameter = tuned_lrc
        if has_xc_tune:
            params.molecule_xc = xc.upper().replace("{TUNE}", str(tuned_lrc))

    if do_eps_tune:
        tuned_eps = _tune_eps0(params)
        # Update for completeness (driver ends after this)
        if hasattr(params, 'cap_dict') and isinstance(params.cap_dict, dict):
            params.cap_dict = dict(params.cap_dict)
            params.cap_dict["eps0"] = tuned_eps

    logger.info(" === TUNE DRIVER COMPLETE (no propagation performed) ===")
