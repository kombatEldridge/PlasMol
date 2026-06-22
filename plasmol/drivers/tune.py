# drivers/tune.py
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
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

logger = logging.getLogger("main")


def _get_homo(mf):
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
    Estimate ε_0 (vacuum level) for CAP.
    Ported from previous molecule.compute_vacuum_level so that tuning logic lives only in the driver.
    """
    logger.info("Calculating vacuum level ε_0 (tune driver)")

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
    lrc = getattr(params, 'molecule_lrc_parameter', 0.0)

    # 1. Ground-state calculations
    if params.molecule_spin == 0:
        mf_n = dft.RKS(mol_neutral, xc=xc)
    else:
        mf_n = dft.UKS(mol_neutral, xc=xc)
    mf_n.omega = lrc
    mf_n.kernel()
    E_n = mf_n.e_tot

    # Virtual orbital energies from neutral
    if params.molecule_spin == 0:
        virt_energies = mf_n.mo_energy[mf_n.mo_occ < 0.5]
    else:
        virt_energies = mf_n.mo_energy[0][mf_n.mo_occ[0] < 0.5]

    # Anion: always UKS
    mf_a = dft.UKS(mol_anion, xc=xc)
    mf_a.omega = lrc
    mf_a.kernel()
    E_a = mf_a.e_tot

    EA1 = (E_a - E_n) * 27.2114

    # 2. LRTDDFT on the anion
    n_virt = len(virt_energies)
    n_states = max(20, n_virt - 1)

    td = tdscf.TDA(mf_a)
    td.conv_tol = 1e-5
    td.max_cycle = 100
    td.lindep = 1e-8
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
