import numpy as np
import pytest
from pyscf import gto, dft

from plasmol.drivers.custom_drivers.tune import (
    _get_homo,
    _resolve_lrc,
    _tune_eps0,
    _tune_lrc_parameter,
    _virt_energies,
)


class _TuneParams:
    def __init__(self, atom, basis, charge=0, spin=0, xc="pbe0", lrc=None):
        self.molecule_coords = atom
        self.molecule_basis = basis
        self.molecule_charge = charge
        self.molecule_spin = spin
        self.molecule_xc = xc
        if lrc is not None:
            self.molecule_lrc_parameter = lrc


def test_resolve_lrc_from_functional_default():
    params = _TuneParams("Na 0 0 0", "631g*", xc="HYB_GGA_XC_LC_WPBE")
    assert np.isclose(_resolve_lrc(params), 0.4)


def test_resolve_lrc_numeric():
    params = _TuneParams("Na 0 0 0", "631g*", xc="HYB_GGA_XC_LC_WPBE", lrc=0.339)
    assert np.isclose(_resolve_lrc(params), 0.339)


def test_get_homo_open_shell_uses_alpha():
    mol = gto.M(atom="Na 0 0 0", basis="631g*", spin=1, verbose=0)
    mf = dft.UKS(mol, xc="HYB_GGA_XC_LC_WPBE")
    mf.omega = 0.339
    mf.kernel()

    en = np.asarray(mf.mo_energy)
    occ = np.asarray(mf.mo_occ)
    alpha_homo = en[0][occ[0] > 0.5][-1]

    assert np.isclose(_get_homo(mf), alpha_homo)
    assert _get_homo(mf) > en[1][occ[1] > 0.5][-1]


def test_virt_energies_open_shell_matches_alpha_channel():
    mol = gto.M(atom="Na 0 0 0", basis="631g*", spin=1, verbose=0)
    mf = dft.UKS(mol, xc="HYB_GGA_XC_LC_WPBE")
    mf.omega = 0.339
    mf.kernel()

    en = np.asarray(mf.mo_energy)
    occ = np.asarray(mf.mo_occ)
    expected = en[0][occ[0] < 0.5]

    assert np.allclose(_virt_energies(mf), expected)


def test_virt_energies_closed_shell():
    mol = gto.M(atom="Li 0 0 0; Li 0 0 3.779446474", unit="B", basis="def2-TZVP", spin=0, verbose=0)
    mf = dft.RKS(mol, xc="HYB_GGA_XC_LC_WPBE")
    mf.omega = 0.258369
    mf.kernel()

    expected = mf.mo_energy[mf.mo_occ < 0.5]
    assert np.allclose(_virt_energies(mf), expected)


def test_tune_lrc_parameter_na_open_shell():
    params = _TuneParams(
        "Na 0 0 0",
        "631g*",
        charge=0,
        spin=1,
        xc="HYB_GGA_XC_LC_WPBE",
    )
    omega = _tune_lrc_parameter(params)
    assert 0.2 < omega < 0.5


def test_tune_eps0_na_open_shell():
    params = _TuneParams(
        "Na 0 0 0",
        "631g*",
        charge=0,
        spin=1,
        xc="HYB_GGA_XC_LC_WPBE",
        lrc=0.339181,
    )
    eps0 = _tune_eps0(params)
    assert 0.015 < eps0 < 0.035


def test_tune_eps0_li2_closed_shell():
    # 2 Å Li-Li distance expressed in Bohr (matches Li2.xyz + angstrom input)
    params = _TuneParams(
        "Li 0 0 0; Li 0 0 3.779446474",
        "def2-TZVP",
        charge=0,
        spin=0,
        xc="HYB_GGA_XC_LC_WPBE",
        lrc=0.258369,
    )
    eps0 = _tune_eps0(params)
    assert np.isclose(eps0, 0.009207, rtol=0.05)


def test_tune_eps0_h_atom_single_virtual_extrapolates():
    params = _TuneParams(
        "H 0 0 0",
        "6-31g",
        charge=0,
        spin=1,
        xc="pbe0",
        lrc=0.0,
    )
    eps0 = _tune_eps0(params)
    assert np.isfinite(eps0)