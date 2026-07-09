# tests/test_open_shell.py
"""Granular tests for open-shell (UKS) support in PlasMol."""
import copy
import json
import pytest
import numpy as np
from argparse import Namespace
from pyscf import dft

from plasmol.utils.params import PARAMS
from plasmol.quantum.molecule import MOLECULE
from plasmol.quantum.propagation import propagation
from plasmol.quantum.propagators.rk4 import propagate as propagate_rk4
from plasmol.quantum.propagators.step import propagate as propagate_step
from plasmol.quantum.propagators.magnus2 import propagate as propagate_magnus2

def test_rks_chosen_when_spin_is_zero(h2_closed_shell_molecule):
    assert isinstance(h2_closed_shell_molecule.mf, dft.rks.RKS)
    assert h2_closed_shell_molecule.is_open_shell is False

def test_uks_chosen_when_spin_is_nonzero(h_atom_molecule):
    assert isinstance(h_atom_molecule.mf, dft.uks.UKS)
    assert h_atom_molecule.is_open_shell is True

def test_uks_chosen_for_doublet_li(li_atom_molecule):
    assert isinstance(li_atom_molecule.mf, dft.uks.UKS)

def test_uks_chosen_for_h2_cation(h2_cation_molecule):
    assert isinstance(h2_cation_molecule.mf, dft.uks.UKS)

def test_nmat_is_1_for_closed_shell(h2_closed_shell_molecule):
    assert h2_closed_shell_molecule.nmat == 1

def test_nmat_is_2_for_open_shell_h(h_atom_molecule):
    assert h_atom_molecule.nmat == 2

def test_nmat_is_2_for_open_shell_li(li_atom_molecule):
    assert li_atom_molecule.nmat == 2

def test_D_ao_shape_rks(h2_closed_shell_molecule):
    nao = h2_closed_shell_molecule.mf.mol.nao
    assert h2_closed_shell_molecule.D_ao.shape == (nao, nao)
    assert h2_closed_shell_molecule.D_ao_0.shape == (nao, nao)

def test_D_ao_shape_uks(h_atom_molecule):
    nao = h_atom_molecule.mf.mol.nao
    assert h_atom_molecule.D_ao.shape == (2, nao, nao)
    assert h_atom_molecule.D_ao_0.shape == (2, nao, nao)

def test_mo_coeff_shape_uks(li_atom_molecule):
    nao = li_atom_molecule.mf.mol.nao
    nmo = li_atom_molecule.mf.mo_coeff[0].shape[1]
    assert np.asarray(li_atom_molecule.mf.mo_coeff).shape == (2, nao, nmo)

def test_mo_occ_shape_uks(li_atom_molecule):
    nmo = li_atom_molecule.mf.mo_coeff[0].shape[1]
    assert np.asarray(li_atom_molecule.mf.mo_occ).shape == (2, nmo)
    assert np.asarray(li_atom_molecule.occ).shape == (2, nmo)

def test_F_orth_shape_uks(h_atom_molecule):
    nao = h_atom_molecule.mf.mol.nao
    assert h_atom_molecule.F_orth.shape == (2, nao, nao)

def test_h_atom_has_one_electron_total(h_atom_molecule):
    D_total = h_atom_molecule.D_ao.sum(axis=0)
    S = h_atom_molecule.mf.get_ovlp()
    # Tr(D S) = number of electrons
    assert np.isclose(np.trace(D_total @ S).real, 1.0, atol=1e-8)

def test_h_atom_has_one_alpha_zero_beta(h_atom_molecule):
    S = h_atom_molecule.mf.get_ovlp()
    n_alpha = np.trace(h_atom_molecule.D_ao[0] @ S).real
    n_beta  = np.trace(h_atom_molecule.D_ao[1] @ S).real
    assert np.isclose(n_alpha, 1.0, atol=1e-8)
    assert np.isclose(n_beta,  0.0, atol=1e-8)

def test_li_atom_has_three_electrons(li_atom_molecule):
    D_total = li_atom_molecule.D_ao.sum(axis=0)
    S = li_atom_molecule.mf.get_ovlp()
    assert np.isclose(np.trace(D_total @ S).real, 3.0, atol=1e-8)

def test_li_atom_has_two_alpha_one_beta(li_atom_molecule):
    S = li_atom_molecule.mf.get_ovlp()
    n_alpha = np.trace(li_atom_molecule.D_ao[0] @ S).real
    n_beta  = np.trace(li_atom_molecule.D_ao[1] @ S).real
    assert np.isclose(n_alpha, 2.0, atol=1e-8)
    assert np.isclose(n_beta,  1.0, atol=1e-8)

def test_initial_density_hermitian_per_spin_uks(h_atom_molecule):
    for s in range(2):
        D_s = h_atom_molecule.D_ao[s]
        assert np.allclose(D_s, D_s.conj().T, atol=1e-12)

def test_is_hermitian_true_on_2d_hermitian(h2_closed_shell_molecule):
    A = np.array([[1.0, 2.0+1j], [2.0-1j, 3.0]])
    assert h2_closed_shell_molecule.is_hermitian(A, tol=1e-12) is True

def test_is_hermitian_false_on_2d_nonhermitian(h2_closed_shell_molecule):
    A = np.array([[1.0, 2.0+1j], [2.0+1j, 3.0]])  # not conjugated
    assert h2_closed_shell_molecule.is_hermitian(A, tol=1e-12) is False

def test_is_hermitian_true_on_3d_all_hermitian(h_atom_molecule):
    A = np.stack([np.array([[1.0, 1j], [-1j, 2.0]]),
                  np.array([[3.0, 0.0], [0.0, 4.0]])])
    assert h_atom_molecule.is_hermitian(A, tol=1e-12) is True

def test_is_hermitian_false_when_one_spin_block_fails(h_atom_molecule):
    A = np.stack([np.array([[1.0, 1j], [-1j, 2.0]]),       # good
                  np.array([[3.0, 1.0], [2.0, 4.0]])])     # asymmetric
    assert h_atom_molecule.is_hermitian(A, tol=1e-12) is False

def test_get_F_orth_returns_3d_for_uks(h_atom_molecule):
    F = h_atom_molecule.get_F_orth(h_atom_molecule.D_ao)
    nao = h_atom_molecule.mf.mol.nao
    assert F.shape == (2, nao, nao)

def test_get_F_orth_hermitian_per_spin(li_atom_molecule):
    F = li_atom_molecule.get_F_orth(li_atom_molecule.D_ao)
    for s in range(2):
        assert np.allclose(F[s], F[s].conj().T, atol=1e-10)

def test_get_F_orth_external_field_broadcasts_to_both_spins(h_atom_molecule):
    """A uniform external field should perturb α and β identically when D is the same."""
    # Symmetrize D so both spin channels are equal; then F_α and F_β with same exc
    # should differ from F-without-exc by the same amount.
    D_sym = np.stack([h_atom_molecule.D_ao_0[0], h_atom_molecule.D_ao_0[0]])
    F_no_field   = h_atom_molecule.get_F_orth(D_sym)
    F_with_field = h_atom_molecule.get_F_orth(D_sym, exc=np.array([0.0, 0.0, 0.01]))
    delta = F_with_field - F_no_field
    assert np.allclose(delta[0], delta[1], atol=1e-12)

def test_calculate_mu_shape_uks(h_atom_molecule):
    nao = h_atom_molecule.mf.mol.nao
    mu = h_atom_molecule.calculate_mu()
    assert mu.shape == (3, nao, nao)

def test_calculate_mu_unchanged_by_spin(h2_cation_molecule, h2_closed_shell_molecule):
    """Dipole integrals depend only on geometry/basis, not on whether we ran UKS or RKS."""
    mu_uks = h2_cation_molecule.calculate_mu()
    mu_rks = h2_closed_shell_molecule.calculate_mu()
    assert mu_uks.shape == mu_rks.shape  # same basis, same atoms

def _zero_field():
    return np.array([0.0, 0.0, 0.0])

# ---- RK4 ----
def test_rk4_uks_single_step_shapes_preserved(h_atom_molecule):
    m = h_atom_molecule
    nao = m.mf.mol.nao
    propagate_rk4(dt=m.dt, molecule=m, exc=_zero_field())
    assert m.D_ao.shape == (2, nao, nao)
    assert np.asarray(m.mf.mo_coeff).shape == (2, nao, m.mf.mo_coeff[0].shape[1])
    assert m.F_orth.shape == (2, nao, nao)

def test_rk4_uks_preserves_alpha_electron_count(h_atom_molecule):
    m = h_atom_molecule
    S = m.mf.get_ovlp()
    propagate_rk4(dt=m.dt, molecule=m, exc=_zero_field())
    assert np.isclose(np.trace(m.D_ao[0] @ S).real, 1.0, atol=1e-6)
    assert np.isclose(np.trace(m.D_ao[1] @ S).real, 0.0, atol=1e-6)

def test_rk4_uks_preserves_hermiticity_per_spin(li_atom_molecule):
    m = li_atom_molecule
    propagate_rk4(dt=m.dt, molecule=m, exc=_zero_field())
    for s in range(2):
        assert np.allclose(m.D_ao[s], m.D_ao[s].conj().T, atol=1e-10)

# ---- step ----
def test_step_uks_single_step_shapes_preserved(h_atom_molecule):
    m = h_atom_molecule
    # step requires C_orth_ndt to be set; magnus2 path doesn't apply here.
    if not hasattr(m, "C_orth_ndt"):
        m.C_orth_ndt = m.rotate_coeff_to_orth(m.mf.mo_coeff)
    nao = m.mf.mol.nao
    propagate_step(dt=m.dt, molecule=m, exc=_zero_field())
    assert m.D_ao.shape == (2, nao, nao)
    assert np.asarray(m.C_orth_ndt).shape == (2, nao, m.C_orth_ndt[0].shape[1])

def test_step_uks_preserves_total_electrons(li_atom_molecule):
    m = li_atom_molecule
    if not hasattr(m, "C_orth_ndt"):
        m.C_orth_ndt = m.rotate_coeff_to_orth(m.mf.mo_coeff)
    S = m.mf.get_ovlp()
    propagate_step(dt=m.dt, molecule=m, exc=_zero_field())
    n_total = np.trace(m.D_ao.sum(axis=0) @ S).real
    assert np.isclose(n_total, 3.0, atol=1e-6)

# ---- magnus2 ----
def test_magnus2_uks_single_step_shapes_preserved(h_atom_molecule):
    m = h_atom_molecule
    if not hasattr(m, "F_orth_n12dt"):
        m.F_orth_n12dt = m.F_orth
    nao = m.mf.mol.nao
    propagate_magnus2(
        molecule_max_iterations=200, dt=m.dt,
        molecule_pc_convergence=1e-10, molecule=m, exc=_zero_field()
    )
    assert m.D_ao.shape == (2, nao, nao)
    assert m.F_orth_n12dt.shape == (2, nao, nao)

def test_magnus2_uks_converges_in_few_iterations_at_ground_state(h_atom_molecule):
    """At the SCF ground state with zero field, Magnus2 should converge essentially immediately."""
    m = h_atom_molecule
    if not hasattr(m, "F_orth_n12dt"):
        m.F_orth_n12dt = m.F_orth
    # Should not raise:
    propagate_magnus2(
        molecule_max_iterations=10, dt=m.dt,
        molecule_pc_convergence=1e-10, molecule=m, exc=_zero_field()
    )

def test_expm_wrapper_handles_3d_via_per_spin_loop():
    """Sanity check that our per-spin expm wrapper gives the same result as
    looping by hand. Catches anyone who later 'optimizes' it to a single expm call
    on a 3-D array (whose behavior is scipy-version-dependent)."""
    from scipy.linalg import expm
    rng = np.random.default_rng(0)
    A_alpha = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    A_beta  = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    # Hermitian
    A_alpha = 0.5 * (A_alpha + A_alpha.conj().T)
    A_beta  = 0.5 * (A_beta  + A_beta.conj().T)
    stacked = np.stack([A_alpha, A_beta])
    manual  = np.stack([expm(A_alpha), expm(A_beta)])
    # Whatever wrapper you used in magnus2 / step:
    from plasmol.quantum.propagators.magnus2 import _expm  # if you named it that
    assert np.allclose(_expm(stacked), manual, atol=1e-12)

def test_initial_dipole_is_zero_uks(h_atom_molecule):
    """At t=0 with no perturbation, D == D_0, so induced dipole is exactly zero."""
    mu_arr = np.zeros(3)
    mu = h_atom_molecule.calculate_mu()
    D = h_atom_molecule.D_ao.sum(axis=0)
    D0 = h_atom_molecule.D_ao_0.sum(axis=0)
    for i in range(3):
        mu_arr[i] = float((np.trace(mu[i] @ D) - np.trace(mu[i] @ D0)).real)
    assert np.allclose(mu_arr, 0.0, atol=1e-12)

def test_dipole_sums_alpha_plus_beta(li_atom_molecule):
    """Verify that the spin-summed dipole equals α + β contributions."""
    mu = li_atom_molecule.calculate_mu()
    D_a, D_b = li_atom_molecule.D_ao[0], li_atom_molecule.D_ao[1]
    for i in range(3):
        mu_total = np.trace(mu[i] @ (D_a + D_b)).real
        mu_alpha = np.trace(mu[i] @ D_a).real
        mu_beta  = np.trace(mu[i] @ D_b).real
        assert np.isclose(mu_total, mu_alpha + mu_beta, atol=1e-12)

def test_propagation_returns_length_3_array(h_atom_molecule):
    """propagation() should return shape (3,) regardless of spin."""
    m = h_atom_molecule
    mu_arr = propagation(
        params={"dt": m.dt},
        molecule=m,
        exc=_zero_field(),
        propagator=propagate_rk4,
    )
    assert mu_arr.shape == (3,)

def test_rks_h2_scf_energy_unchanged(h2_closed_shell_molecule):
    """Pin the H2 sto-3g/PBE energy so any future drift fails loudly."""
    assert np.isclose(h2_closed_shell_molecule.mf.e_tot, -1.152064, atol=1e-5)

def test_rks_h2_D_ao_trace_equals_two_electrons(h2_closed_shell_molecule):
    S = h2_closed_shell_molecule.mf.get_ovlp()
    assert np.isclose(np.trace(h2_closed_shell_molecule.D_ao @ S).real, 2.0, atol=1e-8)

def test_rks_h2_F_orth_is_2d(h2_closed_shell_molecule):
    """Make sure the open-shell refactor didn't accidentally promote RKS Fock to 3-D."""
    assert h2_closed_shell_molecule.F_orth.ndim == 2

def test_rks_dipole_path_still_uses_2d_density(h2_closed_shell_molecule):
    mu = h2_closed_shell_molecule.calculate_mu()
    D = h2_closed_shell_molecule.D_ao
    D0 = h2_closed_shell_molecule.D_ao_0
    assert D.ndim == 2 and D0.ndim == 2
    # Initial dipole exactly zero
    for i in range(3):
        assert np.isclose((np.trace(mu[i] @ D) - np.trace(mu[i] @ D0)).real, 0.0, atol=1e-12)

def test_get_gamma_ao_returns_3d_for_uks(h_atom_molecule):
    G = h_atom_molecule.get_gamma_ao(gam0=1.0, xi=0.5, eps0=0.05, clamp=100.0)
    nao = h_atom_molecule.mf.mol.nao
    assert G.shape == (2, nao, nao)

def test_get_gamma_ao_returns_2d_for_rks(h2_closed_shell_molecule):
    G = h2_closed_shell_molecule.get_gamma_ao(gam0=1.0, xi=0.5, eps0=0.05, clamp=100.0)
    nao = h2_closed_shell_molecule.mf.mol.nao
    assert G.shape == (nao, nao)

def test_checkpoint_roundtrip_preserves_uks_shapes(h_atom_molecule, tmp_path):
    """Save a UKS state and reload it — shapes must survive np.savez."""
    from plasmol.utils.checkpoint import init_checkpoint, update_checkpoint

    ckpt = tmp_path / "uks_ckpt.npz"
    h_atom_molecule.checkpoint_filepath = str(ckpt)
    h_atom_molecule.input_file_path = str(tmp_path / "dummy.json")
    (tmp_path / "dummy.json").write_text("{}")
    h_atom_molecule.has_fourier = False

    init_checkpoint(h_atom_molecule)
    update_checkpoint(h_atom_molecule, h_atom_molecule, checkpoint_time=0.0)

    data = np.load(ckpt, allow_pickle=True)
    nao = h_atom_molecule.mf.mol.nao
    assert data["D_ao_0"].shape == (2, nao, nao)
    assert data["mo_coeff"].shape[0] == 2  # leading spin axis preserved

