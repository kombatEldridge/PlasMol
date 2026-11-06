"""Orchestration tests for the core_hole survey driver."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from plasmol.drivers.custom_drivers import core_hole as core_hole_mod
from plasmol.drivers import get_driver


def test_get_driver_core_hole():
    fn = get_driver("core_hole")
    assert fn is core_hole_mod.run


def test_get_driver_unknown():
    with pytest.raises(ValueError, match="Unknown driver"):
        get_driver("not_a_real_driver")


def test_get_driver_no_legacy_dch():
    with pytest.raises(ValueError, match="Unknown driver"):
        get_driver("dch")


def test_survey_always_runs_and_skips_propagation(monkeypatch):
    params = SimpleNamespace(
        has_core_hole=True,
        mo_removal_index_dict={0: 2, 1: 2},
    )
    fake_mol = MagicMock()
    fake_mol.mf.mo_coeff = __import__("numpy").zeros((3, 3))
    contrib = MagicMock()
    monkeypatch.setattr(core_hole_mod, "MOLECULE", MagicMock(return_value=fake_mol))
    monkeypatch.setattr(core_hole_mod, "_mo_atom_contribution", contrib)
    core_hole_mod.run(params)
    assert params.has_core_hole is False
    assert contrib.call_count == 2
    contrib.assert_any_call(fake_mol, 0)
    contrib.assert_any_call(fake_mol, 1)


def _molecule_with_populations(weights, labels):
    weights = np.asarray(weights, dtype=float)
    # c = 1 and S = diag(w) makes the Mulliken weight c (S c) equal w,
    # including negative components.
    return SimpleNamespace(
        is_open_shell=False,
        mf=SimpleNamespace(mo_coeff=np.ones((weights.size, 1))),
        S=np.diag(weights),
        mol=SimpleNamespace(ao_labels=lambda fmt=False: labels),
    )


def test_shell_contributions_keep_atom_index_and_sum_components():
    labels = [
        (0, "C", "1s", ""),
        (0, "C", "2s", ""),
        (0, "C", "2p", "x"),
        (0, "C", "2p", "y"),
        (0, "C", "2p", "z"),
        (1, "O", "1s", ""),
        (1, "O", "2p", "x"),
        (1, "O", "2p", "y"),
        (1, "O", "2p", "z"),
    ]
    weights = [0.50, 0.005, 0.10, 0.10, -0.04, 0.20, 0.004, 0.003, 0.002]
    molecule = _molecule_with_populations(weights, labels)
    rows = core_hole_mod._mo_atom_contribution(molecule, 0)
    assert rows == [
        "0\t0\tC\t1s\t50.00",
        "0\t1\tO\t1s\t20.00",
        "0\t0\tC\t2p\t16.00",
    ]


def test_open_shell_survey_uses_alpha_coefficients():
    labels = [(0, "Na", "1s", ""), (0, "Na", "2s", "")]
    alpha = np.array([[1.0], [0.0]])
    beta = np.array([[0.0], [1.0]])
    molecule = SimpleNamespace(
        is_open_shell=True,
        mf=SimpleNamespace(mo_coeff=np.stack([alpha, beta])),
        S=np.eye(2),
        mol=SimpleNamespace(ao_labels=lambda fmt=False: labels),
    )
    rows = core_hole_mod._mo_atom_contribution(molecule, 0)
    assert rows == ["0\t0\tNa\t1s\t100.00"]


def test_survey_file_lists_atom_index_and_shell(monkeypatch, tmp_path):
    labels = [(0, "Na", "1s", ""), (0, "Na", "2s", "")]
    molecule = _molecule_with_populations([1.0, 0.0], labels)
    monkeypatch.setattr(core_hole_mod, "MOLECULE", lambda params: molecule)
    monkeypatch.setattr(core_hole_mod, "SURVEY_FILE", str(tmp_path / "mo_survey.txt"))
    params = SimpleNamespace(has_core_hole=True, mo_removal_index_dict={0: 2})
    core_hole_mod.run(params)
    text = (tmp_path / "mo_survey.txt").read_text()
    assert text.splitlines() == [
        "mo_index\tatom_index\telement\tshell\tcontribution_percent",
        "0\t0\tNa\t1s\t100.00",
    ]


def test_survey_rejects_label_count_mismatch():
    molecule = _molecule_with_populations(
        [1.0, 0.0],
        [(0, "Na", "1s", "")],
    )
    with pytest.raises(ValueError, match="AO label count"):
        core_hole_mod._mo_atom_contribution(molecule, 0)


def test_survey_rejects_out_of_range_mo(monkeypatch):
    params = SimpleNamespace(
        has_core_hole=False,
        mo_removal_index_dict={5: 2},
    )
    fake_mol = MagicMock()
    fake_mol.mf.mo_coeff = __import__("numpy").zeros((2, 2))
    monkeypatch.setattr(core_hole_mod, "MOLECULE", MagicMock(return_value=fake_mol))
    with pytest.raises(ValueError, match="out of range"):
        core_hole_mod.run(params)
