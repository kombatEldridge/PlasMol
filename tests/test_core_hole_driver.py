"""Orchestration tests for the core_hole survey driver."""
from types import SimpleNamespace
from unittest.mock import MagicMock

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
