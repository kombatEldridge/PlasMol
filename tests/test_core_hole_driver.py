from pathlib import Path
"""Orchestration tests for core_hole driver (mocked quantum / plot)."""
import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


def test_survey_mode_skips_quantum(tmp_path, monkeypatch):
    params = SimpleNamespace(
        check_mo_contrib_by_atom=True,
        has_core_hole=True,
        mo_removal_index_dict={0: 2},
        core_hole_mo_occ_filepath=str(tmp_path / "mo.csv"),
    )
    fake_mol = MagicMock()
    fake_mol.mf.mo_coeff = __import__("numpy").zeros((2, 2))
    run_q = MagicMock()
    monkeypatch.setattr(core_hole_mod, "MOLECULE", MagicMock(return_value=fake_mol))
    monkeypatch.setattr(core_hole_mod, "run_quantum", run_q)
    monkeypatch.setattr(core_hole_mod, "_mo_atom_contribution", MagicMock())
    core_hole_mod.run(params)
    run_q.assert_not_called()
    assert params.has_core_hole is False  # survey disables hole path for SCF build


def test_run_calls_quantum_and_plot(tmp_path, monkeypatch):
    occ = str(tmp_path / "mo_occ.csv")
    Path(occ).write_text("Timestamps (au),MO index 0\n0.0,2.0\n")
    params = SimpleNamespace(
        check_mo_contrib_by_atom=False,
        has_checkpoint=False,
        core_hole_mo_occ_filepath=occ,
        core_hole_watch_indices=[0],
        core_hole_filter_by_amplitude=False,
        core_hole_amplitude_threshold=0.2,
    )
    run_q = MagicMock()
    plot = MagicMock()
    monkeypatch.setattr(core_hole_mod, "run_quantum", run_q)
    monkeypatch.setattr(core_hole_mod, "plot_core_hole_mo_occupations", plot)
    core_hole_mod.run(params)
    run_q.assert_called_once_with(params)
    plot.assert_called_once()
    kwargs = plot.call_args
    assert kwargs[0][0] == occ
