"""End-to-end checkpoint: full run vs half-run + resume must match the dipole."""
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from plasmol.drivers.quantum import run as run_quantum
from plasmol.utils.checkpoint import init_checkpoint, restore_files_from_checkpoint
from plasmol.utils.csv import read_field_csv
from plasmol.utils.params import PARAMS

DT = 0.5
T_FULL = 4.0
T_HALF = 2.0


def _cfg(*, t_end, checkpoint=False):
    cfg = {
        "settings": {"dt": DT, "t_end": t_end},
        "molecule": {
            "geometry": [
                {"atom": "H", "coord": [0.0, 0.0, 0.0]},
                {"atom": "H", "coord": [0.0, 0.0, 1.4]},
            ],
            "geometry_units": "bohr",
            "charge": 0,
            "spin": 0,
            "basis": "sto3g",
            "xc": "pbe",
            "propagator": {"type": "rk4"},
            "source": {
                "type": "kick",
                "intensity": 0.001,
                "peak_time": 0.0,
                "width_steps": 1,
                "component": "z",
            },
        },
        "files": {
            "field_e_filepath": "field_e.csv",
            "field_p_filepath": "field_p.csv",
            "spectra_e_vs_p_filepath": "output.png",
        },
    }
    if checkpoint:
        cfg["files"]["checkpoint"] = {
            "filepath": "checkpoint.npz",
            "frequency_steps": 1,
        }
    return cfg


def _write_input(directory, cfg, name="input.json"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(cfg, indent=2))
    return path


def _arm_checkpointing(params):
    if not getattr(params, "has_checkpoint", False):
        return
    if params.resumed_from_checkpoint:
        params.checkpoint_filepath = (
            f"{Path(params.checkpoint_filepath).with_suffix('')}_new.npz"
        )
    init_checkpoint(params)
    params.checkpoint_written_after_init = False
    params.final_checkpoint_filepath = f"final-{params.checkpoint_filepath}"
    init_checkpoint(params, params.final_checkpoint_filepath)
    params.final_checkpoint_written_after_init = False


def _run_quantum(directory, json_path, monkeypatch):
    monkeypatch.chdir(directory)
    params = PARAMS(
        Namespace(input=str(json_path), verbose=0, log=None, checkpoint=None)
    )
    _arm_checkpointing(params)
    run_quantum(params)
    return params


def _load_p(directory):
    t, x, y, z = read_field_csv(str(directory / "field_p.csv"))
    return (
        np.asarray(t, dtype=float),
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
    )


def test_split_checkpoint_resume_matches_full_response(tmp_path, monkeypatch):
    """
    1) Full RT-TDDFT to t_end.
    2) Same input stopped at t_end/2 with checkpointing (interrupted-run stand-in).
    3) Resume from (2) to the original t_end.

    Concatenating (2) with the new tail of (3) must reproduce (1)'s dipole.
    """
    monkeypatch.setattr(
        "plasmol.drivers.quantum.plot_e_p_fields", lambda *a, **k: None
    )

    full_dir = tmp_path / "full"
    half_dir = tmp_path / "half"
    resume_dir = tmp_path / "resume"

    full_json = _write_input(full_dir, _cfg(t_end=T_FULL, checkpoint=False))
    half_json = _write_input(half_dir, _cfg(t_end=T_HALF, checkpoint=True))

    _run_quantum(full_dir, full_json, monkeypatch)
    _run_quantum(half_dir, half_json, monkeypatch)

    ckpt = half_dir / "final-checkpoint.npz"
    assert ckpt.is_file(), "half-length run must write final-checkpoint.npz"

    resume_dir.mkdir()
    monkeypatch.chdir(resume_dir)
    restored = restore_files_from_checkpoint(str(ckpt.resolve()))
    restored_json = Path(restored["restored_input_filepath"])
    payload = json.loads(restored_json.read_text())
    payload["settings"]["t_end"] = T_FULL
    restored_json.write_text(json.dumps(payload, indent=2))

    _run_quantum(resume_dir, restored_json, monkeypatch)

    t1, x1, y1, z1 = _load_p(full_dir)
    t2, x2, y2, z2 = _load_p(half_dir)
    t3, x3, y3, z3 = _load_p(resume_dir)

    assert t2.size >= 2
    assert t2[-1] == pytest.approx(T_HALF)
    assert t1[-1] == pytest.approx(T_FULL)
    assert t3[-1] == pytest.approx(T_FULL)

    join = t2[-1]
    tail = t3 > join + 1e-12
    t_cat = np.concatenate([t2, t3[tail]])
    x_cat = np.concatenate([x2, x3[tail]])
    y_cat = np.concatenate([y2, y3[tail]])
    z_cat = np.concatenate([z2, z3[tail]])

    assert t_cat.shape == t1.shape, (
        f"concat time grid {t_cat.shape} != full run {t1.shape}"
    )
    np.testing.assert_allclose(t_cat, t1, atol=1e-12)
    np.testing.assert_allclose(x_cat, x1, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(y_cat, y1, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(z_cat, z1, rtol=1e-6, atol=1e-10)
    # The resumed CSV should already be the full trajectory.
    np.testing.assert_allclose(z3, z1, rtol=1e-6, atol=1e-10)
    # Half-run prefix must be a real (non-trivial) piece of the full dipole.
    assert np.linalg.norm(z1) > 0
    assert np.linalg.norm(z2) > 0
