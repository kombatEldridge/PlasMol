"""Unit tests for parallel / perpendicular absorption-driver helpers (no full Meep run)."""
import logging
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from plasmol.drivers.custom_drivers.absorption import (
    absorption_single,
    build_parallel_abs_spec_runs,
    build_perpendicular_abs_spec_runs,
    build_single_abs_spec_runs,
    ensure_transverse_plane_wave_source,
    fold_single,
    np_mol_axis_vector,
    resolve_parallel_component,
    resolve_perpendicular_component,
    source_face_normal_index,
    write_single_reference_e_field,
)
from plasmol.drivers.custom_drivers.absorption.setup import set_up_params_copy_plasmol


def _base_params(**kwargs):
    p = SimpleNamespace(
        xyz=["x", "y", "z"],
        plasmol_molecule_position=[0.03, 0.0, 0.0],
        nanoparticle_center=[0.0, 0.0, 0.0],
        has_nanoparticle=True,
        has_molecule_position=True,
        absorption_perp_component=None,
        absorption_use_existing_e_field_ref=False,
        plasmon_source_component="z",
        plasmon_source_center=[-0.04, 0.0, 0.0],
        plasmon_source_size=[0.0, 0.2, 0.2],
        plasmon_cell_length=0.2,
        plasmon_pml_thickness=0.05,
        field_e_x_filepath="x_dir/field_e.csv",
        field_e_y_filepath="y_dir/field_e.csv",
        field_e_z_filepath="z_dir/field_e.csv",
        field_p_x_filepath="x_dir/field_p.csv",
        field_p_y_filepath="y_dir/field_p.csv",
        field_p_z_filepath="z_dir/field_p.csv",
        spectra_e_x_vs_p_x_filepath="x_dir/out.png",
        spectra_e_y_vs_p_y_filepath="y_dir/out.png",
        spectra_e_z_vs_p_z_filepath="z_dir/out.png",
    )
    for k, v in kwargs.items():
        setattr(p, k, v)
    return p


def test_np_mol_axis_vector_and_parallel_on_x():
    p = _base_params()
    axis, norm = np_mol_axis_vector(p)
    assert np.allclose(axis, [0.03, 0, 0])
    assert resolve_parallel_component(p) == "x"


def test_perpendicular_defaults_to_y_for_x_axis():
    p = _base_params()
    assert resolve_perpendicular_component(p) == "y"


def test_perpendicular_user_override():
    p = _base_params(absorption_perp_component="z")
    assert resolve_perpendicular_component(p) == "z"


def test_perpendicular_user_override_rejects_parallel_axis():
    p = _base_params(absorption_perp_component="x")
    with pytest.raises(ValueError, match="not perpendicular"):
        resolve_perpendicular_component(p)


def test_build_parallel_one_prod_one_ref(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = _base_params()
    prod, ref, comp = build_parallel_abs_spec_runs(p)
    assert comp == "x"
    assert len(prod) == 1 and len(ref) == 1
    assert prod[0].plasmon_source_component == "x"
    assert prod[0].dir_path == "fields"
    assert prod[0].field_e_filepath == "fields/field_e.csv"
    assert prod[0].field_p_filepath == "fields/field_p.csv"
    assert ref[0].record_field_only is True
    assert ref[0].has_nanoparticle is False
    assert ref[0].has_molecule is False
    assert ref[0].dir_path == "fields"
    assert ref[0].field_e_filepath == "fields/field_e_ref.csv"
    assert os.path.isdir("fields")
    assert not os.path.isdir("x_dir")


def test_full_mode_still_uses_component_subdirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    copies = set_up_params_copy_plasmol(_base_params())
    assert {c.dir_path for c in copies} == {"x_dir", "y_dir", "z_dir"}
    assert all(os.path.isdir(c.dir_path) for c in copies)
    by_comp = {c.plasmon_source_component: c for c in copies}
    assert by_comp["x"].field_e_filepath == "x_dir/field_e.csv"


def test_build_parallel_skips_ref_when_existing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = _base_params(absorption_use_existing_e_field_ref=True)
    prod, ref, comp = build_parallel_abs_spec_runs(p)
    assert comp == "x"
    assert len(prod) == 1 and len(ref) == 0


def test_build_single_keeps_json_component(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = _base_params()
    prod, ref, comp = build_single_abs_spec_runs(p)
    assert comp == "z"
    assert prod[0].plasmon_source_component == "z"
    assert prod[0].plasmon_source_size == [0.0, 0.2, 0.2]
    assert prod[0].plasmon_source_center == [-0.04, 0.0, 0.0]
    assert prod[0].dir_path == "fields"
    assert prod[0].field_e_filepath == "fields/field_e.csv"
    assert ref[0].plasmon_source_component == "z"
    assert ref[0].field_e_filepath == "fields/field_e_ref.csv"
    assert os.path.isdir("fields")
    assert not os.path.isdir("z_dir")
    assert not os.path.isdir("x_dir")


def test_build_single_requires_component():
    p = _base_params(plasmon_source_component=None)
    with pytest.raises(ValueError, match="JSON source"):
        build_single_abs_spec_runs(p)


def test_build_perpendicular_one_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = _base_params()
    prod, ref, comp = build_perpendicular_abs_spec_runs(p)
    assert comp == "y"
    assert len(prod) == 1 and len(ref) == 1
    assert prod[0].plasmon_source_component == "y"
    assert prod[0].dir_path == "fields"
    assert ref[0].field_e_filepath == "fields/field_e_ref.csv"
    assert os.path.isdir("fields")
    assert not os.path.isdir("y_dir")


def test_fold_single_and_write_ref(tmp_path):
    p_csv = tmp_path / "field_p.csv"
    e_csv = tmp_path / "field_e_ref_raw.csv"
    t = np.linspace(0.1, 1.0, 10)
    # production dipole
    pd.DataFrame({
        "Timestamps (au)": t,
        "X Values": np.sin(t),
        "Y Values": np.zeros_like(t),
        "Z Values": np.zeros_like(t),
    }).to_csv(p_csv, index=False)
    # vacuum E
    pd.DataFrame({
        "Timestamps (au)": t,
        "X Values": np.cos(t),
        "Y Values": np.zeros_like(t),
        "Z Values": np.zeros_like(t),
    }).to_csv(e_csv, index=False)

    time, dip = fold_single(str(p_csv), "x")
    assert dip.shape == (3, 10)
    assert np.allclose(dip[0], np.sin(t))
    assert np.allclose(dip[1], 0) and np.allclose(dip[2], 0)

    out = tmp_path / "field_e_ref.csv"
    write_single_reference_e_field(str(e_csv), "x", str(out))
    df = pd.read_csv(out, comment="#")
    assert list(df.columns) == ["time", "xx", "yy", "zz"]
    assert np.allclose(df["xx"], np.cos(t))
    assert np.allclose(df["yy"], 0) and np.allclose(df["zz"], 0)


def test_absorption_single_shape():
    freqs = np.array([1.0, 2.0, 3.0])
    imag = np.array([-0.1, -0.2, -0.3])
    a = absorption_single(imag, freqs)
    assert a.shape == (3,)
    assert np.all(a > 0)


def test_source_face_normal_index():
    assert source_face_normal_index([0, 0.2, 0.2]) == 0
    assert source_face_normal_index([0.2, 0, 0.2]) == 1
    assert source_face_normal_index([0.2, 0.2, 0]) == 2
    assert source_face_normal_index([0.2, 0.2, 0.2]) is None


def test_ensure_keeps_transverse_face():
    """E || z, face normal x → already transverse; keep size/center."""
    p = _base_params(
        plasmon_source_component="z",
        plasmon_source_center=[-0.04, 0.0, 0.0],
        plasmon_source_size=[0.0, 0.2, 0.2],
    )
    info = ensure_transverse_plane_wave_source(p, component="z")
    assert info["kept"] is True
    assert info["modified"] is False
    assert info["k_component"] == "x"
    assert p.plasmon_source_size == [0.0, 0.2, 0.2]
    assert p.plasmon_source_center == [-0.04, 0.0, 0.0]


def test_ensure_rebuilds_longitudinal_face_by_rearrange():
    """E || x with face normal x → swap slots: [0,0.2,0.2]→[0.2,0,0.2], center too."""
    p = _base_params(
        plasmon_source_component="x",
        plasmon_source_center=[-0.04, 0.0, 0.0],
        plasmon_source_size=[0.0, 0.2, 0.2],
    )
    info = ensure_transverse_plane_wave_source(p, component="x")
    assert info["modified"] is True
    assert info["kept"] is False
    assert info["component"] == "x"
    # Default pick among {y,z} with no center offset → y
    assert info["k_component"] == "y"
    assert p.plasmon_source_size == [0.2, 0.0, 0.2]
    assert p.plasmon_source_center == [0.0, -0.04, 0.0]


def test_ensure_prefer_k_rearranges_to_z():
    p = _base_params(
        plasmon_source_component="x",
        plasmon_source_center=[-0.04, 0.0, 0.0],
        plasmon_source_size=[0.0, 0.2, 0.2],  # longitudinal for E||x
    )
    info = ensure_transverse_plane_wave_source(p, component="x", prefer_k="z")
    assert info["k_component"] == "z"
    assert p.plasmon_source_size == [0.2, 0.2, 0.0]
    assert p.plasmon_source_center == [0.0, 0.0, -0.04]


def test_ensure_nonplanar_raises():
    p = _base_params(
        plasmon_source_component="x",
        plasmon_source_size=[0.2, 0.2, 0.2],
    )
    with pytest.raises(ValueError, match="not a clear planar face"):
        ensure_transverse_plane_wave_source(p, component="x")


def test_parallel_builder_defers_face_rearrange(tmp_path, monkeypatch):
    """Parent copies keep the user face; workers rearrange independently to the same k ⊥ E."""
    monkeypatch.chdir(tmp_path)
    user_size = [0.0, 0.2, 0.2]
    user_center = [-0.04, 0.0, 0.0]
    p = _base_params(
        # typical user source: k||x, E||z — parallel mode switches E to x
        plasmon_source_component="z",
        plasmon_source_center=user_center,
        plasmon_source_size=user_size,
    )
    prod, ref, comp = build_parallel_abs_spec_runs(p)
    assert comp == "x"
    assert prod[0].plasmon_source_component == "x"
    assert ref[0].plasmon_source_component == "x"
    # still the JSON face (longitudinal for E||x) until the Meep worker runs
    assert prod[0].plasmon_source_size == user_size
    assert prod[0].plasmon_source_center == user_center
    assert ref[0].plasmon_source_size == user_size
    assert ref[0].plasmon_source_center == user_center

    info_prod = ensure_transverse_plane_wave_source(prod[0], component="x")
    info_ref = ensure_transverse_plane_wave_source(ref[0], component="x")
    assert info_prod["modified"] is True and info_ref["modified"] is True
    assert info_prod["k_component"] == info_ref["k_component"]
    k = source_face_normal_index(prod[0].plasmon_source_size)
    assert k is not None and k != 0
    assert source_face_normal_index(ref[0].plasmon_source_size) == k
    assert prod[0].plasmon_source_size == ref[0].plasmon_source_size
    assert prod[0].plasmon_source_center == ref[0].plasmon_source_center


def test_direction_prefix_tags_rearrange_logs(caplog):
    """Worker prefix is installed before the face rearrange, so logs are tagged."""
    from plasmol.drivers.custom_drivers.absorption.workers import direction_log_prefix

    p = _base_params(
        plasmon_source_component="x",
        plasmon_source_center=[-0.04, 0.0, 0.0],
        plasmon_source_size=[0.0, 0.2, 0.2],
    )
    caplog.set_level(logging.INFO, logger="main")
    with direction_log_prefix("ref-x"):
        ensure_transverse_plane_wave_source(p, component="x")
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any(
        m.startswith("[ref-x-dir] ") and "longitudinal" in m for m in msgs
    ), msgs
    assert any(
        m.startswith("[ref-x-dir] ") and "Rearranged plane-wave source" in m
        for m in msgs
    ), msgs
