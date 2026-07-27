# Time-series I/O, folding, damping, and reference E_inc files.
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from plasmol.utils.csv import init_csv, update_csv, read_field_csv

logger = logging.getLogger("main")


def apply_damping(mu_arrs, tau):
    """
    Apply damping to the polarizability array.

    mu_damped = mu * exp(-t/tau).
    """
    t = np.array(mu_arrs[0])
    damped_mu_x = mu_arrs[1] * np.exp(-t / tau)
    damped_mu_y = mu_arrs[2] * np.exp(-t / tau)
    damped_mu_z = mu_arrs[3] * np.exp(-t / tau)
    return damped_mu_x, damped_mu_y, damped_mu_z


def fold(file_x, file_y, file_z):
    """Fold diagonal components: X from x-run, Y from y-run, Z from z-run."""
    def read_component(filename, column):
        df = pd.read_csv(filename, delimiter=',', header=0, comment='#')
        time = np.array(df['Timestamps (au)'].values, dtype=float)
        values = np.array(df[column].values, dtype=float)
        logger.debug(f"Loaded {len(time)} points from {filename}")
        return time, values

    tx, dx = read_component(file_x, 'X Values')
    ty, dy = read_component(file_y, 'Y Values')
    tz, dz = read_component(file_z, 'Z Values')

    dtx = tx[1] - tx[0]
    dty = ty[1] - ty[0]
    dtz = tz[1] - tz[0]
    if not (np.isclose(dtx, dty) and np.isclose(dtx, dtz)):
        raise ValueError("Inconsistent timesteps across files!")

    min_length = min(len(tx), len(ty), len(tz))
    time_points = tx[:min_length]
    dx = dx[:min_length]
    dy = dy[:min_length]
    dz = dz[:min_length]
    stacked = np.vstack([dx, dy, dz])
    return time_points, stacked


def load_dipole_from_csv(file_path):
    """Load x/y/z dipole components from a single field CSV."""
    time_values, dx, dy, dz = read_field_csv(file_path)
    if len(time_values) < 2:
        raise ValueError(
            f"No induced dipole data found in {file_path}. "
            "Check that the plasmon field exceeds plasmon_tolerance_field_e and that t_end is long enough."
        )
    time_points = np.array(time_values, dtype=float)
    dipole_moment = np.vstack([dx, dy, dz])
    return time_points, dipole_moment


def read_xyz_field_csv(filepath):
    """Read a standard field CSV (Timestamps, X, Y, Z) → time (N,), data (3, N)."""
    df = pd.read_csv(filepath, delimiter=',', header=0, comment='#')
    if 'Timestamps (au)' not in df.columns:
        raise ValueError(f"Expected 'Timestamps (au)' column in {filepath}; got {list(df.columns)}")
    for col in ('X Values', 'Y Values', 'Z Values'):
        if col not in df.columns:
            raise ValueError(f"Expected '{col}' column in {filepath}; got {list(df.columns)}")
    time = df['Timestamps (au)'].to_numpy(dtype=float)
    data = np.vstack([
        df['X Values'].to_numpy(dtype=float),
        df['Y Values'].to_numpy(dtype=float),
        df['Z Values'].to_numpy(dtype=float),
    ])
    return time, data


def merge_reference_e_fields(file_x, file_y, file_z, output_filepath):
    """
    Merge three vacuum reference field CSVs into one E_inc CSV.

    Columns: time, xx, yy, zz
    where xx is Ex under x-polarized incidence (and yy, zz analogously).
    """
    tx, ex = read_xyz_field_csv(file_x)
    ty, ey = read_xyz_field_csv(file_y)
    tz, ez = read_xyz_field_csv(file_z)

    dtx, dty, dtz = tx[1] - tx[0], ty[1] - ty[0], tz[1] - tz[0]
    if not (np.isclose(dtx, dty) and np.isclose(dtx, dtz)):
        raise ValueError("Inconsistent timesteps across reference E-field files!")

    n = min(len(tx), len(ty), len(tz))
    if n < 2:
        raise ValueError("Reference E-field series too short to merge.")

    out = pd.DataFrame({
        'time': tx[:n],
        'xx': ex[0, :n],
        'yy': ey[1, :n],
        'zz': ez[2, :n],
    })
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath, 'w', newline='') as f:
        f.write(
            "# Vacuum reference incident electric field (atomic units).\n"
            "# Columns: time, xx, yy, zz — Ex/Ey/Ez at the sample point for "
            "x/y/z-polarized vacuum incidence.\n"
        )
        out.to_csv(f, index=False)
    logger.info(f"Merged vacuum reference E_inc written to {output_filepath}")
    return output_filepath


def load_reference_e_tensor(filepath):
    """
    Load a merged vacuum E_inc CSV (time, xx, yy, zz).

    Returns
    -------
    time : (N,) ndarray
    field_diag : (3, N) ndarray
        (xx, yy, zz) used for μ_i / E_ii deconvolution.
    """
    df = pd.read_csv(filepath, delimiter=',', header=0, comment='#')
    if 'time' in df.columns:
        time_col = 'time'
    elif 'Timestamps (au)' in df.columns:
        time_col = 'Timestamps (au)'
    else:
        raise ValueError(
            f"Reference file '{filepath}' must have a 'time' column; got {list(df.columns)}"
        )
    REF_E_COLUMNS = ['time', 'xx', 'yy', 'zz']
    missing = [c for c in ('xx', 'yy', 'zz') if c not in df.columns]
    if missing:
        raise ValueError(
            f"Reference file '{filepath}' missing columns {missing}. "
            f"Expected columns: {REF_E_COLUMNS}"
        )
    time = df[time_col].to_numpy(dtype=float)
    field_diag = np.vstack([
        df['xx'].to_numpy(dtype=float),
        df['yy'].to_numpy(dtype=float),
        df['zz'].to_numpy(dtype=float),
    ])
    if len(time) < 2:
        raise ValueError(f"Reference file '{filepath}' has fewer than 2 time samples.")
    logger.info(f"Loaded vacuum reference E_inc from {filepath} ({len(time)} samples)")
    return time, field_diag


def validate_reference_times(ref_time, expected_times, atol=1e-5, filepath=None):
    """Require every reference timestamp to match params.times (same length, |Δt| ≤ atol)."""
    ref_time = np.asarray(ref_time, dtype=float)
    expected_times = np.asarray(expected_times, dtype=float)
    src = f" '{filepath}'" if filepath else ""

    if expected_times.size < 2:
        raise ValueError(
            "params.times is missing or too short to validate the reference E_inc time grid."
        )
    if ref_time.size != expected_times.size:
        raise ValueError(
            f"Reference E_inc{src} time grid length ({ref_time.size}) does not match "
            f"params.times ({expected_times.size}). "
            f"Reference range [{ref_time[0]}, {ref_time[-1]}], "
            f"params.times range [{expected_times[0]}, {expected_times[-1]}]. "
            f"Regenerate the reference with the same dt/t_end (and Meep-adjusted timestep)."
        )

    diff = np.abs(ref_time - expected_times)
    bad = np.flatnonzero(diff > atol)
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"Reference E_inc{src} time mismatch at index {i}: "
            f"reference={ref_time[i]!r}, params.times={expected_times[i]!r}, "
            f"|diff|={diff[i]!r} > atol={atol}. "
            f"({bad.size} / {ref_time.size} samples exceed tolerance.)"
        )
    logger.info(
        f"Reference E_inc time grid matches params.times "
        f"({ref_time.size} samples, max |Δt|={float(diff.max()):.3e} ≤ {atol})."
    )


def apply_tau_damping_arrays(time, arrays_3n, tau):
    """Apply exp(-t/tau) to each row of a (3, N) array. Returns damped copy."""
    if np.isclose(tau, 0):
        return arrays_3n
    window = np.exp(-np.asarray(time, dtype=float) / tau)
    return np.asarray(arrays_3n, dtype=float) * window


def apply_tau_damping(field_filepath, tau, time_rounding_decimals, label="polarizability"):
    if np.isclose(tau, 0):
        return field_filepath

    arrs = read_field_csv(field_filepath)
    damped_x, damped_y, damped_z = apply_damping(arrs, tau)
    damped_filepath = f"{Path(field_filepath).with_suffix('')}_damped.csv"
    init_csv(
        damped_filepath,
        f"# {label} field in atomic units, damped with "
        f"signal_damped = signal * exp(-t/tau) where tau = {tau}",
    )
    for t, x, y, z in zip(arrs[0], damped_x, damped_y, damped_z):
        update_csv(damped_filepath, round(t, time_rounding_decimals), x, y, z)
    logger.info(f"Damped {label} field written to {damped_filepath}")
    return damped_filepath


def fold_single(field_p_filepath, component):
    """
    Load co-polarized dipole μ_i(t) from one directional field_p CSV.

    Returns
    -------
    time : (N,)
    dipole : (3, N) with only the active Cartesian row filled (others zero).
    """
    component = component.lower().strip()
    col = {'x': 'X Values', 'y': 'Y Values', 'z': 'Z Values'}.get(component)
    if col is None:
        raise ValueError(f"Invalid component '{component}' for fold_single.")
    df = pd.read_csv(field_p_filepath, delimiter=',', header=0, comment='#')
    if 'Timestamps (au)' not in df.columns or col not in df.columns:
        raise ValueError(
            f"Expected 'Timestamps (au)' and '{col}' in {field_p_filepath}; "
            f"got {list(df.columns)}"
        )
    time = df['Timestamps (au)'].to_numpy(dtype=float)
    if len(time) < 2:
        raise ValueError(
            f"No induced dipole data found in {field_p_filepath}. "
            "Check plasmon_tolerance_field_e and t_end."
        )
    values = df[col].to_numpy(dtype=float)
    dipole = np.zeros((3, len(time)), dtype=float)
    dipole[{'x': 0, 'y': 1, 'z': 2}[component]] = values
    return time, dipole


def write_single_reference_e_field(ref_csv_filepath, component, output_filepath):
    """
    Write a vacuum E_inc CSV (time, xx, yy, zz) from one directional reference run.

    Only the co-polarized diagonal column is filled; the other two are zero so
    the file remains compatible with ``load_reference_e_tensor``.
    """
    component = component.lower().strip()
    axis = {'x': 0, 'y': 1, 'z': 2}.get(component)
    if axis is None:
        raise ValueError(f"Invalid component '{component}'.")
    time, data = read_xyz_field_csv(ref_csv_filepath)
    if len(time) < 2:
        raise ValueError(f"Reference E-field series too short: {ref_csv_filepath}")
    cols = {'xx': np.zeros_like(time), 'yy': np.zeros_like(time), 'zz': np.zeros_like(time)}
    key = ('xx', 'yy', 'zz')[axis]
    cols[key] = data[axis]
    out = pd.DataFrame({'time': time, **cols})
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath, 'w', newline='') as f:
        f.write(
            f"# Vacuum reference incident electric field (atomic units), single polarization '{component}'.\n"
            "# Columns: time, xx, yy, zz — only the co-polarized diagonal is non-zero.\n"
        )
        out.to_csv(f, index=False)
    logger.info(
        f"Single-pol vacuum reference E_inc ('{component}') written to {output_filepath}"
    )
    return output_filepath


def align_series(time_a, arr_a, time_b, arr_b, time_rounding_decimals, label_a="dipole", label_b="field"):
    """Trim two (3, N) series to a common length; warn on timestamp mismatch."""
    n = min(len(time_a), len(time_b))
    if n < 2:
        raise ValueError(
            f"Insufficient time-series data to align {label_a} and {label_b} "
            f"({len(time_a)} vs {len(time_b)} samples)."
        )
    if len(time_a) != len(time_b):
        logger.warning(
            f"{label_a} and {label_b} series lengths differ "
            f"({len(time_a)} vs {len(time_b)}); trimming to {n} samples."
        )
    time_a = time_a[:n]
    arr_a = arr_a[:, :n]
    time_b = time_b[:n]
    arr_b = arr_b[:, :n]
    atol = 10 ** (-time_rounding_decimals)
    if not np.allclose(time_a, time_b, rtol=0, atol=atol):
        logger.warning(
            f"{label_a} and {label_b} timestamps are not identical after fold; "
            f"using {label_a} timestamps for the FFT grid."
        )
    return time_a, arr_a, arr_b
