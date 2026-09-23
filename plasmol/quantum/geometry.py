# quantum/geometry.py
"""Rotate the nuclear frame before SCF.

``molecule.rotation`` is applied to the Bohr coordinates already stored on
``params.molecule_coords``. The source ``.xyz`` is not modified. A copy of the
rotated frame is written to ``rotated_geometry.xyz`` in the working directory.
"""
import json
import logging
from pathlib import Path

import numpy as np
from pyscf import gto

logger = logging.getLogger("main")

_AXIS_NAMES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
_ROTATED_XYZ_NAME = "rotated_geometry.xyz"


def rotate_molecule_geometry(params):
    """Apply ``molecule.rotation`` once, before any SCF or AO-matrix setup.

    Rotation is about the nuclear-charge center (the molecule does not
    translate). Each step is an active right-handed rotation. A list of steps
    is applied in order; a later ``align`` sees the frame left by earlier steps.
    Axis-angle axes are fixed lab axes, not molecule-fixed axes.
    """
    if getattr(params, "molecule_rotation_applied", False):
        return
    rotation = getattr(params, "molecule_rotation", None)
    if not rotation:
        return

    steps = normalize_rotation(rotation)
    atoms, coords = _parse_coords(params.molecule_coords)
    charges = np.array([float(gto.charge(atom)) for atom in atoms], dtype=float)
    if charges.sum() == 0:
        center = coords.mean(axis=0)
    else:
        center = (charges[:, None] * coords).sum(axis=0) / charges.sum()

    for step in steps:
        matrix = _step_matrix(step, coords)
        coords = (matrix @ (coords - center).T).T + center

    params.molecule_atoms = list(atoms)
    params.molecule_coords = "; ".join(
        f"{atom} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}"
        for atom, xyz in zip(atoms, coords)
    )
    path = Path.cwd() / _ROTATED_XYZ_NAME
    _write_xyz(path, atoms, coords, steps)
    params.molecule_rotated_xyz = str(path.resolve())
    params.molecule_rotation_applied = True
    logger.info(
        "Rotated the nuclear frame about the nuclear-charge center before SCF. "
        f"Wrote {params.molecule_rotated_xyz}."
    )


def normalize_rotation(rotation):
    """Return validated rotation steps. Raises ``ValueError`` on a bad spec."""
    if isinstance(rotation, dict):
        raw_steps = [rotation]
    elif isinstance(rotation, list):
        raw_steps = rotation
    else:
        raise ValueError(
            "molecule.rotation must be an object or a list of objects, "
            f"got {type(rotation).__name__}."
        )
    if len(raw_steps) == 0:
        raise ValueError("molecule.rotation is empty.")

    steps = []
    for index, step in enumerate(raw_steps):
        if not isinstance(step, dict):
            raise ValueError(
                f"molecule.rotation[{index}] must be an object, got {type(step).__name__}."
            )
        has_align = "align" in step
        has_axis_angle = "axis" in step or "angle_deg" in step
        if has_align and has_axis_angle:
            raise ValueError(
                f"molecule.rotation[{index}] cannot mix 'align' with 'axis'/'angle_deg'. "
                "Use 'twist_deg' for a rotation about the aligned axis."
            )
        if "twist_deg" in step and not has_align:
            raise ValueError(
                f"molecule.rotation[{index}] uses 'twist_deg' without 'align'."
            )
        unknown = set(step) - {"axis", "angle_deg", "align", "twist_deg"}
        if unknown:
            raise ValueError(
                f"molecule.rotation[{index}] has unknown keys: {sorted(unknown)}."
            )
        if has_align:
            steps.append(_normalize_align(step, index))
        else:
            if "axis" not in step or "angle_deg" not in step:
                raise ValueError(
                    f"molecule.rotation[{index}] needs 'axis' and 'angle_deg', or 'align'."
                )
            steps.append({
                "kind": "axis",
                "axis": _as_axis(step["axis"], f"molecule.rotation[{index}].axis"),
                "angle_deg": _as_angle(step["angle_deg"], f"molecule.rotation[{index}].angle_deg"),
            })
    return steps


def _normalize_align(step, index):
    align = step["align"]
    where = f"molecule.rotation[{index}].align"
    if not isinstance(align, dict):
        raise ValueError(f"{where} must be an object.")
    unknown = set(align) - {"from_atom", "to_atom", "axis"}
    if unknown:
        raise ValueError(f"{where} has unknown keys: {sorted(unknown)}.")
    for key in ("from_atom", "to_atom", "axis"):
        if key not in align:
            raise ValueError(f"{where} requires '{key}'.")
    from_atom = _as_index(align["from_atom"], f"{where}.from_atom")
    to_atom = _as_index(align["to_atom"], f"{where}.to_atom")
    if from_atom == to_atom:
        raise ValueError(f"{where} from_atom and to_atom must be different.")
    twist = _as_angle(step.get("twist_deg", 0), f"molecule.rotation[{index}].twist_deg")
    return {
        "kind": "align",
        "from_atom": from_atom,
        "to_atom": to_atom,
        "axis": _as_axis(align["axis"], f"{where}.axis"),
        "twist_deg": twist,
    }


def _as_index(value, where):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a 0-based atom index, got {value!r}.")
    return value


def _as_angle(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number in degrees, got {value!r}.")
    angle = float(value)
    if not np.isfinite(angle):
        raise ValueError(f"{where} must be finite, got {value!r}.")
    return angle


def _as_axis(value, where):
    if isinstance(value, str):
        key = value.lower().strip()
        if key not in _AXIS_NAMES:
            raise ValueError(f"{where} must be 'x', 'y', 'z', or a 3-vector, got {value!r}.")
        axis = np.array(_AXIS_NAMES[key], dtype=float)
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            axis = np.array([float(component) for component in value], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where} must be three numbers, got {value!r}.") from exc
        if not np.all(np.isfinite(axis)):
            raise ValueError(f"{where} must be finite, got {value!r}.")
    else:
        raise ValueError(f"{where} must be 'x', 'y', 'z', or a 3-vector, got {value!r}.")
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError(f"{where} must be a non-zero axis.")
    return axis / norm


def _step_matrix(step, coords):
    if step["kind"] == "axis":
        return _rotation_matrix(step["axis"], step["angle_deg"])
    n_atom = len(coords)
    i, j = step["from_atom"], step["to_atom"]
    if i >= n_atom or j >= n_atom:
        raise ValueError(
            f"align atom index {max(i, j)} is out of range ({n_atom} atoms, 0-based)."
        )
    bond = coords[j] - coords[i]
    if np.linalg.norm(bond) == 0:
        raise ValueError(f"align atoms {i} and {j} are on top of each other.")
    matrix = _rotation_between(bond, step["axis"])
    if step["twist_deg"] != 0:
        matrix = _rotation_matrix(step["axis"], step["twist_deg"]) @ matrix
    return matrix


def _rotation_matrix(axis, angle_deg):
    """Active right-handed rotation about a unit axis."""
    theta = np.deg2rad(angle_deg)
    kx, ky, kz = axis
    cross = np.array([
        [0.0, -kz, ky],
        [kz, 0.0, -kx],
        [-ky, kx, 0.0],
    ])
    return (
        np.eye(3)
        + np.sin(theta) * cross
        + (1.0 - np.cos(theta)) * (cross @ cross)
    )


def _rotation_between(source, target):
    """Active rotation that maps ``source`` onto ``target`` (both non-zero)."""
    a = np.asarray(source, dtype=float)
    b = np.asarray(target, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if cosine > 1.0 - 1e-12:
        return np.eye(3)
    if cosine < -1.0 + 1e-12:
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = helper - a * np.dot(helper, a)
        return _rotation_matrix(axis / np.linalg.norm(axis), 180.0)
    axis = np.cross(a, b)
    return _rotation_matrix(axis / np.linalg.norm(axis), np.rad2deg(np.arccos(cosine)))


def _parse_coords(coords_str):
    atoms = []
    coords = []
    for block in str(coords_str).split(";"):
        parts = block.split()
        if not parts:
            continue
        if len(parts) < 4:
            raise ValueError(f"Cannot parse molecule coordinate block {block!r}.")
        atoms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not atoms:
        raise ValueError("molecule.rotation found no nuclear coordinates to rotate.")
    return atoms, np.asarray(coords, dtype=float)


def _write_xyz(path, atoms, coords, steps):
    comment = (
        "rotated about nuclear-charge center; units: bohr; "
        + json.dumps(steps, default=_json_default, separators=(",", ":"))
    )
    lines = [str(len(atoms)), comment]
    for atom, xyz in zip(atoms, coords):
        lines.append(f"{atom}  {xyz[0]:.10f}  {xyz[1]:.10f}  {xyz[2]:.10f}")
    path.write_text("\n".join(lines) + "\n")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return [float(component) for component in value]
    raise TypeError(f"Cannot encode {type(value).__name__}")
