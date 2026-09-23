# quantum/orientation_check.py
"""Rotation that puts the field-free DCH dipole on the nanoparticle axis.

D3–D6 keep the dipole as a three-component vector. Only the part that moves
along the gap radiates onto the gold sphere. D1, run on the stock geometry
with no molecule.rotation, measures that direction. This command turns it
into the rotation to paste into D3–D6.

K1's three Cartesian dipole files, when they exist, are reported too. That
isotropic spectrum does not need a rotation. Its components are the valence
axis for a later Gaussian run, which is a separate choice from D3–D6.

Run from the repository root after D1:

    python -m plasmol.quantum.orientation_check
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from plasmol.quantum.geometry import _rotation_matrix

_LAB_X = np.array([1.0, 0.0, 0.0])


def load_mu(path):
    """Return (time, mu) from a PlasMol field_p CSV. mu has shape (N, 3)."""
    rows = []
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("Timestamps"):
                continue
            parts = stripped.split(",")
            if len(parts) < 4:
                continue
            rows.append([float(part) for part in parts[:4]])
    if len(rows) < 2:
        raise ValueError(f"{path} does not contain a dipole trace.")
    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1:4]


def swing_axis(mu):
    """
    Unit direction of the largest motion in μ(t).

    The mean is removed first. field_p is already the change from the initial
    density; subtracting the mean drops any leftover offset. A static dipole
    does not radiate. The returned axis is unsigned.
    """
    centered = np.asarray(mu, dtype=float) - np.mean(mu, axis=0)
    rms = np.sqrt(np.mean(centered ** 2, axis=0))
    cov = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    norm = np.linalg.norm(axis)
    if norm == 0 or float(np.max(rms)) == 0.0:
        raise ValueError("Dipole does not move; there is no swing axis.")
    return axis / norm, rms


def axis_angle_deg(a, b):
    """Smallest angle in degrees between two unsigned axes, in [0, 90]."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cosine = float(np.clip(np.abs(np.dot(a, b)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def rotation_onto_x(axis):
    """
    Active rotation that maps ``axis`` onto the x line.

    The axis is unsigned, so the smaller rotation is used and the result is
    at most 90 degrees. None means the motion is already along x.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    if axis[0] < 0.0:
        axis = -axis
    angle = axis_angle_deg(axis, _LAB_X)
    if angle <= 1e-6:
        return None
    cross = np.cross(axis, _LAB_X)
    cross_norm = np.linalg.norm(cross)
    if cross_norm == 0.0:
        return None
    rot_axis = cross / cross_norm
    mapped = _rotation_matrix(rot_axis, angle) @ axis
    if mapped[0] < 0.0:
        rot_axis = -rot_axis
    return {
        "axis": [round(float(component), 8) for component in rot_axis],
        "angle_deg": round(float(angle), 4),
    }


def _has_rotation(path):
    payload = json.loads(Path(path).read_text())
    return bool(payload.get("molecule", {}).get("rotation"))


def _fmt(vector):
    return f"[{vector[0]: .6f}, {vector[1]: .6f}, {vector[2]: .6f}]"


def _k1_report(root):
    names = ("x", "y", "z")
    paths = [
        root / "Step_5" / "K1" / f"{name}_dir" / "field_p_K1.csv" for name in names
    ]
    if not all(path.is_file() for path in paths):
        return (
            "K1 has not been run. Its three dipole files are the valence axis "
            "for the Gaussian nanoparticle runs. They are not required for D3–D6."
        )
    rms = []
    for path in paths:
        _time, mu = load_mu(path)
        _axis, component_rms = swing_axis(mu)
        rms.append(float(np.linalg.norm(component_rms)))
    brightest = names[int(np.argmax(rms))]
    lines = [
        "K1 valence response (root-mean-square of each kick's dipole):",
        f"  x={rms[0]:.6e}  y={rms[1]:.6e}  z={rms[2]:.6e}",
        f"Brightest lab axis: {brightest}",
        "Use that only if a Gaussian run needs the valence axis on +x. "
        "It is not the rotation for D3–D6.",
    ]
    return "\n".join(lines)


def compare_campaign(root):
    """
    Report the D1 swing axis and the rotation that puts it on +x.

    Raises FileNotFoundError if D1 has not been run, or ValueError if D1
    was itself rotated.
    """
    root = Path(root)
    d1_json = root / "Step_4" / "D1" / "D1.json"
    d1_mu = root / "Step_4" / "D1" / "field_p_D1.csv"
    missing = [path for path in (d1_json, d1_mu) if not path.is_file()]
    if missing:
        listed = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "D3–D6 orientation needs the stock-frame D1 dipole.\n"
            f"Missing:\n{listed}"
        )
    if _has_rotation(d1_json):
        raise ValueError(
            f"{d1_json} sets molecule.rotation. "
            "D1 has to stay on the stock geometry."
        )

    _time, mu = load_mu(d1_mu)
    axis, rms = swing_axis(mu)
    rotation = rotation_onto_x(axis)
    angle = axis_angle_deg(axis, _LAB_X)
    if rotation is None:
        pasted = "D3–D6 do not need molecule.rotation. The core-hole motion is already along x."
    else:
        block = json.dumps(rotation, separators=(", ", ": "))
        pasted = (
            "Paste this into molecule on D3, D4, D5, and D6 only:\n"
            f"  \"rotation\": {block}"
        )

    lines = [
        "D1 double-core-hole dipole, stock frame",
        "",
        "field_p is the dipole change from the density at the moment of ionization.",
        "The axis below is the direction that moves. A static dipole does not radiate.",
        "",
        f"Swing axis (unsigned): {_fmt(axis)}",
        "Root-mean-square after removing the mean: "
        f"x={rms[0]:.6e}  y={rms[1]:.6e}  z={rms[2]:.6e}",
        f"Angle from the nanoparticle axis (+x): {angle:.2f} degrees",
        "",
        pasted,
        "",
        "D1 and D2 stay on the stock geometry. Do not edit Step_1/3p.xyz.",
        "",
        _k1_report(root),
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="jobs/final",
        help="Campaign directory (default: jobs/final)",
    )
    args = parser.parse_args(argv)
    root = Path(args.root)
    try:
        report = compare_campaign(root)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    out = root / "orientation_check.txt"
    out.write_text(report)
    print(report, end="")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
