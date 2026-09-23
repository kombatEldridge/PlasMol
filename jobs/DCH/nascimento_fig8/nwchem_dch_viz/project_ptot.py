#!/usr/bin/env python3
"""Project NWChem ptot_ao_re dumps onto frozen neutral MOs (Nascimento Eq. 2)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import FortranFile
from scipy.signal import find_peaks

AU2FS = 0.024188843265857
DT_AU = 0.05
NBF = 144
NOCC = 24
HERE = Path(__file__).resolve().parent
XYZ = HERE.parent / "3p_c2v.xyz"
TARGET_PERIOD = 0.136
PERIOD_WINDOW = (0.12, 0.16)


def read_ptot(path: Path, nbf: int = NBF) -> np.ndarray:
    f = FortranFile(path, "r")
    try:
        cols = [f.read_reals(dtype=np.float64) for _ in range(nbf)]
    finally:
        f.close()
    for i, col in enumerate(cols):
        if col.size != nbf:
            raise ValueError(f"{path} column {i} has length {col.size}, expected {nbf}")
    P = np.column_stack(cols)
    return 0.5 * (P + P.T)


def list_dumps(directory: Path) -> list[tuple[int, Path]]:
    hits = []
    for p in directory.glob("*ptot_ao_re.*"):
        tail = p.name.split(".")[-1]
        if tail.isdigit():
            hits.append((int(tail), p))
    hits.sort()
    return hits


def read_movecs(path: Path):
    """Read a closed-shell NWChem .movecs (unformatted) into occ, energy, C."""
    f = FortranFile(path, "r")
    recs = [f.read_record(dtype=np.uint8) for _ in range(9)]

    def i8(buf):
        return int(np.frombuffer(buf.tobytes(), dtype="<i8")[0])

    nsets, nbf, nmo = i8(recs[6]), i8(recs[7]), i8(recs[8])
    if nsets != 1:
        raise RuntimeError(f"{path} has nsets={nsets}, expected closed-shell 1")
    occ = f.read_reals(dtype=np.float64)
    ene = f.read_reals(dtype=np.float64)
    C = np.zeros((nbf, nmo))
    for k in range(nmo):
        vec = f.read_reals(dtype=np.float64)
        if vec.size != nbf:
            raise RuntimeError(f"{path} MO {k} length {vec.size}, expected {nbf}")
        C[:, k] = vec
    f.close()
    return occ, ene, C


def period_fs(t_fs: np.ndarray, y: np.ndarray) -> float:
    yd = y - np.mean(y)
    if yd.std() < 1e-8 or len(t_fs) < 8:
        return float("nan")
    dist = max(int(0.04 / np.median(np.diff(t_fs))), 1)
    peaks, _ = find_peaks(yd, prominence=0.15 * np.std(yd), distance=dist)
    if len(peaks) < 3:
        return float("nan")
    return float(np.median(np.diff(t_fs[peaks])))


def project(directory: Path) -> dict:
    dumps = list_dumps(directory)
    if not dumps:
        raise FileNotFoundError(f"no ptot_ao_re.* dumps in {directory}")

    movecs = directory / "neutral.movecs"
    if not movecs.exists():
        raise FileNotFoundError(movecs)
    n0, ene, C = read_movecs(movecs)
    # C is S-orthonormal: C^T S C = I  =>  S = (C C^T)^{-1}
    S = np.linalg.inv(C @ C.T)
    nmo = C.shape[1]
    if C.shape[0] != NBF:
        raise RuntimeError(f"expected nao={NBF}, got {C.shape[0]}")

    times, holes = [], []
    traces = []
    for it, path in dumps:
        P = read_ptot(path)
        nelec = float(np.trace(S @ P).real)
        n_e = np.diag(C.T @ S @ P @ S @ C).real
        h = 0.5 * (n0 - n_e)
        t_au = (it - 1) * DT_AU
        times.append(t_au)
        holes.append(h[:26])
        traces.append({
            "it": it,
            "t_au": t_au,
            "nelec": nelec,
            "n": n_e,
            "h": h,
        })

    t_au = np.asarray(times)
    H = np.vstack(holes)
    t_fs = t_au * AU2FS

    cols = {"Timestamps (au)": t_au}
    for k in range(H.shape[1]):
        cols[f"MO index {k}"] = H[:, k]
    df = pd.DataFrame(cols)
    csv_path = directory / "mo_occ_projected.csv"
    with csv_path.open("w") as f:
        f.write(
            "# Hole occupations from NWChem P_AO(t) projected on frozen neutral MOs\n"
            "# h_k = (n0_k - n_k)/2  (0-1 scale). C_n and S from neutral.movecs.\n"
        )
        df.to_csv(f, index=False)

    t0 = traces[0]
    n_e0 = t0["n"]
    checks = {
        "n_dumps": len(dumps),
        "t0_au": t0["t_au"],
        "t_end_au": float(t_au[-1]),
        "nelec_t0": t0["nelec"],
        "n0_core": float(n_e0[0]),
        "mean_abs_occ_err": float(np.mean(np.abs(n_e0[1:NOCC] - 2.0))),
        "mean_virt": float(np.mean(n_e0[NOCC:])),
        "h21_t0": float(t0["h"][21]),
        "h23_t0": float(t0["h"][23]),
        "h24_t0": float(t0["h"][24]),
        "period_21_fs": period_fs(t_fs, H[:, 21]),
        "period_24_fs": period_fs(t_fs, H[:, 24]),
        "e_mo0": float(ene[0]),
    }
    t0_ok = (
        abs(checks["nelec_t0"] - 46.0) < 1e-3
        and checks["n0_core"] < 0.05
        and checks["mean_abs_occ_err"] < 0.05
        and checks["mean_virt"] < 0.05
        and abs(checks["h21_t0"]) < 0.05
        and abs(checks["h23_t0"]) < 0.05
        and abs(checks["h24_t0"]) < 0.05
    )
    per21, per24 = checks["period_21_fs"], checks["period_24_fs"]
    period_ok = (
        PERIOD_WINDOW[0] <= per21 <= PERIOD_WINDOW[1]
        and PERIOD_WINDOW[0] <= per24 <= PERIOD_WINDOW[1]
    )
    checks["t0_ok"] = bool(t0_ok)
    checks["period_ok"] = bool(period_ok)
    checks["green"] = bool(t0_ok and period_ok)
    checks["csv"] = str(csv_path)

    (directory / "phase_verdict.json").write_text(json.dumps(checks, indent=2) + "\n")

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 1.0,
    })
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    for idx, color in ((21, "#E0B040"), (23, "#3CA87A"), (24, "#4C6FE0")):
        ax.plot(t_fs, H[:, idx], color=color, lw=1.3, label=str(idx))
    ax.axhline(0.0, color="0.7", lw=0.6, zorder=0)
    ax.set_xlim(0.0, max(1.0, t_fs[-1]))
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("Time [fs]")
    ax.set_ylabel("hole occupation number")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )
    fig.tight_layout()
    fig.savefig(directory / "mo_21_23_24_projected.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return checks


def main() -> int:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    checks = project(directory)
    print(json.dumps(checks, indent=2))
    return 0 if checks["green"] else 2


if __name__ == "__main__":
    sys.exit(main())
