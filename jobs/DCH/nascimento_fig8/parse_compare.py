#!/usr/bin/env python3
"""Period table + overlays for the Nascimento Fig. 8a experiments."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks, savgol_filter

AU2FS = 0.024188843265857
HERE = Path(__file__).resolve().parent
GOLD = HERE / "Gold Data"
SWEEP = HERE / "pyscf_sweep"

PAIRS = [
    (21, "Nascimento H-1.csv", "#E0B040", "21 / H-1"),
    (23, "Nascimento H.csv", "#3CA87A", "23 / H"),
    (24, "Nascimento L.csv", "#4C6FE0", "24 / L"),
]


def load_gold(path: Path):
    arr = np.loadtxt(path, delimiter=",")
    x, y = arr[:, 0], arr[:, 1]
    order = np.argsort(x)
    x, y = x[order], y[order]
    uniq, inv = np.unique(x, return_inverse=True)
    y_mean = np.zeros_like(uniq)
    counts = np.zeros_like(uniq)
    np.add.at(y_mean, inv, y)
    np.add.at(counts, inv, 1.0)
    x, y = uniq, y_mean / counts
    xg = np.arange(x[0], x[-1] + 0.0025, 0.005)
    yg = PchipInterpolator(x, y)(xg)
    win = 9
    return xg, savgol_filter(yg, window_length=win, polyorder=3)


def load_plasmol(csv: Path, divide_by_two: bool):
    df = pd.read_csv(csv, comment="#")
    t = df.iloc[:, 0].to_numpy() * AU2FS
    scale = 0.5 if divide_by_two else 1.0
    series = {}
    for idx, _, _, _ in PAIRS:
        col = f"MO index {idx}"
        if col in df.columns:
            series[idx] = scale * df[col].to_numpy()
    return t, series


def period(t, y):
    yd = y - np.mean(y)
    dist = max(int(0.04 / np.median(np.diff(t))), 1)
    peaks, _ = find_peaks(yd, prominence=0.15 * np.std(yd), distance=dist)
    if len(peaks) < 3:
        return np.nan
    return float(np.median(np.diff(t[peaks])))


def parse_nwchem(path: Path):
    times, rows = [], []
    for line in path.read_text(errors="replace").splitlines():
        if "<rt_tddft>" not in line or "MO Occupations" not in line:
            continue
        body = line.split("<rt_tddft>")[1].split("#")[0]
        nums = [float(x) for x in body.split()]
        if len(nums) < 25:
            continue
        times.append(nums[0] * AU2FS)
        rows.append(nums[1:])
    if not times:
        return None, {}
    arr = np.asarray(rows)
    # NWChem prints 0-based-from-1 MOs as 0-2 electron occupations.
    series = {}
    for idx, _, _, _ in PAIRS:
        if idx < arr.shape[1]:
            series[idx] = 0.5 * (2.0 - arr[:, idx])  # hole on 0-1 scale, if n is 0-2 occ
            # If n is already hole-like, this will look wrong; caller can inspect.
    return np.asarray(times), series


def main():
    jobs = {"baseline": (HERE / "mo_occ.csv", True)}
    for name in ["cart", "grid5", "grid7", "hf20", "hf30", "rk4", "dt025"]:
        csv = SWEEP / name / "mo_occ.csv"
        if csv.exists():
            jobs[name] = (csv, False)

    print(f"{'job':<12} {'T21':>8} {'T24':>8} {'vs base 21':>10} {'vs gold 21':>10}")
    gold_t, gold_y = load_gold(GOLD / "Nascimento H-1.csv")
    t_gold_l, y_gold_l = load_gold(GOLD / "Nascimento L.csv")
    t_g21 = period(gold_t, gold_y)
    t_g24 = period(t_gold_l, y_gold_l)
    base_t21 = base_t24 = np.nan

    for name, (csv, div2) in jobs.items():
        t, series = load_plasmol(csv, div2)
        t21 = period(t, series[21]) if 21 in series else np.nan
        t24 = period(t, series[24]) if 24 in series else np.nan
        if name == "baseline":
            base_t21, base_t24 = t21, t24
        d21 = 100 * (t21 - base_t21) / base_t21 if base_t21 else np.nan
        g21 = 100 * (t21 - t_g21) / t_g21 if t_g21 else np.nan
        print(f" {name:<12} {t21:8.4f} {t24:8.4f} {d21:9.2f}% {g21:9.2f}%")

    nw = HERE / "nwchem_dch" / "3p_dch.out"
    if nw.exists():
        tn, sn = parse_nwchem(nw)
        if tn is not None and 21 in sn:
            print(f" {'nwchem':<12} {period(tn, sn[21]):8.4f} {period(tn, sn[24]):8.4f}")


if __name__ == "__main__":
    main()
