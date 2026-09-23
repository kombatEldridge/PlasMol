"""Recreate Nascimento et al., J. Phys. Chem. Lett. 2020, Figure 8a
from WebPlotDigitizer-style point dumps of the three traces."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent

SERIES = [
    {
        "file": "Nascimento H-1.csv",
        "label": "H-1",
        "color": "#E0B040",
        "zorder": 3,
    },
    {
        "file": "Nascimento H.csv",
        "label": "H",
        "color": "#3CA87A",
        "zorder": 4,
    },
    {
        "file": "Nascimento L.csv",
        "label": "L",
        "color": "#B050E0",
        "zorder": 2,
    },
]

# ~0.18 fs oscillation; keep the window well below one period.
GRID_DT_FS = 0.005
SAVGOL_WINDOW_FS = 0.045
SAVGOL_POLY = 3


def load_xy(path):
    arr = np.loadtxt(path, delimiter=",")
    x, y = arr[:, 0], arr[:, 1]
    order = np.argsort(x)
    x, y = x[order], y[order]
    # Mean-merge identical times so PCHIP sees a function.
    uniq, inv = np.unique(x, return_inverse=True)
    y_mean = np.zeros_like(uniq)
    counts = np.zeros_like(uniq)
    np.add.at(y_mean, inv, y)
    np.add.at(counts, inv, 1.0)
    return uniq, y_mean / counts


def smooth_trace(x, y):
    xg = np.arange(x[0], x[-1] + 0.5 * GRID_DT_FS, GRID_DT_FS)
    yg = PchipInterpolator(x, y)(xg)
    win = int(round(SAVGOL_WINDOW_FS / GRID_DT_FS))
    if win % 2 == 0:
        win += 1
    win = max(win, SAVGOL_POLY + 2 + (SAVGOL_POLY % 2 == 0))
    if win >= len(yg):
        return xg, yg
    return xg, savgol_filter(yg, window_length=win, polyorder=SAVGOL_POLY)


def plot_fig8a(out_path):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(5.55, 3.55), dpi=200)

    handles = []
    labels = []
    for spec in SERIES:
        x, y = load_xy(HERE / spec["file"])
        xs, ys = smooth_trace(x, y)
        (line,) = ax.plot(
            xs,
            ys,
            color=spec["color"],
            lw=2,
            solid_capstyle="round",
            zorder=spec["zorder"],
            label=spec["label"],
        )
        handles.append(line)
        labels.append(spec["label"])

    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xticks(np.arange(0.0, 5.01, 0.5))
    ax.set_yticks(np.arange(-1.2, 1.21, 0.2))
    ax.set_xlabel("Time [fs]", fontsize=11)
    ax.set_ylabel("Hole occupation number", fontsize=11)
    ax.tick_params(axis="both", which="major", labelsize=9, direction="out", length=3.5, width=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")

    # Paper puts the panel label inside the frame, top left.
    ax.text(
        0.025,
        0.96,
        "(a) 3-pentanone",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        zorder=5,
    )

    ax.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        fontsize=10,
        handlelength=1.7,
        columnspacing=1.8,
        borderpad=0.45,
        handletextpad=0.5,
        borderaxespad=0.7,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    plot_fig8a(HERE / "fig8a.png")
