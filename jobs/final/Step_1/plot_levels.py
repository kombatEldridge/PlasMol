#!/usr/bin/env python3
"""Every recorded LC-ωPBE root at or below 10 eV, one row per molecule."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

import run_screen as rs

HERE = Path(__file__).resolve().parent
EV_NM = 1239.84193
WINDOW = 12.0
# Dipole extinction peak of a sphere in water, Rakic Lorentz–Drude dielectric,
# diameter 15–100 nm. Ag and Au are meep.materials. Al uses the same Rakic
# coefficients; Meep's Al object is gated off below 207 nm. The peak redshifts
# monotonically, so the range is the 15 nm peak to the 100 nm peak.
LSPR_NM = {
    "Al": (185.5, 393.0),
    "Ag": (407.5, 506.0),
    "Au": (518.5, 580.0),
}
LSPR_COLOR = {"Al": "#6baed6", "Ag": "#fb6a4a", "Au": "#fec44f"}

LABELS = {
    "n2": r"N$_2$",
    "co": "CO",
    "co2": r"CO$_2$",
    "n2o": r"N$_2$O",
    "o2": r"O$_2$",
    "h2o": r"H$_2$O",
    "nh3": r"NH$_3$",
    "ch4": r"CH$_4$",
    "h2co": r"H$_2$CO",
    "c2h2": r"C$_2$H$_2$",
    "c2h4": r"C$_2$H$_4$",
    "c2h6": r"C$_2$H$_6$",
    "c6h6": r"C$_6$H$_6$",
    "para_aminophenol": "para-aminophenol",
    "ortho_aminophenol": "ortho-aminophenol",
    "meta_aminophenol": "meta-aminophenol",
    "h2s": r"H$_2$S",
    "so2": r"SO$_2$",
    "cs2": r"CS$_2$",
    "lif": "LiF",
    "beo": "BeO",
    "bf": "BF",
    "pyrimidine": "pyrimidine",
    "purine": "purine",
    "uracil": "uracil",
    "cytosine": "cytosine",
    "thymine": "thymine",
    "adenine": "adenine",
    "guanine": "guanine",
    "formamide": "formamide",
    "pentan3one": "3-pentanone",
    "pentan2one": "2-pentanone",
    "pentanal": "pentanal",
}

# Molecules that an experimental double-core-hole paper measured.
# Ortho- and meta-aminophenol, the nucleobases, the pentanones, LiF, BeO,
# BF, and formaldehyde were calculation only.
EXPERIMENTAL = {
    "n2",
    "co",
    "co2",
    "n2o",
    "o2",
    "h2o",
    "nh3",
    "ch4",
    "c2h2",
    "c2h4",
    "c2h6",
    "c6h6",
    "para_aminophenol",
    "h2s",
    "so2",
    "cs2",
}


def load(folder: Path):
    rows = []
    for line in (folder / "absorption.txt").read_text().splitlines()[1:]:
        if not line.strip():
            continue
        _root, _spin, energy, _nm, strength, _below = line.split("\t")
        rows.append((float(energy), float(strength)))
    return rows


def main():
    names = [row[0] for row in rs.MOLECULES]
    # Brightest low-energy line first, so the plasmon region is at the top.
    def sort_key(name):
        rows = load(HERE / name)
        bright = [energy for energy, strength in rows if energy <= WINDOW and strength >= 0.01]
        if bright:
            return (0, min(bright))
        dark = [energy for energy, _strength in rows if energy <= WINDOW]
        if dark:
            return (1, min(dark))
        return (2, min(energy for energy, _strength in rows))

    names = sorted(names, key=sort_key)
    fig, ax = plt.subplots(figsize=(10.5, 11.5))
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=-4.0, vmax=0.0)
    half = 0.36
    for index, name in enumerate(names):
        if name in EXPERIMENTAL:
            ax.axhspan(index - 0.5, index + 0.5, color="#f9f1de", zorder=0)
        for energy, strength in load(HERE / name):
            if energy > WINDOW or strength <= 0.0:
                continue
            color = cmap(norm(np.log10(strength)))
            ax.vlines(energy, index - half, index + half, colors=color, lw=2.0, zorder=3)

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    bar = fig.colorbar(mappable, ax=ax, pad=0.01, fraction=0.03)
    bar.set_label("oscillator strength")
    bar.set_ticks([-4, -3, -2, -1, 0])
    bar.set_ticklabels(["10$^{-4}$", "10$^{-3}$", "0.01", "0.1", "1"])

    for metal, (short_nm, long_nm) in LSPR_NM.items():
        ax.axvspan(
            EV_NM / long_nm,
            EV_NM / short_nm,
            color=LSPR_COLOR[metal],
            alpha=0.18,
            zorder=1,
        )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABELS[name] for name in names])
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlim(1.4, WINDOW + 0.15)
    ax.set_xlabel("Excitation energy (eV)")
    ax.set_title("LC-ωPBE, IP-tuned μ / 6-311G* roots at or below 12 eV")
    ax.grid(True, axis="x", color="0.9", zorder=0)
    ax.tick_params(axis="y", length=0)

    handles = [
        Patch(facecolor=LSPR_COLOR["Al"], alpha=0.45, edgecolor="none", label="Al 186–393 nm"),
        Patch(facecolor=LSPR_COLOR["Ag"], alpha=0.45, edgecolor="none", label="Ag 408–506 nm"),
        Patch(facecolor=LSPR_COLOR["Au"], alpha=0.45, edgecolor="none", label="Au 518–580 nm"),
        Patch(facecolor="#f6e7c1", edgecolor="none", label="measured in a DCH experiment"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout()
    figure = HERE / "absorption_levels.png"
    fig.savefig(figure, dpi=150, bbox_inches="tight")
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
