#!/usr/bin/env python3
"""
Model 2 — Quasistatic Mie sphere + Lorentzian molecular point dipole.

Two Meep-matched gold dielectrics (meep.materials):

  * "Au"            — Rakić et al. (1998) Drude–Lorentz
  * "Au_JC_visible" — Johnson–Christy visible-range fit (Barchiesi & Grosges)

Geometry matched to jobs/parallel_abs (Au sphere + Na on the radial axis).
Host medium: constant refractive index n (default 1.33, water), ε_b = n².

    α_eff(ω) = α_m(ω) G(ω) / (1 − α_m(ω) S(ω))
    A(ω) ∝ −ω Im[α_eff(ω)]   (peak-normalized)
    Fröhlich: Re ε(ω_LSPR) = −2 ε_b

Edit the PARAMETERS block at the bottom, then run:

    python spectrum_model2.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Meep material library (parameters in eV; formula matches Meep susceptibilities)
# ---------------------------------------------------------------------------
#
# Meep (c = 1) uses frequency f in 1/μm.  With eV_um_scale = 1/1.23984193,
# E[eV] = f / eV_um_scale.  The 2π factors in Meep's ω = 2πf cancel, leaving:
#
#   Lorentz:  χ = (σ E0²) / (E0² − E² − i Γ E)
#   Drude:    χ = (σ E0²) / (     − E² − i Γ E)
#
# where E0, Γ are the Meep frequency/gamma converted to eV.

# Term: (kind, E0_eV, gamma_eV, amp) with amp = σ * E0²  [eV²]
# kind ∈ {"drude", "lorentz"}

MEEP_AU_MATERIALS: dict[str, dict] = {
    # meep.materials.Au — A.D. Rakić et al., Appl. Opt. 37, 5271 (1998)
    "Au": {
        "label": "Meep Au (Rakić 1998)",
        "reference": "A.D. Rakic et al., Applied Optics 37, 5271 (1998)",
        "eps_inf": 1.0,
        "terms": [
            # Drude: f0≈0, σ huge → amp = f0 * ω_p² = 0.760 * 9.03²
            ("drude", 0.0, 0.053, 0.760 * 9.03**2),
            ("lorentz", 0.415, 0.241, 0.024 * 9.03**2),
            ("lorentz", 0.830, 0.345, 0.010 * 9.03**2),
            ("lorentz", 2.969, 0.870, 0.071 * 9.03**2),
            ("lorentz", 4.304, 2.494, 0.601 * 9.03**2),
            ("lorentz", 13.32, 2.214, 4.384 * 9.03**2),
        ],
        # amp for Lorentz = f_j * ω_p²  in Rakić (since σ = f_j ω_p²/ω_j² ⇒ σ ω_j² = f_j ω_p²)
    },
    # meep.materials.Au_JC_visible — JC fit, visible band
    "Au_JC_visible": {
        "label": "Meep Au_JC_visible (Johnson–Christy)",
        "reference": (
            "P.B. Johnson & R.W. Christy, Phys. Rev. B 6, 4370 (1972); "
            "fit of Barchiesi & Grosges, J. Nanophotonics 8, 083097 (2014)"
        ),
        "eps_inf": 6.1599,
        "terms": [
            # DrudeSusceptibility(frequency≈8.87 eV, gamma≈0.04745 eV, sigma=1)
            # amp = σ E0² = 1 * 8.87000103281206**2
            ("drude", 8.87000103281206, 0.047454447168797244, 1.0 * 8.87000103281206**2),
            # Lorentzian(frequency≈3.068 eV, gamma≈1.099 eV, sigma=2.07118534879440)
            (
                "lorentz",
                3.0684255933804754,
                1.098818992781705,
                2.07118534879440 * 3.0684255933804754**2,
            ),
        ],
    },
}

# Fix Rakić Lorentz amps to f_j * ω_p² (not f_j * ω_p² / something wrong)
# Rakić: σ_j = f_j * ω_p² / ω_j², amp = σ_j * ω_j² = f_j * ω_p²
_WP2 = 9.03**2
MEEP_AU_MATERIALS["Au"]["terms"] = [
    ("drude", 0.0, 0.053, 0.760 * _WP2),
    ("lorentz", 0.415, 0.241, 0.024 * _WP2),
    ("lorentz", 0.830, 0.345, 0.010 * _WP2),
    ("lorentz", 2.969, 0.870, 0.071 * _WP2),
    ("lorentz", 4.304, 2.494, 0.601 * _WP2),
    ("lorentz", 13.32, 2.214, 4.384 * _WP2),
]


def epsilon_meep_material(omega: np.ndarray, material: str) -> np.ndarray:
    """
    Complex dielectric function for a Meep metal key ("Au" or "Au_JC_visible").

    omega : energy in eV
    """
    if material not in MEEP_AU_MATERIALS:
        known = ", ".join(sorted(MEEP_AU_MATERIALS))
        raise ValueError(f"Unknown material '{material}'. Choose one of: {known}")

    spec = MEEP_AU_MATERIALS[material]
    omega = np.asarray(omega, dtype=float)
    eps = np.full(omega.shape, complex(spec["eps_inf"], 0.0), dtype=complex)

    for kind, E0, gamma, amp in spec["terms"]:
        if kind == "drude":
            # χ = amp / (−E² − i Γ E)
            eps = eps + amp / (-(omega**2) - 1j * gamma * omega)
        elif kind == "lorentz":
            eps = eps + amp / (E0**2 - omega**2 - 1j * gamma * omega)
        else:
            raise ValueError(f"Unknown term kind '{kind}'")
    return eps


def list_meep_materials() -> list[str]:
    return list(MEEP_AU_MATERIALS.keys())


# ---------------------------------------------------------------------------
# Mie / Gersten–Nitzan building blocks (stable a/d multipoles)
# ---------------------------------------------------------------------------

def sphere_multipole_factor(eps: np.ndarray, l: int, eps_b: float = 1.0) -> np.ndarray:
    """
    Quasistatic multipole factor in a host of permittivity ε_b:

        f_ℓ = (ε − ε_b) / (ε + ((ℓ+1)/ℓ) ε_b)

    Vacuum: ε_b = 1.  Non-dispersive water n=1.33: ε_b = n² ≈ 1.769.
    """
    return (eps - eps_b) / (eps + ((l + 1) / l) * eps_b)


def local_field_G(
    a: float,
    d: float,
    eps: np.ndarray,
    orientation: str,
    eps_b: float = 1.0,
) -> np.ndarray:
    """
    Incident-field enhancement (dipole term) relative to E_inc in the medium:

      G_∥ = 1 + 2 (a/d)³ f_1
      G_⊥ = 1 −     (a/d)³ f_1

    with f_1 = (ε − ε_b)/(ε + 2 ε_b).
    """
    ad = a / d
    ad3 = ad**3
    f1 = sphere_multipole_factor(eps, 1, eps_b)
    orient = orientation.lower().strip()
    if orient in ("parallel", "radial", "longitudinal", "||"):
        return 1.0 + 2.0 * ad3 * f1
    if orient in ("perpendicular", "tangential", "perp", "⊥"):
        return 1.0 - ad3 * f1
    raise ValueError(f"Unknown orientation '{orientation}'")


def reflected_propagator_S(
    a: float,
    d: float,
    eps: np.ndarray,
    orientation: str,
    l_max: int,
    eps_b: float = 1.0,
) -> np.ndarray:
    """
    Self-field propagator S (polarizability-volume units) in host ε_b.

    S_∥ = Σ_l (l+1)² (a/d)^{2l+1} f_l / d³
    S_⊥ = Σ_l [l(l+1)/2] (a/d)^{2l+1} f_l / d³

    with medium-corrected f_ℓ.
    """
    ad = a / d
    inv_d3 = 1.0 / (d**3)
    orient = orientation.lower().strip()
    S = np.zeros_like(eps, dtype=complex)
    power = ad**3  # (a/d)^{2l+1} at l=1
    ad2 = ad**2
    for l in range(1, l_max + 1):
        f = sphere_multipole_factor(eps, l, eps_b)
        if orient in ("parallel", "radial", "longitudinal", "||"):
            pref = (l + 1) ** 2
        elif orient in ("perpendicular", "tangential", "perp", "⊥"):
            pref = 0.5 * l * (l + 1)
        else:
            raise ValueError(f"Unknown orientation '{orientation}'")
        S = S + pref * power * inv_d3 * f
        power *= ad2
        if power < 1e-300:
            break
    return S


def alpha_molecule_lorentz(
    omega: np.ndarray,
    omega_m: float,
    gamma_m: float,
    alpha0: float,
) -> np.ndarray:
    """α_m(ω) = α0 · ω_m² / (ω_m² − ω² − i γ_m ω)  [volume]."""
    return alpha0 * (omega_m**2) / (omega_m**2 - omega**2 - 1j * gamma_m * omega)


def alpha_eff_molecule(
    alpha_m: np.ndarray,
    G: np.ndarray,
    S: np.ndarray,
    include_back_action: bool,
) -> np.ndarray:
    if include_back_action:
        denom = 1.0 - alpha_m * S
        denom = np.where(np.abs(denom) < 1e-30, 1e-30 + 0j, denom)
        return alpha_m * G / denom
    return alpha_m * G


def absorption_like(omega: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return -omega * np.imag(alpha)


def peak_normalize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    y[~np.isfinite(y)] = 0.0
    peak = np.max(np.abs(y))
    if peak <= 0.0:
        return y
    if y[np.argmax(np.abs(y))] < 0.0:
        y = -y
    return y / np.max(np.abs(y))


def frohlich_resonance_energy(
    e_min: float,
    e_max: float,
    eps_fn: Callable[[np.ndarray], np.ndarray],
    eps_b: float = 1.0,
    n_scan: int = 5000,
) -> float:
    """Energy where Re[ε] is closest to −2 ε_b (quasistatic LSPR in host ε_b)."""
    w = np.linspace(e_min, e_max, n_scan)
    eps = eps_fn(w)
    idx = int(np.argmin(np.abs(np.real(eps) + 2.0 * eps_b)))
    return float(w[idx])


def compute_spectrum(
    *,
    material: str,
    a_nm: float,
    d_nm: float,
    orientation: str,
    l_max: int,
    include_back_action: bool,
    omega_m: float,
    gamma_m: float,
    alpha0_nm3: float,
    e_min: float,
    e_max: float,
    n_points: int,
    n_host: float = 1.33,
) -> dict:
    eps_b = float(n_host) ** 2
    omega = np.linspace(e_min, e_max, n_points)
    eps = epsilon_meep_material(omega, material)
    w_lsp = frohlich_resonance_energy(
        e_min,
        e_max,
        lambda w: epsilon_meep_material(w, material),
        eps_b=eps_b,
    )
    G = local_field_G(a_nm, d_nm, eps, orientation, eps_b=eps_b)
    S = reflected_propagator_S(a_nm, d_nm, eps, orientation, l_max, eps_b=eps_b)
    alpha_m = alpha_molecule_lorentz(omega, omega_m, gamma_m, alpha0_nm3)
    alpha_eff = alpha_eff_molecule(alpha_m, G, S, include_back_action)
    A = peak_normalize(absorption_like(omega, alpha_eff))
    A_bare = peak_normalize(absorption_like(omega, alpha_m))
    return {
        "omega": omega,
        "A": A,
        "A_bare": A_bare,
        "eps": eps,
        "G": G,
        "alpha_eff": alpha_eff,
        "w_lsp": w_lsp,
        "material": material,
        "label": MEEP_AU_MATERIALS[material]["label"],
        "n_host": float(n_host),
        "eps_b": eps_b,
    }


def run(
    *,
    material: str,
    a_nm: float,
    d_nm: float,
    orientation: str,
    l_max: int,
    include_back_action: bool,
    omega_m: float,
    gamma_m: float,
    alpha0_nm3: float,
    e_min: float,
    e_max: float,
    n_points: int,
    compare_csv: str | None,
    out_csv: str,
    out_png: str,
    show_bare: bool,
    show_G: bool,
    run_both: bool = False,
    n_host: float = 1.33,
) -> None:
    out_dir = Path(__file__).resolve().parent
    materials = list_meep_materials() if run_both else [material]

    # If run_both, write per-material files and a comparison plot
    results = []
    for mat in materials:
        r = compute_spectrum(
            material=mat,
            a_nm=a_nm,
            d_nm=d_nm,
            orientation=orientation,
            l_max=l_max,
            include_back_action=include_back_action,
            omega_m=omega_m,
            gamma_m=gamma_m,
            alpha0_nm3=alpha0_nm3,
            e_min=e_min,
            e_max=e_max,
            n_points=n_points,
            n_host=n_host,
        )
        results.append(r)

        tag = mat.replace("/", "_")
        csv_name = out_csv if not run_both and len(materials) == 1 else f"spectrum_model2_{tag}.csv"
        png_name = out_png if not run_both and len(materials) == 1 else f"spectrum_model2_{tag}.png"
        # Always also write the selected material to out_csv/out_png when single
        if not run_both:
            csv_name, png_name = out_csv, out_png

        omega = r["omega"]
        G_abs = np.abs(r["G"])
        df = pd.DataFrame(
            {
                "Frequency": omega,
                "Absorption": r["A"],
                "Re_alpha_eff": np.real(r["alpha_eff"]),
                "Im_alpha_eff": np.imag(r["alpha_eff"]),
                "Re_eps": np.real(r["eps"]),
                "Im_eps": np.imag(r["eps"]),
                "abs_G": G_abs,
                "Re_G": np.real(r["G"]),
                "Im_G": np.imag(r["G"]),
            }
        )
        csv_path = out_dir / csv_name
        png_path = out_dir / png_name
        df.to_csv(csv_path, index=False)

        print(f"=== Model 2 [{r['label']}] ===")
        print(f"  material = '{mat}'")
        print(f"  a = {a_nm:.4f} nm,  d = {d_nm:.4f} nm,  a/d = {a_nm/d_nm:.4f}")
        print(f"  gap = {d_nm - a_nm:.4f} nm")
        print(f"  orientation = {orientation},  l_max = {l_max},  back_action = {include_back_action}")
        print(f"  host n = {r['n_host']:.4f},  ε_b = n² = {r['eps_b']:.4f}")
        print(
            f"  Quasistatic Fröhlich LSPR (Re ε ≈ −2 ε_b = {(-2*r['eps_b']):.3f}): "
            f"{r['w_lsp']:.4f} eV"
        )
        print(f"  ω_m = {omega_m:.4f} eV,  γ_m = {gamma_m:.4f} eV,  α0 = {alpha0_nm3:g} nm³")
        print(f"  Peak of A(ω) at {omega[np.argmax(r['A'])]:.4f} eV")
        print(f"  max |G| = {G_abs.max():.3f} at {omega[np.argmax(G_abs)]:.4f} eV")
        print(f"  Wrote {csv_path}")

        fig, axes = plt.subplots(
            2 if show_G else 1,
            1,
            figsize=(10, 8 if show_G else 6),
            sharex=True,
            constrained_layout=True,
        )
        if not show_G:
            axes = [axes]
        ax = axes[0]

        ax.plot(omega, r["A"], color="C0", lw=2.0, label=r"$A\propto-\omega\,\mathrm{Im}\,\alpha_{\mathrm{eff}}$")
        if show_bare:
            ax.plot(omega, r["A_bare"], color="C2", ls="--", lw=1.5, label="bare molecule (normalized)")

        if compare_csv:
            cmp = Path(compare_csv)
            if not cmp.is_file():
                for candidate in (out_dir / compare_csv, out_dir.parent / compare_csv):
                    if candidate.is_file():
                        cmp = candidate
                        break
            if cmp.is_file():
                cdf = pd.read_csv(cmp)
                ax.plot(
                    cdf.iloc[:, 0],
                    cdf.iloc[:, 1],
                    color="0.35",
                    lw=1.4,
                    alpha=0.85,
                    label=f"PlasMol ({cmp.name})",
                )
            else:
                print(f"  WARNING: compare_csv not found: {compare_csv}")

        ax.axvline(omega_m, color="C2", ls=":", lw=0.9, alpha=0.6, label=r"$\omega_m$")
        ax.axvline(
            r["w_lsp"],
            color="C3",
            ls=":",
            lw=0.9,
            alpha=0.6,
            label=rf"LSPR ($\mathrm{{Re}}\,\varepsilon=-2\varepsilon_b$, $n={r['n_host']:.2f}$)",
        )
        ax.set_ylabel("Absorption (peak-normalized)", fontsize=13)
        ax.set_title(
            f"Model 2: {r['label']}\nquasistatic sphere + Na in n={r['n_host']:.2f} ({orientation})",
            fontsize=14,
        )
        ax.set_xlim(e_min, e_max)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=10, loc="best")

        if show_G:
            ax2 = axes[1]
            ax2.plot(omega, G_abs, color="C4", lw=1.8, label=r"$|G(\omega)|$")
            ax2.plot(omega, np.real(r["G"]), color="C4", ls="--", lw=1.2, alpha=0.8, label=r"$\mathrm{Re}\,G$")
            ax2.axvline(r["w_lsp"], color="C3", ls=":", lw=0.9, alpha=0.6)
            ax2.set_xlabel("Energy (eV)", fontsize=13)
            ax2.set_ylabel("Local-field factor", fontsize=13)
            ax2.grid(True, alpha=0.35)
            ax2.legend(fontsize=10, loc="best")
        else:
            ax.set_xlabel("Energy (eV)", fontsize=13)

        fig.savefig(png_path, dpi=300)
        print(f"  Wrote {png_path}")
        plt.close(fig)

    if run_both and len(results) > 1:
        # Comparison figure
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        colors = ["C0", "C1"]
        for i, r in enumerate(results):
            ax.plot(
                r["omega"],
                r["A"],
                color=colors[i % len(colors)],
                lw=2.0,
                label=f"{r['material']}  (LSPR={r['w_lsp']:.3f} eV)",
            )
        ax.axvline(omega_m, color="0.5", ls=":", lw=1.0, label=r"$\omega_m$")
        if compare_csv:
            cmp = Path(compare_csv)
            if not cmp.is_file():
                for candidate in (out_dir / compare_csv, out_dir.parent / compare_csv):
                    if candidate.is_file():
                        cmp = candidate
                        break
            if cmp.is_file():
                cdf = pd.read_csv(cmp)
                ax.plot(cdf.iloc[:, 0], cdf.iloc[:, 1], color="0.35", lw=1.3, alpha=0.85, label="PlasMol")
        ax.set_xlabel("Energy (eV)", fontsize=13)
        ax.set_ylabel("Absorption (peak-normalized)", fontsize=13)
        ax.set_title("Model 2 comparison: Meep Au vs Au_JC_visible", fontsize=14)
        ax.set_xlim(e_min, e_max)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=10, loc="best")
        cmp_png = out_dir / "spectrum_model2_compare.png"
        fig.savefig(cmp_png, dpi=300)
        print(f"  Wrote comparison {cmp_png}")
        plt.close(fig)


# ===========================================================================
# PARAMETERS — edit these
# ===========================================================================

# --- Meep gold dielectric ---
# "Au"            → meep.materials.Au            (Rakić 1998)
# "Au_JC_visible" → meep.materials.Au_JC_visible (Johnson–Christy visible fit)
# Joint fit to parallel_abs + perp_abs prefers Au_JC_visible (see fit_params_joint.txt).
material = "Au_JC_visible"
# If True, generate spectra for both materials + a comparison plot
run_both = False

# --- Host medium ---
# Best joint fit uses n=1.33 even though PlasMol Fourier refs are vacuum runs:
# water-like ε_b red-shifts the quasistatic LSPR and improves hybrid lineshape.
n_host = 1.33            # refractive index; ε_b = n_host²
                         # use 1.0 for vacuum Fröhlich Re ε = −2

# --- Geometry from jobs/parallel_abs/Na.json (μm → nm) ---
a_nm = 25.0              # Au sphere radius (nm); 0.025 μm
d_nm = 26.451            # molecule–center distance (nm); 0.026451 μm
orientation = "parallel"  # "parallel" or "perpendicular"
l_max = 25
include_back_action = True  # small but helpful for Na-scale α0 from the joint fit

# --- Molecule (joint fit: shared params for parallel + perp PlasMol abs) ---
# See fit_params_joint.txt / spectrum_model2_joint_fit.png
omega_m = 2.05492        # eV
gamma_m = 0.07636        # eV
alpha0_nm3 = 0.42546     # nm³ (effective classical polarizability volume)

# --- Spectrum grid ---
e_min = 1.5
e_max = 5.0
n_points = 4000

# --- Output / comparison ---
compare_csv = "../parallel_abs/spectrum_parallel.csv"  # set None to skip
out_csv = "spectrum_model2.csv"
out_png = "spectrum_model2.png"
show_bare = True
show_G = True

# ===========================================================================

if __name__ == "__main__":
    run(
        material=material,
        a_nm=a_nm,
        d_nm=d_nm,
        orientation=orientation,
        l_max=l_max,
        include_back_action=include_back_action,
        omega_m=omega_m,
        gamma_m=gamma_m,
        alpha0_nm3=alpha0_nm3,
        e_min=e_min,
        e_max=e_max,
        n_points=n_points,
        compare_csv=compare_csv,
        out_csv=out_csv,
        out_png=out_png,
        show_bare=show_bare,
        show_G=show_G,
        run_both=run_both,
        n_host=n_host,
    )
