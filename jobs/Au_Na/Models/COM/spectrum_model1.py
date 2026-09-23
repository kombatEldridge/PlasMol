#!/usr/bin/env python3
"""
Model 1 — Two classical coupled Lorentz oscillators (plasmon + molecule).

Produces a peak-normalized absorption-like spectrum comparable in definition
to PlasMol's Fourier hybrid path for the *molecular* channel:

    A(ω) ∝ −ω Im[ μ_m(ω) / E_inc(ω) ]   (then peak-normalized)

Edit the PARAMETERS block at the bottom of this file, then run:

    python spectrum_model1.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Physics (do not edit unless you know what you are changing)
# ---------------------------------------------------------------------------

def oscillator_denom(omega: np.ndarray, omega0: float, gamma: float) -> np.ndarray:
    """Ω_j²(ω) = ω0² − ω² − i γ ω  (complex, units of energy² if ω in eV)."""
    return omega0**2 - omega**2 - 1j * gamma * omega


def coupled_displacements(
    omega: np.ndarray,
    omega_p: float,
    gamma_p: float,
    f_p: float,
    omega_m: float,
    gamma_m: float,
    f_m: float,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the driven 2×2 coupled-oscillator system for unit E_inc:

        (Ω_p²) x_p + κ x_m = f_p E
        κ x_p + (Ω_m²) x_m = f_m E

    Returns (x_p, x_m) for E = 1.
    """
    Op = oscillator_denom(omega, omega_p, gamma_p)
    Om = oscillator_denom(omega, omega_m, gamma_m)
    det = Op * Om - kappa**2
    # Avoid exact zeros (should not happen off the real axis with damping).
    det = np.where(np.abs(det) < 1e-30, 1e-30 + 0j, det)
    x_p = (Om * f_p - kappa * f_m) / det
    x_m = (Op * f_m - kappa * f_p) / det
    return x_p, x_m


def molecular_alpha_eff(
    omega: np.ndarray,
    omega_p: float,
    gamma_p: float,
    f_p: float,
    omega_m: float,
    gamma_m: float,
    f_m: float,
    kappa: float,
) -> np.ndarray:
    """
    Effective molecular polarizability α_m_eff = μ_m / E_inc.

    With μ_m ∝ f_m * x_m in this reduced units convention we take
    α_m_eff ≡ x_m (E=1), which already carries the oscillator strength f_m
    through the inhomogeneous term. Overall scale drops out after peak
    normalization of A(ω).
    """
    _, x_m = coupled_displacements(
        omega, omega_p, gamma_p, f_p, omega_m, gamma_m, f_m, kappa
    )
    return x_m


def hybrid_mode_frequencies(
    omega_p: float,
    gamma_p: float,
    omega_m: float,
    gamma_m: float,
    kappa: float,
) -> tuple[complex, complex]:
    """
    Approximate hybrid poles from the undriven characteristic equation
    Ω_p² Ω_m² − κ² = 0, linearized near resonance (report complex ω roots
    of the quadratic in ω² for the lossless-like estimate used in notes).

    More precisely we report the two roots of
        (ω_p² − z)(ω_m² − z) − κ² = 0
    with z = ω² + i… ignored for a quick real-axis estimate of splitting;
    here we solve for complex z with damping folded as z_j = ω_j² − i γ_j ω_j
    evaluated at ω ≈ ω_j (fixed-point once).
    """
    # Use a single-pass complex-frequency estimate at each bare resonance.
    zp = omega_p**2 - 1j * gamma_p * omega_p
    zm = omega_m**2 - 1j * gamma_m * omega_m
    # Characteristic: (zp − λ)(zm − λ) − κ² = 0 for eigenvalue λ = ω²
    # → λ² − (zp+zm)λ + zp zm − κ² = 0
    disc = (zp + zm) ** 2 - 4.0 * (zp * zm - kappa**2)
    lam_p = 0.5 * (zp + zm + np.sqrt(disc))
    lam_m = 0.5 * (zp + zm - np.sqrt(disc))
    # Convert λ ≈ ω² → ω ≈ sqrt(λ) (principal branch)
    return np.sqrt(lam_p), np.sqrt(lam_m)


def absorption_like(omega: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """A(ω) ∝ −ω Im[α(ω)], matching PlasMol's sign convention for Im[μ/E]."""
    return -omega * np.imag(alpha)


def peak_normalize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    peak = np.max(np.abs(y))
    if peak <= 0.0:
        return y
    # Orient so the strongest |feature| is positive (same idea as PlasMol).
    if y[np.argmax(np.abs(y))] < 0.0:
        y = -y
    return y / np.max(np.abs(y))


def bare_alpha(omega: np.ndarray, omega0: float, gamma: float, f: float) -> np.ndarray:
    """Uncoupled Lorentzian polarizability α = f / Ω²."""
    return f / oscillator_denom(omega, omega0, gamma)


def run(
    omega_p: float,
    gamma_p: float,
    f_p: float,
    omega_m: float,
    gamma_m: float,
    f_m: float,
    kappa: float,
    e_min: float,
    e_max: float,
    n_points: int,
    compare_csv: str | None,
    out_csv: str,
    out_png: str,
    show_bare: bool,
    show_hybrid_poles: bool,
) -> None:
    out_dir = Path(__file__).resolve().parent
    omega = np.linspace(e_min, e_max, n_points)

    alpha_eff = molecular_alpha_eff(
        omega, omega_p, gamma_p, f_p, omega_m, gamma_m, f_m, kappa
    )
    A = peak_normalize(absorption_like(omega, alpha_eff))

    # Also build uncoupled references for interpretation.
    A_bare_m = peak_normalize(
        absorption_like(omega, bare_alpha(omega, omega_m, gamma_m, f_m))
    )
    A_bare_p = peak_normalize(
        absorption_like(omega, bare_alpha(omega, omega_p, gamma_p, f_p))
    )

    df = pd.DataFrame(
        {
            "Frequency": omega,
            "Absorption": A,
            "Re_alpha_m_eff": np.real(alpha_eff),
            "Im_alpha_m_eff": np.imag(alpha_eff),
        }
    )
    csv_path = out_dir / out_csv
    png_path = out_dir / out_png
    df.to_csv(csv_path, index=False)

    wp, wm = hybrid_mode_frequencies(omega_p, gamma_p, omega_m, gamma_m, kappa)
    print("=== Model 1: coupled Lorentz oscillators ===")
    print(f"  ω_p = {omega_p:.4f} eV,  γ_p = {gamma_p:.4f} eV,  f_p = {f_p:g}")
    print(f"  ω_m = {omega_m:.4f} eV,  γ_m = {gamma_m:.4f} eV,  f_m = {f_m:g}")
    print(f"  κ   = {kappa:.6f} eV²")
    print(f"  Hybrid pole estimates (complex eV):")
    print(f"    ω_+ ≈ {wp.real:.4f} + {wp.imag:.4f} j")
    print(f"    ω_- ≈ {wm.real:.4f} + {wm.imag:.4f} j")
    print(f"  Peak of A(ω) at {omega[np.argmax(A)]:.4f} eV")
    print(f"  Wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(omega, A, color="C0", lw=2.0, label=r"coupled $A\propto-\omega\,\mathrm{Im}\,\alpha_m^{\mathrm{eff}}$")

    if show_bare:
        ax.plot(omega, A_bare_m, color="C2", ls="--", lw=1.5, label="bare molecule (normalized)")
        ax.plot(omega, A_bare_p, color="C3", ls=":", lw=1.5, label="bare plasmon (normalized)")

    if compare_csv:
        cmp = Path(compare_csv)
        if not cmp.is_file():
            # allow path relative to this script or to jobs/
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

    if show_hybrid_poles:
        for w, lab in ((wp, r"$\omega_+$"), (wm, r"$\omega_-$")):
            if e_min <= w.real <= e_max:
                ax.axvline(w.real, color="C1", ls="-.", lw=1.0, alpha=0.8, label=lab)

    ax.axvline(omega_m, color="C2", ls=":", lw=0.9, alpha=0.5)
    ax.axvline(omega_p, color="C3", ls=":", lw=0.9, alpha=0.5)
    ax.set_xlabel("Energy (eV)", fontsize=14)
    ax.set_ylabel("Absorption (peak-normalized)", fontsize=14)
    ax.set_title("Model 1: Coupled Lorentz oscillators (Au LSPR + Na)", fontsize=15)
    ax.set_xlim(e_min, e_max)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    print(f"  Wrote {png_path}")
    plt.close(fig)


# ===========================================================================
# PARAMETERS — edit these
# ===========================================================================

# --- Bare resonances (eV) from your Au NP + Na system ---
omega_p = 2.384          # Au LSPR peak energy (eV)
gamma_p = 0.25           # Au LSPR linewidth FWHM-like damping (eV)
f_p = 1.0                # plasmon oscillator strength (relative units)

omega_m = 2.109          # Na absorption peak (eV)
gamma_m = 0.08           # Na linewidth (eV); CAP/RT-TDDFT often broader
f_m = 0.15               # molecular oscillator strength (relative; ≪ f_p)

# Coupling constant κ (units of eV²).  For near-field dipole–dipole,
# κ ~ g * sqrt(f_p f_m) with g an energy-scale coupling; start small and
# increase until hybrid features appear.  κ = 0 → uncoupled molecule.
# Detuning δ = ω_p − ω_m ≈ 0.275 eV; strong hybridization needs
# effective g ≳ δ/2 with g² ~ κ (order-of-magnitude).
kappa = -0.08             # eV²  (try 0.0, 0.05, 0.12, 0.25, …)

# --- Spectrum grid ---
e_min = 1.5              # eV
e_max = 5.0              # eV
n_points = 4000

# --- Output / comparison ---
compare_csv = "../parallel_abs/spectrum_parallel.csv"  # set None to skip
out_csv = "spectrum_model1.csv"
out_png = "spectrum_model1.png"
show_bare = True
show_hybrid_poles = True

# ===========================================================================

if __name__ == "__main__":
    run(
        omega_p=omega_p,
        gamma_p=gamma_p,
        f_p=f_p,
        omega_m=omega_m,
        gamma_m=gamma_m,
        f_m=f_m,
        kappa=kappa,
        e_min=e_min,
        e_max=e_max,
        n_points=n_points,
        compare_csv=compare_csv,
        out_csv=out_csv,
        out_png=out_png,
        show_bare=show_bare,
        show_hybrid_poles=show_hybrid_poles,
    )
