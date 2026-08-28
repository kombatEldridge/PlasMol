# FFT, deconvolution, absorption assembly, and spectrum plotting.
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plasmol.utils import constants

logger = logging.getLogger("main")


def fourier(time, dipole, damp, min_ev, max_ev, npz=None, field_e=None, e_floor_rel=1e-8):
    """
    Fourier-transform the induced dipole for absorption spectroscopy.

    Quantum (kick) path
        field_e is None. Uses Im[μ(ω)] directly (valid for a δ-kick drive).

    Meep / Gaussian path
        field_e is the folded *incident* field from vacuum reference runs
        (no NP, no molecule; sampled at the molecule site). Uses
        Im[μ(ω)/E_inc(ω)] so the Gaussian pulse spectrum/phase are removed
        while local-field / NP effects in μ are retained.
    """
    dt = time[1] - time[0]
    abs_real = [[], [], []]
    abs_imag = [[], [], []]

    freqs_au = np.fft.fftfreq(len(time), d=dt) * 2 * np.pi
    freqs_ev = freqs_au * 27.211386
    mask = (freqs_ev >= min_ev) & (freqs_ev <= max_ev)
    freqs_out = freqs_ev[mask]
    window = np.exp(-damp * time)
    deconvolve = field_e is not None

    if deconvolve:
        logger.debug(
            f"Performing deconvolved Fourier transform Im[μ/E] with damping gamma={damp} "
            f"and frequency range {min_ev}-{max_ev} eV..."
        )
        field_e = np.asarray(field_e, dtype=float)
        if field_e.shape != np.asarray(dipole).shape:
            raise ValueError(
                f"field_e shape {field_e.shape} does not match dipole shape {np.asarray(dipole).shape}."
            )
    else:
        logger.debug(
            f"Performing kick Fourier transform Im[μ] with damping gamma={damp} "
            f"and frequency range {min_ev}-{max_ev} eV..."
        )

    for axis in (0, 1, 2):
        axis_name = {0: 'x', 1: 'y', 2: 'z'}[axis]
        logger.debug(f"Starting Fourier transform of direction {axis_name}")
        S_mu = np.fft.fft(dipole[axis] * window) * dt

        if deconvolve:
            S_e = np.fft.fft(field_e[axis] * window) * dt
            S_mu_b = S_mu[mask]
            S_e_b = S_e[mask]
            e_max = np.max(np.abs(S_e_b)) if len(S_e_b) else 0.0
            floor = e_floor_rel * e_max if e_max > 0 else 0.0
            alpha = np.zeros_like(S_mu_b, dtype=complex)
            valid = np.abs(S_e_b) > floor
            if not np.any(valid):
                logger.warning(
                    f"No usable E(ω) amplitude for {axis_name}-pol in the spectrum window; "
                    f"contribution set to zero (max |E| too small for deconvolution)."
                )
            else:
                alpha[valid] = S_mu_b[valid] / S_e_b[valid]
            abs_real[axis] = alpha.real
            abs_imag[axis] = alpha.imag
        else:
            abs_real[axis] = S_mu.real[mask]
            abs_imag[axis] = S_mu.imag[mask]

    logger.debug("Fourier transform done!")
    for i in range(3):
        abs_real[i] = np.array(abs_real[i])
        abs_imag[i] = np.array(abs_imag[i])

    if npz:
        save_kw = dict(abs_imag=abs_imag, abs_real=abs_real, freqs=freqs_out, deconvolved=deconvolve)
        np.savez(npz, **save_kw)
        logger.debug(f"Fourier transform saved to {npz}!")

    return abs_imag, freqs_out


def absorption(imag, freqs):
    """Isotropic average of three Cartesian Im[α_i] (or Im[μ_i] for kicks)."""
    fullsum = imag[0] + imag[1] + imag[2]
    return - 4 * np.pi * freqs / 3 / constants.C_AU * fullsum


def absorption_single(imag_component, freqs):
    """
    Single-polarization absorption-like spectrum.

    Uses the same prefactor as the isotropic formula without the 1/3 sum over
    three directions: A(ω) ∝ −ω Im[α] for the active polarization only.
    Peak-normalization (downstream) removes overall scale.
    """
    return -4 * np.pi * freqs / constants.C_AU * np.asarray(imag_component, dtype=float)


def orient_spectrum_sign(abs_vals, freqs=None):
    """
    Global sign choice: if the strongest |A| feature is negative, flip A → −A.

    Keeps relative lineshape (including secondary lobes) but forces the dominant
    peak to point “up” for peak-normalized plots. FFT / μ–E phase conventions
    can otherwise leave a physical resonance with A < 0.

    Returns
    -------
    abs_vals : ndarray
        Possibly sign-flipped copy.
    flipped : bool
        True if a global minus sign was applied.
    """
    abs_vals = np.asarray(abs_vals, dtype=float).copy()
    if abs_vals.size == 0:
        return abs_vals, False

    i_peak = int(np.argmax(np.abs(abs_vals)))
    peak_val = float(abs_vals[i_peak])
    if peak_val >= 0 or np.isclose(peak_val, 0.0):
        return abs_vals, False

    abs_vals *= -1.0
    if freqs is not None and len(freqs) > i_peak:
        e_peak = float(np.asarray(freqs, dtype=float)[i_peak])
        logger.info(
            f"Spectrum sign flipped: largest |A| feature was negative "
            f"(A={peak_val:.6g} at {e_peak:.4f} eV); multiplied A by −1 so the "
            f"dominant peak is positive."
        )
    else:
        logger.info(
            f"Spectrum sign flipped: largest |A| feature was negative "
            f"(A={peak_val:.6g} at index {i_peak}); multiplied A by −1 so the "
            f"dominant peak is positive."
        )
    return abs_vals, True


def save_spectrum_plot(freqs, normalized, params, title='Absorption Spectrum', label='Spectrum'):
    pd.DataFrame({'Frequency': freqs, 'Absorption': normalized}).to_csv(
        Path(params.fourier_spectrum_filepath).with_suffix(".csv"), index=False
    )
    plt.figure(figsize=(14, 8))
    plt.plot(freqs, normalized, color='green', label=label)
    plt.xlabel('Energy (eV)', fontsize=16)
    plt.ylabel('Absorption', fontsize=16)
    plt.title(title, fontsize=20)
    plt.xlim(params.fourier_min_ev, params.fourier_max_ev)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(params.fourier_spectrum_filepath, dpi=600)
    logger.info(f"Absorption spectrum written to '{params.fourier_spectrum_filepath}'.")
