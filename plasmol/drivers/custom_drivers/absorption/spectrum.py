# FFT, deconvolution, absorption assembly, and spectrum plotting.
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plasmol.utils import constants
from plasmol.utils.npz import save_npz

logger = logging.getLogger("main")

ABSORPTION_OBSERVABLES = ('cross_section', 'dissipative_power', 'A_raw')
OBSERVABLE_TITLES = {
    'cross_section': 'Molecular absorption cross section',
    'dissipative_power': 'Dissipated-power spectrum',
    'A_raw': 'Absorption spectrum ($A_{\\mathrm{raw}}$)',
}
OBSERVABLE_LABELS = {
    'cross_section': r'$\sigma_m$',
    'dissipative_power': r'$A_{\mathrm{diss}}$',
    'A_raw': r'$A_{\mathrm{raw}}$',
}


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
        save_npz(
            npz,
            abs_imag=abs_imag,
            abs_real=abs_real,
            freqs=freqs_out,
            deconvolved=deconvolve,
        )

    return abs_imag, freqs_out


def frequency_mask(time, min_ev, max_ev):
    """FFT frequency axis (eV) and boolean mask for ``[min_ev, max_ev]``."""
    dt = time[1] - time[0]
    freqs_au = np.fft.fftfreq(len(time), d=dt) * 2 * np.pi
    freqs_ev = freqs_au * 27.211386
    mask = (freqs_ev >= min_ev) & (freqs_ev <= max_ev)
    return freqs_ev[mask], mask


def dft_3(time, data, damp, mask):
    """Windowed DFT of a (3, N) time series, restricted to ``mask``."""
    data = np.asarray(data, dtype=float)
    dt = time[1] - time[0]
    window = np.exp(-damp * np.asarray(time, dtype=float))
    nfreq = int(np.count_nonzero(mask))
    out = np.empty((3, nfreq), dtype=complex)
    for axis in (0, 1, 2):
        S = np.fft.fft(data[axis] * window) * dt
        out[axis] = S[mask]
    return out


def imag_for_observable(name, S_mu, S_loc=None, S_inc=None, e_floor_rel=1e-8):
    """
    Cartesian imaginary parts for one observable, before the −4πω/c dressing.

    ``A_raw``
        Im[μ/E_inc], or Im[μ] if ``S_inc`` is None (δ-kick).
    ``dissipative_power``
        Im[μ E_loc*].
    ``cross_section``
        Im[μ E_loc*] / |E_inc|².
    """
    if name == 'A_raw':
        if S_inc is None:
            return np.asarray(S_mu, dtype=complex).imag
        S_inc = np.asarray(S_inc, dtype=complex)
        S_mu = np.asarray(S_mu, dtype=complex)
        imag = np.zeros(S_mu.shape, dtype=float)
        for ax in range(3):
            e_max = np.max(np.abs(S_inc[ax])) if S_inc[ax].size else 0.0
            floor = e_floor_rel * e_max if e_max > 0 else 0.0
            valid = np.abs(S_inc[ax]) > floor
            if not np.any(valid):
                logger.warning(
                    f"No usable E_inc(ω) for {['x', 'y', 'z'][ax]}-pol in A_raw; "
                    "contribution set to zero."
                )
            else:
                imag[ax, valid] = (S_mu[ax, valid] / S_inc[ax, valid]).imag
        return imag

    if S_loc is None:
        raise ValueError(
            f"Observable '{name}' requires the local field E_loc "
            "(production field_e.csv)."
        )
    S_mu = np.asarray(S_mu, dtype=complex)
    S_loc = np.asarray(S_loc, dtype=complex)
    product_imag = (S_mu * np.conjugate(S_loc)).imag
    if name == 'dissipative_power':
        return product_imag
    if name == 'cross_section':
        if S_inc is None:
            raise ValueError(
                "Observable 'cross_section' requires E_inc "
                "(vacuum reference or kick field)."
            )
        S_inc = np.asarray(S_inc, dtype=complex)
        imag = np.zeros_like(product_imag)
        for ax in range(3):
            e_max = np.max(np.abs(S_inc[ax])) if S_inc[ax].size else 0.0
            floor = e_floor_rel * e_max if e_max > 0 else 0.0
            valid = np.abs(S_inc[ax]) > floor
            if not np.any(valid):
                logger.warning(
                    f"No usable E_inc(ω) for {['x', 'y', 'z'][ax]}-pol in "
                    "cross_section; contribution set to zero."
                )
            else:
                denom = np.abs(S_inc[ax, valid]) ** 2
                imag[ax, valid] = product_imag[ax, valid] / denom
        return imag
    raise ValueError(f"Unknown absorption observable '{name}'.")


def dress_spectrum(imag, freqs, axis=None):
    """Apply −4πω/c. ``axis`` None → isotropic 1/3 sum; else one Cartesian row."""
    if axis is None:
        return absorption(imag, freqs)
    return absorption_single(imag[axis], freqs)


def observable_output_paths(spectrum_filepath, observables):
    """Map each observable to a PNG path. One observable keeps ``spectrum_filepath``."""
    path = Path(spectrum_filepath)
    if len(observables) == 1:
        return {observables[0]: str(path)}
    return {
        name: str(path.with_name(f"{path.stem}_{name}{path.suffix}"))
        for name in observables
    }


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


def save_spectrum_plot(freqs, normalized, params, title='Absorption Spectrum', label='Spectrum', filepath=None):
    filepath = filepath or params.absorption_spectrum_filepath
    pd.DataFrame({'Frequency': freqs, 'Absorption': normalized}).to_csv(
        Path(filepath).with_suffix(".csv"), index=False
    )
    plt.figure(figsize=(14, 8))
    plt.plot(freqs, normalized, color='green', label=label)
    plt.xlabel('Energy (eV)', fontsize=16)
    plt.ylabel('Absorption', fontsize=16)
    plt.title(title, fontsize=20)
    plt.xlim(params.absorption_min_ev, params.absorption_max_ev)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(filepath, dpi=600)
    logger.info(f"Absorption spectrum written to '{filepath}'.")
