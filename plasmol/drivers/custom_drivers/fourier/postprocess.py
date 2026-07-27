# Post-run FFT / deconvolution / spectrum writing.
import logging
import os

import numpy as np

from plasmol.drivers.custom_drivers.fourier.io_fields import (
    align_series,
    apply_tau_damping_arrays,
    fold,
    fold_single,
    load_reference_e_tensor,
    merge_reference_e_fields,
    validate_reference_times,
    write_single_reference_e_field,
)
from plasmol.drivers.custom_drivers.fourier.spectrum import (
    absorption,
    absorption_single,
    fourier,
    orient_spectrum_sign,
    save_spectrum_plot,
)

logger = logging.getLogger("main")


def fourier_post_process_single(field_p_filepath, component, params, ref_e_filepath=None):
    """
    Single-polarization Meep Fourier post-process (parallel or perpendicular mode).

    Loads μ_i from one production CSV, deconvolves with vacuum E_inc for that
    same polarization, and writes a polarization-resolved spectrum.
    """
    component = component.lower().strip()
    axis = {'x': 0, 'y': 1, 'z': 2}[component]
    time_points, dipole_moment = fold_single(field_p_filepath, component)

    field_e = None
    if params.has_plasmon:
        if params.fourier_use_existing_e_field_ref:
            time_e, field_e = load_reference_e_tensor(params.fourier_field_e_ref_filepath)
            validate_reference_times(
                time_e,
                params.times,
                atol=1e-5,
                filepath=params.fourier_field_e_ref_filepath,
            )
            logger.info(
                f"Single-pol Fourier ({component}): Im[μ/E] using precomputed "
                f"reference E_inc '{params.fourier_field_e_ref_filepath}'."
            )
        else:
            if not ref_e_filepath:
                raise ValueError(
                    "Single-pol Fourier post-process needs a vacuum reference CSV "
                    "when use_existing_e_field_ref is false."
                )
            write_single_reference_e_field(
                ref_e_filepath, component, params.fourier_field_e_ref_filepath
            )
            time_e, field_e = load_reference_e_tensor(params.fourier_field_e_ref_filepath)
            logger.info(
                f"Single-pol Fourier ({component}): Im[μ/E] using vacuum reference "
                f"E_inc written to '{params.fourier_field_e_ref_filepath}'."
            )

        if not np.isclose(params.fourier_tau, 0):
            field_e = apply_tau_damping_arrays(time_e, field_e, params.fourier_tau)
            logger.info(f"Applied tau={params.fourier_tau} damping to reference E_inc (in memory).")

        time_points, dipole_moment, field_e = align_series(
            time_points,
            dipole_moment,
            time_e,
            field_e,
            params.time_rounding_decimals,
            label_a="dipole",
            label_b="reference E_inc",
        )

    abs_imag, freqs = fourier(
        time_points,
        dipole_moment,
        params.fourier_gamma,
        params.fourier_min_ev,
        params.fourier_max_ev,
        npz=getattr(params, 'fourier_npz_filepath', None),
        field_e=field_e,
    )

    abs_vals = absorption_single(abs_imag[axis], freqs)
    if len(freqs) == 0:
        raise ValueError("No valid frequencies found for Fourier transform. Try running the simulation for longer.")

    abs_vals, _ = orient_spectrum_sign(abs_vals, freqs)
    peak = np.max(np.abs(abs_vals))
    normalized = abs_vals / peak if peak else abs_vals
    pol_mode = getattr(params, 'fourier_polarization', 'single')
    title = f'Absorption Spectrum ({pol_mode}, E || {component})'
    save_spectrum_plot(freqs, normalized, params, title=title, label=f'{pol_mode} ({component})')


def fourier_post_process(x_e_file, y_e_file, z_e_file, x_p_file, y_p_file, z_p_file, params):
    time_points, dipole_moment = fold(x_p_file, y_p_file, z_p_file)

    field_e = None
    if params.has_plasmon:
        if params.fourier_use_existing_e_field_ref:
            time_e, field_e = load_reference_e_tensor(params.fourier_field_e_ref_filepath)
            validate_reference_times(
                time_e,
                params.times,
                atol=1e-5,
                filepath=params.fourier_field_e_ref_filepath,
            )
            logger.info(
                "Meep/plasmon Fourier path: Im[μ(ω)/E_inc(ω)] using precomputed "
                f"reference E_inc '{params.fourier_field_e_ref_filepath}'."
            )
        else:
            ref_candidates = [f"{d}_dir/field_e_ref.csv" for d in params.xyz]
            if all(os.path.isfile(p) for p in ref_candidates):
                merge_reference_e_fields(
                    ref_candidates[0], ref_candidates[1], ref_candidates[2],
                    params.fourier_field_e_ref_filepath,
                )
            else:
                merge_reference_e_fields(
                    x_e_file, y_e_file, z_e_file, params.fourier_field_e_ref_filepath
                )
            time_e, field_e = load_reference_e_tensor(params.fourier_field_e_ref_filepath)
            logger.info(
                "Meep/plasmon Fourier path: Im[μ(ω)/E_inc(ω)] using vacuum reference "
                f"E_inc written to '{params.fourier_field_e_ref_filepath}'."
            )

        if not np.isclose(params.fourier_tau, 0):
            field_e = apply_tau_damping_arrays(time_e, field_e, params.fourier_tau)
            logger.info(f"Applied tau={params.fourier_tau} damping to reference E_inc (in memory).")

        time_points, dipole_moment, field_e = align_series(
            time_points,
            dipole_moment,
            time_e,
            field_e,
            params.time_rounding_decimals,
            label_a="dipole",
            label_b="reference E_inc",
        )

    abs_imag, freqs = fourier(
        time_points,
        dipole_moment,
        params.fourier_gamma,
        params.fourier_min_ev,
        params.fourier_max_ev,
        npz=getattr(params, 'fourier_npz_filepath', None),
        field_e=field_e,
    )

    abs_vals = absorption(abs_imag, freqs)
    if len(freqs) == 0:
        raise ValueError("No valid frequencies found for Fourier transform. Try running the simulation for longer.")

    abs_vals, _ = orient_spectrum_sign(abs_vals, freqs)
    peak = np.max(np.abs(abs_vals))
    normalized = abs_vals / peak if peak else abs_vals
    save_spectrum_plot(freqs, normalized, params)
