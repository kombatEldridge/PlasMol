# Post-run FFT / deconvolution / spectrum writing.
import logging
import os

import numpy as np

from plasmol.drivers.custom_drivers.absorption.io_fields import (
    align_series,
    apply_tau_damping_arrays,
    fold,
    fold_single,
    load_reference_e_tensor,
    merge_reference_e_fields,
    validate_reference_times,
    write_single_reference_e_field,
)
from plasmol.drivers.custom_drivers.absorption.spectrum import (
    OBSERVABLE_LABELS,
    OBSERVABLE_TITLES,
    dft_3,
    dress_spectrum,
    frequency_mask,
    imag_for_observable,
    observable_output_paths,
    orient_spectrum_sign,
    save_spectrum_plot,
)
from plasmol.utils.npz import save_npz

logger = logging.getLogger("main")


def _requested_observables(params):
    obs = getattr(params, 'absorption_observables', None)
    if not obs:
        return ['cross_section']
    return list(obs)


def _needs_local_field(observables):
    return any(name in ('cross_section', 'dissipative_power') for name in observables)


def _needs_incident_field(observables, has_plasmon):
    if 'cross_section' in observables:
        return True
    if 'A_raw' in observables and has_plasmon:
        return True
    return False


def _write_observables(time_points, dipole, field_loc, field_inc, params, axis=None, title_suffix=''):
    observables = _requested_observables(params)
    freqs, mask = frequency_mask(time_points, params.absorption_min_ev, params.absorption_max_ev)
    if len(freqs) == 0:
        raise ValueError(
            "No valid frequencies found for Fourier transform. "
            "Try running the simulation for longer."
        )

    S_mu = dft_3(time_points, dipole, params.absorption_gamma, mask)
    S_loc = None if field_loc is None else dft_3(time_points, field_loc, params.absorption_gamma, mask)
    S_inc = None if field_inc is None else dft_3(time_points, field_inc, params.absorption_gamma, mask)

    paths = observable_output_paths(params.absorption_spectrum_filepath, observables)
    npz_payload = {'freqs': freqs}
    for name in observables:
        imag = imag_for_observable(name, S_mu, S_loc=S_loc, S_inc=S_inc)
        abs_vals = dress_spectrum(imag, freqs, axis=axis)
        abs_vals, _ = orient_spectrum_sign(abs_vals, freqs)
        peak = np.max(np.abs(abs_vals))
        normalized = abs_vals / peak if peak else abs_vals
        save_spectrum_plot(
            freqs, normalized, params,
            title=OBSERVABLE_TITLES[name] + title_suffix,
            label=OBSERVABLE_LABELS[name],
            filepath=paths[name],
        )
        npz_payload[name] = abs_vals
        npz_payload[f'{name}_normalized'] = normalized

    npz = getattr(params, 'absorption_npz_filepath', None)
    if npz:
        save_npz(npz, **npz_payload)


def _load_vacuum_e_inc(params, ref_e_filepath=None):
    """Load vacuum E_inc (time, (3, N)) from a merged or per-run reference CSV."""
    if params.absorption_use_existing_e_field_ref:
        time_e, field_e = load_reference_e_tensor(params.absorption_field_e_ref_filepath)
        validate_reference_times(
            time_e,
            params.times,
            atol=1e-5,
            filepath=params.absorption_field_e_ref_filepath,
        )
        logger.info(
            f"Using precomputed vacuum E_inc '{params.absorption_field_e_ref_filepath}'."
        )
        return time_e, field_e
    if not ref_e_filepath:
        raise ValueError(
            "Fourier post-process needs a vacuum reference CSV "
            "when use_existing_e_field_ref is false."
        )
    return None, None


def absorption_post_process_single(
    field_p_filepath, component, params, ref_e_filepath=None, loc_e_filepath=None
):
    """
    Single-polarization hybrid Fourier post-process.

    Loads μ from production ``field_p``, E_loc from production ``field_e``,
    and E_inc from the vacuum reference, then writes each requested observable.
    """
    component = component.lower().strip()
    axis = {'x': 0, 'y': 1, 'z': 2}[component]
    observables = _requested_observables(params)
    time_points, dipole_moment = fold_single(field_p_filepath, component)

    field_loc = None
    field_inc = None
    decimals = params.time_rounding_decimals

    if _needs_local_field(observables):
        if not loc_e_filepath:
            raise ValueError(
                "Observables 'cross_section' and 'dissipative_power' require the "
                "production local field (field_e.csv)."
            )
        time_loc, field_loc = fold_single(loc_e_filepath, component)
        if not np.isclose(params.absorption_tau, 0):
            field_loc = apply_tau_damping_arrays(time_loc, field_loc, params.absorption_tau)
        time_points, dipole_moment, field_loc = align_series(
            time_points, dipole_moment, time_loc, field_loc, decimals,
            label_a="dipole", label_b="local E",
        )

    if params.has_plasmon and _needs_incident_field(observables, True):
        if params.absorption_use_existing_e_field_ref:
            time_e, field_inc = load_reference_e_tensor(params.absorption_field_e_ref_filepath)
            validate_reference_times(
                time_e, params.times, atol=1e-5,
                filepath=params.absorption_field_e_ref_filepath,
            )
            logger.info(
                f"Single-pol Fourier ({component}): E_inc from "
                f"'{params.absorption_field_e_ref_filepath}'."
            )
        else:
            if not ref_e_filepath:
                raise ValueError(
                    "Single-pol Fourier post-process needs a vacuum reference CSV "
                    "when use_existing_e_field_ref is false."
                )
            write_single_reference_e_field(
                ref_e_filepath, component, params.absorption_field_e_ref_filepath
            )
            time_e, field_inc = load_reference_e_tensor(params.absorption_field_e_ref_filepath)
            logger.info(
                f"Single-pol Fourier ({component}): E_inc written to "
                f"'{params.absorption_field_e_ref_filepath}'."
            )
        if not np.isclose(params.absorption_tau, 0):
            field_inc = apply_tau_damping_arrays(time_e, field_inc, params.absorption_tau)
        time_points, dipole_moment, field_inc = align_series(
            time_points, dipole_moment, time_e, field_inc, decimals,
            label_a="dipole", label_b="reference E_inc",
        )
        if field_loc is not None:
            field_loc = field_loc[:, : dipole_moment.shape[1]]

    pol_mode = getattr(params, 'absorption_polarization', 'single')
    suffix = f' ({pol_mode}, E || {component})'
    _write_observables(
        time_points, dipole_moment, field_loc, field_inc, params,
        axis=axis, title_suffix=suffix,
    )


def absorption_post_process(x_e_file, y_e_file, z_e_file, x_p_file, y_p_file, z_p_file, params):
    observables = _requested_observables(params)
    time_points, dipole_moment = fold(x_p_file, y_p_file, z_p_file)
    decimals = params.time_rounding_decimals
    field_loc = None
    field_inc = None

    if _needs_local_field(observables):
        time_loc, field_loc = fold(x_e_file, y_e_file, z_e_file)
        if not np.isclose(params.absorption_tau, 0):
            field_loc = apply_tau_damping_arrays(time_loc, field_loc, params.absorption_tau)
        time_points, dipole_moment, field_loc = align_series(
            time_points, dipole_moment, time_loc, field_loc, decimals,
            label_a="dipole", label_b="local E",
        )

    if params.has_plasmon and _needs_incident_field(observables, True):
        if params.absorption_use_existing_e_field_ref:
            time_e, field_inc = load_reference_e_tensor(params.absorption_field_e_ref_filepath)
            validate_reference_times(
                time_e, params.times, atol=1e-5,
                filepath=params.absorption_field_e_ref_filepath,
            )
            logger.info(
                "Meep/plasmon Fourier path: E_inc from "
                f"'{params.absorption_field_e_ref_filepath}'."
            )
        else:
            ref_candidates = [f"{d}_dir/field_e_ref.csv" for d in params.xyz]
            if all(os.path.isfile(p) for p in ref_candidates):
                merge_reference_e_fields(
                    ref_candidates[0], ref_candidates[1], ref_candidates[2],
                    params.absorption_field_e_ref_filepath,
                )
            else:
                merge_reference_e_fields(
                    x_e_file, y_e_file, z_e_file, params.absorption_field_e_ref_filepath
                )
            time_e, field_inc = load_reference_e_tensor(params.absorption_field_e_ref_filepath)
            logger.info(
                "Meep/plasmon Fourier path: E_inc written to "
                f"'{params.absorption_field_e_ref_filepath}'."
            )
        if not np.isclose(params.absorption_tau, 0):
            field_inc = apply_tau_damping_arrays(time_e, field_inc, params.absorption_tau)
        time_points, dipole_moment, field_inc = align_series(
            time_points, dipole_moment, time_e, field_inc, decimals,
            label_a="dipole", label_b="reference E_inc",
        )
        if field_loc is not None:
            field_loc = field_loc[:, : dipole_moment.shape[1]]
    elif (not params.has_plasmon) and _needs_incident_field(observables, False):
        # Quantum kick: the directional field_e CSVs *are* E_inc = E_loc.
        if field_loc is None:
            time_loc, field_loc = fold(x_e_file, y_e_file, z_e_file)
            if not np.isclose(params.absorption_tau, 0):
                field_loc = apply_tau_damping_arrays(time_loc, field_loc, params.absorption_tau)
            time_points, dipole_moment, field_loc = align_series(
                time_points, dipole_moment, time_loc, field_loc, decimals,
                label_a="dipole", label_b="kick E",
            )
        field_inc = field_loc

    _write_observables(time_points, dipole_moment, field_loc, field_inc, params)
