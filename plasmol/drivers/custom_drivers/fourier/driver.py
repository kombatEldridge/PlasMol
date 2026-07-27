# Fourier driver orchestration (entry point: run).
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from plasmol.utils.checkpoint import merge_per_direction_checkpoints, merge_final_checkpoints
from plasmol.drivers.custom_drivers.fourier.io_fields import (
    apply_tau_damping,
    merge_reference_e_fields,
)
from plasmol.drivers.custom_drivers.fourier.polarization import (
    build_parallel_abs_spec_runs,
    build_perpendicular_abs_spec_runs,
)
from plasmol.drivers.custom_drivers.fourier.postprocess import (
    fourier_post_process,
    fourier_post_process_single,
)
from plasmol.drivers.custom_drivers.fourier.setup import (
    set_up_params_copy_molecule,
    set_up_params_copy_plasmol,
    set_up_params_copy_reference,
)
from plasmol.drivers.custom_drivers.fourier.workers import (
    run_plasmol_with_prefix,
    run_quantum_with_prefix,
    run_reference_with_prefix,
)

logger = logging.getLogger("main")


def run(params):
    if params.has_plasmon and params.has_checkpoint:
        params.has_checkpoint = False

    ref_copies = []
    params_copies = []
    pol_mode = getattr(params, 'fourier_polarization', 'full') or 'full'
    pol_mode = pol_mode.lower().strip()
    single_pol = pol_mode in ('parallel', 'perpendicular')
    active_component = None

    if params.fourier_reference_only:
        if not params.has_plasmon:
            raise ValueError("Fourier reference_only requires a plasmon section.")
        ref_copies = set_up_params_copy_reference(params)
        logger.info(
            f"Fourier reference_only: running {len(ref_copies)} vacuum reference "
            f"(no NP, no molecule) E-field simulations in parallel; "
            f"output → '{params.fourier_field_e_ref_filepath}'."
        )
    elif single_pol:
        if not params.has_plasmon:
            raise ValueError(
                f"Fourier polarization='{pol_mode}' requires a plasmon section."
            )
        if pol_mode == 'parallel':
            params_copies, ref_copies, active_component = build_parallel_abs_spec_runs(params)
        else:
            params_copies, ref_copies, active_component = build_perpendicular_abs_spec_runs(params)
        params.fourier_active_component = active_component
    elif params.has_plasmon:
        params_copies = set_up_params_copy_plasmol(params)
        if params.fourier_use_existing_e_field_ref:
            logger.info(
                f"Running {len(params_copies)} directional plasmol simulations in parallel "
                f"(skipping vacuum reference runs; using E_inc file '{params.fourier_field_e_ref_filepath}')."
            )
        else:
            ref_copies = set_up_params_copy_reference(params)
            logger.info(
                f"Running {len(params_copies)} directional plasmol simulations + "
                f"{len(ref_copies)} vacuum reference (no NP, no molecule) E-field "
                f"simulations in parallel..."
            )
    else:
        params_copies = set_up_params_copy_molecule(params)
        logger.info(f"Running {len(params_copies)} directional molecule simulations in parallel...")

    n_workers = max(1, len(params_copies) + len(ref_copies))
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_label = {}
            if params.fourier_reference_only:
                for ref_copy in ref_copies:
                    future_to_label[
                        executor.submit(run_reference_with_prefix, ref_copy)
                    ] = f"ref-{ref_copy.plasmon_source_component}-dir"
            elif params.has_plasmon:
                for params_copy in params_copies:
                    future_to_label[
                        executor.submit(run_plasmol_with_prefix, params_copy)
                    ] = f"{params_copy.plasmon_source_component}-dir"
                for ref_copy in ref_copies:
                    future_to_label[
                        executor.submit(run_reference_with_prefix, ref_copy)
                    ] = f"ref-{ref_copy.plasmon_source_component}-dir"
            else:
                for params_copy in params_copies:
                    future_to_label[
                        executor.submit(run_quantum_with_prefix, params_copy)
                    ] = f"{params_copy.molecule_source_component}-dir"

            for future in as_completed(future_to_label):
                label = future_to_label[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"{label} run failed: {e}")
                    raise
    finally:
        if getattr(params, 'has_checkpoint', False):
            reg_fp = getattr(params, 'checkpoint_filepath', None)
            if reg_fp:
                try:
                    merge_per_direction_checkpoints(params, reg_fp)
                except Exception as me:
                    logger.error(f"Failed to merge per-direction regular checkpoints: {me}")

            final_fp = getattr(params, 'final_checkpoint_filepath', None)
            if final_fp:
                try:
                    merge_final_checkpoints(params, final_fp)
                except Exception as me:
                    logger.error(f"Failed to merge per-direction final checkpoints: {me}")

    if params.fourier_reference_only:
        merge_reference_e_fields(
            ref_copies[0].field_e_filepath,
            ref_copies[1].field_e_filepath,
            ref_copies[2].field_e_filepath,
            params.fourier_field_e_ref_filepath,
        )
        logger.info(
            f"Fourier reference_only complete. Vacuum E_inc written to '{params.fourier_field_e_ref_filepath}'. "
            f"Reuse it in a full Fourier run via additional_parameters.fourier.field_e_ref_filepath."
        )
        return

    for params_copy in params_copies:
        params_copy.field_p_filepath = apply_tau_damping(
            params_copy.field_p_filepath,
            params.fourier_tau,
            params.time_rounding_decimals,
            label="polarizability",
        )

    if single_pol:
        ref_e_file = ref_copies[0].field_e_filepath if ref_copies else None
        fourier_post_process_single(
            params_copies[0].field_p_filepath,
            active_component,
            params,
            ref_e_filepath=ref_e_file,
        )
    else:
        fourier_post_process(
            params_copies[0].field_e_filepath,
            params_copies[1].field_e_filepath,
            params_copies[2].field_e_filepath,
            params_copies[0].field_p_filepath,
            params_copies[1].field_p_filepath,
            params_copies[2].field_p_filepath,
            params,
        )
