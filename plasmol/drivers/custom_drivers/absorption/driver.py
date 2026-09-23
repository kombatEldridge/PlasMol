# absorption driver orchestration (entry point: run).
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from plasmol.utils.checkpoint import merge_per_direction_checkpoints, merge_final_checkpoints
from plasmol.drivers.custom_drivers.absorption.io_fields import (
    apply_tau_damping,
    merge_reference_e_fields,
)
from plasmol.drivers.custom_drivers.absorption.polarization import (
    build_parallel_abs_spec_runs,
    build_perpendicular_abs_spec_runs,
    build_single_abs_spec_runs,
)
from plasmol.drivers.custom_drivers.absorption.postprocess import (
    absorption_post_process,
    absorption_post_process_single,
)
from plasmol.drivers.custom_drivers.absorption.setup import (
    set_up_params_copy_molecule,
    set_up_params_copy_plasmol,
    set_up_params_copy_reference,
)
from plasmol.drivers.custom_drivers.absorption.workers import (
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
    pol_mode = getattr(params, 'absorption_polarization', 'full') or 'full'
    pol_mode = pol_mode.lower().strip()
    single_pol = pol_mode in ('parallel', 'perpendicular', 'single')
    active_component = None

    if params.absorption_reference_only:
        if not params.has_plasmon:
            raise ValueError("Absorption reference_only requires a plasmon section.")
        ref_copies = set_up_params_copy_reference(params)
        logger.info(
            f"Absorption reference_only: running {len(ref_copies)} vacuum reference "
            f"(no NP, no molecule) E-field simulations in parallel; "
            f"output → '{params.absorption_field_e_ref_filepath}'."
        )
    elif single_pol:
        if not params.has_plasmon:
            raise ValueError(
                f"Absorption polarization='{pol_mode}' requires a plasmon section."
            )
        if pol_mode == 'parallel':
            params_copies, ref_copies, active_component = build_parallel_abs_spec_runs(params)
        elif pol_mode == 'perpendicular':
            params_copies, ref_copies, active_component = build_perpendicular_abs_spec_runs(params)
        else:
            params_copies, ref_copies, active_component = build_single_abs_spec_runs(params)
        params.absorption_active_component = active_component
    elif params.has_plasmon:
        params_copies = set_up_params_copy_plasmol(params)
        if params.absorption_use_existing_e_field_ref:
            logger.info(
                f"Running {len(params_copies)} directional plasmol simulations in parallel "
                f"(skipping vacuum reference runs; using E_inc file '{params.absorption_field_e_ref_filepath}')."
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
            if params.absorption_reference_only:
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

    if params.absorption_reference_only:
        merge_reference_e_fields(
            ref_copies[0].field_e_filepath,
            ref_copies[1].field_e_filepath,
            ref_copies[2].field_e_filepath,
            params.absorption_field_e_ref_filepath,
        )
        logger.info(
            f"Absorption reference_only complete. Vacuum E_inc written to '{params.absorption_field_e_ref_filepath}'. "
            f"Reuse it in a full Fourier run via settings.driver.field_e_ref_filepath."
        )
        return

    for params_copy in params_copies:
        params_copy.field_p_filepath = apply_tau_damping(
            params_copy.field_p_filepath,
            params.absorption_tau,
            params.time_rounding_decimals,
            label="polarizability",
        )

    if single_pol:
        ref_e_file = ref_copies[0].field_e_filepath if ref_copies else None
        absorption_post_process_single(
            params_copies[0].field_p_filepath,
            active_component,
            params,
            ref_e_filepath=ref_e_file,
            loc_e_filepath=params_copies[0].field_e_filepath,
        )
    else:
        absorption_post_process(
            params_copies[0].field_e_filepath,
            params_copies[1].field_e_filepath,
            params_copies[2].field_e_filepath,
            params_copies[0].field_p_filepath,
            params_copies[1].field_p_filepath,
            params_copies[2].field_p_filepath,
            params,
        )
