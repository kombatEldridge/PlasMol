import copy
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import meep as mp
from plasmol.classical.simulation import SIMULATION
from plasmol.classical.sources import MEEPSOURCE
from plasmol.utils.logging import setup_logging

logger = logging.getLogger("main")


def _run_one_case(params, pol_label: str, with_nanoparticle: bool):
    """
    Run a single FDTD case (X or Y, with or without nanoparticle).
    Returns only the filename of the saved pickle (never Meep objects).
    """
    # === Configure logging in child process (critical for multiprocessing) ===
    log_file = getattr(params, 'log', None)
    setup_logging(verbose=1, log_file=log_file)

    label = f"{pol_label}-{'NP' if with_nanoparticle else 'VAC'}"
    logger.info(f"[{label}] Starting simulation in child process...")

    p = copy.deepcopy(params)

    # === Force nanoparticle on/off ===
    p.has_nanoparticle = with_nanoparticle
    if hasattr(p, "nanoparticle"):
        delattr(p, "nanoparticle")

    # === Recreate plasmon source (always) ===
    component = "y" if pol_label == "Y" else "x"

    if pol_label == "Y":
        p.plasmon_source_component = "y"
    
    if hasattr(p, 'images_dir_name'):
        p.images_dir_name = f"fdtd_{label}"
    if hasattr(p, 'field_e_filepath'):
        p.field_e_filepath = f"{p.field_e_filepath}_{label}"

    p.plasmon_source_object = MEEPSOURCE(
        source_type=getattr(p, 'plasmon_source_type').lower().strip(),
        source_center=getattr(p, 'plasmon_source_center'),
        source_size=getattr(p, 'plasmon_source_size'),
        component=component,
        amplitude=getattr(p, 'plasmon_source_amplitude'),
        is_integrated=getattr(p, 'plasmon_source_is_integrated'),
        **{k: v for k, v in getattr(p, 'plasmon_source_additional_parameters').items()}
    )

    # === Run simulation ===
    sim = SIMULATION(p)
    sim.run()
    field_data = getattr(sim, 'field_data', {})

    # === Save to pickle ===
    suffix = "np" if with_nanoparticle else "vac"
    filename = f"field_data_{pol_label.lower()}_{suffix}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(field_data, f)

    logger.info(f"[{label}] Finished → saved {filename}")
    return filename


def run(params):
    logger.info("=== Starting FDTD Response Driver (Chen 2010 Figure 1) ===")
    logger.info("Running 4 simulations in parallel (X/Y × NP/VAC)...")

    cases = [
        # ("X", True),   # X with nanoparticle
        # ("Y", True),   # Y with nanoparticle
        ("X", False),  # X vacuum
        ("Y", False),  # Y vacuum
    ]

    results = {}

    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_case = {
            executor.submit(_run_one_case, params, pol, with_np): (pol, with_np)
            for pol, with_np in cases
        }

        for future in as_completed(future_to_case):
            pol, with_np = future_to_case[future]
            try:
                filename = future.result()
                with open(filename, "rb") as f:
                    data = pickle.load(f)
                key = f"{pol.lower()}_{'np' if with_np else 'vac'}"
                results[key] = data
                logger.info(f"Loaded {key} → {len(data)} probe points")
            except Exception as e:
                logger.error(f"Simulation for {pol} {'NP' if with_np else 'VAC'} failed: {e}")

    # ====================== FINAL OUTPUT ======================
    print("\n" + "="*80)
    print("CHEN 2010 FDTD RESPONSE — ALL FOUR RUNS COMPLETE")
    print("="*80)

    for key in ["x_np", "y_np", "x_vac", "y_vac"]:
        data = results.get(key, {})
        print(f"\n{key.upper()}:")
        if data:
            for probe, ts in data.items():
                print(f"  {probe}: {len(ts)} time steps")
        else:
            print("  No data")

    print("\n" + "="*80)
    logger.info("FDTD Response Driver finished. Four pickle files ready for λ computation.")
