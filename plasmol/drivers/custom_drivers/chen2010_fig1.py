# drivers/custom_drivers/chen2010_fig1.py
import copy
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import meep as mp

from plasmol.classical.simulation import SIMULATION
from plasmol.classical.sources import MEEPSOURCE
from plasmol.utils.logging import setup_logging

logger = logging.getLogger("main")

class PrefixFilter(logging.Filter):
    """Adds a prefix to every log record from that child process."""
    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    def filter(self, record):
        record.msg = f"[{self.prefix}] {record.msg}"
        return True
    

def _save_dielectric_slice(sim, pol_label: str, with_nanoparticle: bool):
    """Save a 2D XY slice of the dielectric constant to visually confirm the NP."""
    eps = sim.simulation.get_array(
        component=mp.Dielectric,
        center=mp.Vector3(),
        size=sim.simulation.cell_size
    )
    
    nz = eps.shape[2]
    slice_2d = eps[:, :, nz // 2]
    
    suffix = "np" if with_nanoparticle else "vac"
    filename = f"dielectric_slice_{pol_label.lower()}_{suffix}.png"
    
    plt.figure(figsize=(7, 7))
    plt.imshow(slice_2d.T, origin='lower', cmap='plasma', vmin=0.9, vmax=slice_2d.max() or 10)
    plt.colorbar(label='Dielectric constant ε')
    plt.title(f'Dielectric Slice (XY) — {pol_label.upper()}-pol {"NP" if with_nanoparticle else "VAC"}\n'
              f'max ε = {slice_2d.max():.3f}')
    plt.xlabel('X (grid points)')
    plt.ylabel('Y (grid points)')
    
    # Highlight the NP boundary with a contour (works even if ε ≈ 1)
    if with_nanoparticle:
        plt.contour(slice_2d.T > 1.01, levels=[0.5], colors='red', linewidths=2, linestyles='--', alpha=0.9)
        plt.text(0.05, 0.95, 'NP boundary (red dashed)', transform=plt.gca().transAxes,
                 color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    max_eps = eps.max()
    logger.info(f"✅ Saved dielectric slice: {filename}  (max ε = {max_eps:.3f})")
    return max_eps > 2.0

def _run_one_case(params, pol_label: str, with_nanoparticle: bool):
    """
    Run a single FDTD case (X or Y, with or without nanoparticle).
    All output from child processes now goes ONLY to the log file with a clear prefix.
    """
    label = f"{pol_label}-{'NP' if with_nanoparticle else 'VAC'}"

    # === FORCE logging redirection IMMEDIATELY ===
    log_file = getattr(params, 'log', None)
    setup_logging(verbose=1, log_file=log_file)

    # Add process-specific prefix to ALL log messages from this child
    prefix_filter = PrefixFilter(label)
    logging.getLogger().addFilter(prefix_filter)

    # Extra-strong redirect for any library that prints directly
    import sys
    from plasmol.utils.logging import PRINTLOGGER
    logger = logging.getLogger()
    sys.stdout = PRINTLOGGER(logger, logging.INFO)
    sys.stderr = PRINTLOGGER(logger, logging.WARNING)

    logger.info("Starting simulation in child process...")

    p = copy.deepcopy(params)

    # Recreate plasmon NP
    p.has_nanoparticle = with_nanoparticle
    if with_nanoparticle:
        mat_name = p.nanoparticle_dict["material"]
        p.nanoparticle_material = p._load_meep_material(mat_name)
        p.nanoparticle = mp.Sphere(
            radius=getattr(p, 'nanoparticle_radius'),
            center=mp.Vector3(*getattr(p, 'nanoparticle_center')),
            material=p.nanoparticle_material
        )
        logger.info(f"Nanoparticle geometry recreated")
        
    # Recreate plasmon source
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
        **{k: v for k, v in getattr(p, 'plasmon_source_additional_parameters', {}).items()}
    )

    # === Run simulation ===
    sim = SIMULATION(p)
    sim.run()

    # === Save dielectric slice AFTER the run ===
    _save_dielectric_slice(sim, pol_label, with_nanoparticle)

    field_data = getattr(sim, 'field_data', {})

    # === Save to pickle ===
    suffix = "np" if with_nanoparticle else "vac"
    filename = f"field_data_{pol_label.lower()}_{suffix}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(field_data, f)

    logger.info("Finished → saved %s", filename)
    return filename


def run(params):
    logger.info("=== Starting FDTD Response Driver (Chen 2010 Figure 1) ===")
    logger.info("Running 4 simulations in parallel (X/Y × NP/VAC)...")

    cases = [
        ("X", True),   # X with nanoparticle
        ("Y", True),   # Y with nanoparticle
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