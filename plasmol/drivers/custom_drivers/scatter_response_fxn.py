# drivers/custom_drivers/scatter_response_fxn.py
# ===================================================
#         This driver is set to replicate           #
#      the Figure 1 from the Chen2010 paper         #
#        https://doi.org/10.1021/jp1043392          #
# ===================================================

import os
import copy
import logging
import pickle
import meep as mp
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

from plasmol.classical.simulation import SIMULATION
from plasmol.classical.sources import MEEPSOURCE
from plasmol.utils.logging import setup_logging

# ====================== CONFIG ======================
PICKLE_DIR = "."
OUTPUT_DIR = "output/."
FREQ_MIN_EV = 0.0
FREQ_MAX_EV = 6
# ===================================================

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
    logger.info(f"Saved dielectric slice: {filename}  (max ε = {max_eps:.3f})")
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
    logger.info("\n" + "="*80)
    logger.info("SCATTERING RESPONSE FUNCTION - ALL FOUR RUNS COMPLETE")
    logger.info("="*80)

    for key in ["x_np", "y_np", "x_vac", "y_vac"]:
        data = results.get(key, {})
        logger.info(f"\n{key.upper()}:")
        if data:
            for probe, ts in data.items():
                logger.info(f"  {probe}: {len(ts)} time steps")
        else:
            logger.info("  No data")

    logger.info("\n" + "="*80)
    logger.info("FDTD Response Driver finished. Four pickle files ready for λ computation.")

    x_np  = load_pickle("field_data_X_NP.pkl")
    y_np  = load_pickle("field_data_Y_NP.pkl")
    x_vac = load_pickle("field_data_X_VAC.pkl")
    y_vac = load_pickle("field_data_Y_VAC.pkl")

    # Time axis (all probes should have identical sampling)
    first_probe = list(x_np.keys())[0]
    t = np.array([row[0] for row in x_np[first_probe]])
    dt = t[1] - t[0] if len(t) > 1 else 0.019

    colors = ['red', 'green', 'blue', 'black', 'purple']

    # ====================== RAW DATA PLOTS ======================
    plot_raw_fields(x_np, y_np, x_vac, y_vac, t, colors)

    # ====================== Frequency axis ======================
    freq_au = np.fft.fftfreq(len(t), d=dt) * 2 * np.pi
    freq_ev = freq_au * 27.211386
    mask = (freq_ev >= FREQ_MIN_EV) & (freq_ev <= FREQ_MAX_EV)

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # ====================== COMBINED 2x2 PLOT  ======================
    logger.info("Computing and plotting combined λ_xx(ω) + λ_yy(ω) figure...")

    def probe_key(p):
        """Handle tuple (11,0,0) or string '(11, 0, 0)' or similar."""
        s = str(p).replace(" ", "").replace("'", "")
        try:
            # Extract first number (11, 12, ...)
            return int(s.split(',')[0].strip('()'))
        except:
            return 0

    probes = sorted(x_np.keys(), key=probe_key)

    fig, axs = plt.subplots(2, 2, figsize=(13, 6), sharex=True)

    # Left column: λ_xx
    for i, probe in enumerate(probes):
        if probe not in x_np or probe not in x_vac:
            continue
        e_total = np.array([row[1] for row in x_np[probe]])   # Ex
        e_vac   = np.array([row[1] for row in x_vac[probe]])

        E_total = np.fft.fft(e_total) * dt
        E_vac   = np.fft.fft(e_vac)   * dt
        lam = (E_total / E_vac) - 1.0
        lam = np.conj(lam)

        col = colors[i % len(colors)]
        lbl = str(probe)

        axs[0, 0].plot(freq_ev[mask], np.real(lam[mask]), color=col, lw=1.9, label=lbl)
        axs[1, 0].plot(freq_ev[mask], np.imag(lam[mask]), color=col, lw=1.9)

    # Right column: λ_yy
    for i, probe in enumerate(probes):
        if probe not in y_np or probe not in y_vac:
            continue
        e_total = np.array([row[2] for row in y_np[probe]])   # Ey
        e_vac   = np.array([row[2] for row in y_vac[probe]])

        E_total = np.fft.fft(e_total) * dt
        E_vac   = np.fft.fft(e_vac)   * dt
        lam = (E_total / E_vac) - 1.0
        lam = np.conj(lam)

        col = colors[i % len(colors)]
        lbl = str(probe)

        # === THIS IS THE IMPORTANT LINE ===
        axs[0, 1].plot(freq_ev[mask], np.real(lam[mask]),
                       color=col, lw=1.9, label=lbl)
        
        axs[1, 1].plot(freq_ev[mask], np.imag(lam[mask]),
                       color=col, lw=1.9)
        
    panel_labels = [
        [r"$\lambda_{xx,\mathrm{real}}$", r"$\lambda_{yy,\mathrm{real}}$"],
        [r"$\lambda_{xx,\mathrm{imaginary}}$", r"$\lambda_{yy,\mathrm{imaginary}}$"]
    ]
    for i in range(2):
        for j in range(2):
            ax = axs[i, j]
            ax.text(0.04, 0.93, panel_labels[i][j], transform=ax.transAxes,
                    fontsize=15, fontweight='bold', va='top', ha='left')

    axs[0, 1].legend(fontsize=9, loc="upper right")

    for ax in axs.flat:
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", ls="--", alpha=0.5)
        ax.set_xlim(FREQ_MIN_EV, FREQ_MAX_EV)

    axs[0, 0].set_ylim(-5, 9)
    axs[0, 1].set_ylim(-5, 9)
    axs[1, 0].set_ylim(-11, 13)
    axs[1, 1].set_ylim(-11, 13)

    for ax in axs[1, :]:
        ax.set_xlabel("Energy (eV)", fontsize=13)
    
    for ax in axs.flat:
        ax.tick_params(direction='in', which='major', length=6, width=1.2,
                       top=True, right=True, bottom=True, left=True)
        ax.minorticks_on()
        ax.tick_params(which='minor', direction='in', length=3, width=1.0,
                       top=True, right=True, bottom=True, left=True)
        ax.tick_params(labeltop=False, labelright=False)

    plt.tight_layout()
    out_path = Path(OUTPUT_DIR) / "lambda_chen2010_fig1.png"
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {out_path}")

    logger.info("\nAll plotting and λ computation complete.")



def load_pickle(name):
    path = Path(PICKLE_DIR) / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_raw_fields(x_np, y_np, x_vac, y_vac, t, colors):
    logger.info("Plotting raw time-domain data from the four pickle files...")
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    for pol, np_data, vac_data, comp_idx, comp_name in [
        ('x', x_np, x_vac, 1, 'Ex'),
        ('y', y_np, y_vac, 2, 'Ey')
    ]:
        fig, axs = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

        for i, (probe, data_total) in enumerate(np_data.items()):
            if probe not in vac_data:
                continue
            e_total = np.array([row[comp_idx] for row in data_total])
            e_vac   = np.array([row[comp_idx] for row in vac_data[probe]])

            axs[0].plot(t, e_total, color=colors[i], lw=1.9, label=probe)
            axs[1].plot(t, e_vac,   color=colors[i], lw=1.9, label=probe)

        axs[0].set_title(f'Time-domain {comp_name}(t) — 20 nm Ag Nanoparticle')
        axs[1].set_title(f'Time-domain {comp_name}(t) — Vacuum reference')
        for ax in axs:
            ax.set_xlabel('Time (a.u.)')
            ax.set_ylabel(f'{comp_name} (a.u.)')
            ax.grid(True, alpha=0.3)
            ax.legend(title="Probe distance", fontsize=9)

        plt.tight_layout()
        fig.savefig(Path(OUTPUT_DIR) / f"raw_{pol}_fields.png",
                    dpi=350, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved {OUTPUT_DIR}/raw_{pol}_fields.png")

    logger.info("Raw field data plotting complete.\n")

