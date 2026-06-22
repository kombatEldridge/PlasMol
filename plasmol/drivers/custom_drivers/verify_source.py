# drivers/custom_drivers/verify_source.py
"""
Verify Source Driver

A minimal driver for testing and visualizing custom plasmon sources in an
empty FDTD (Meep) simulation. No molecule/quantum component is used.

The driver will:
- Run a pure classical simulation.
- Record E-field at the source center over time.
- Save a plot "verify_source_Ez.png" (and other components if recorded).
"""
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import meep as mp

from plasmol.classical.simulation import SIMULATION

logger = logging.getLogger("main")

def run(params):
    logger.info(" === RUNNING EMPTY FDTD SOURCE VERIFICATION === ")

    # Force a pure classical (empty) simulation - no molecule at all
    params.has_molecule = False
    params.has_molecule_source = False
    params.has_plasmon = True

    # Ensure we have a probe at the source center for field recording
    source_center = getattr(params, 'plasmon_source_center')
    params.probe_points = [source_center]
    logger.debug(f"Using source center {source_center} as default probe.")

    # Initialize and run the pure classical simulation
    sim = SIMULATION(params)
    sim.run()

    # After the run, generate verification plots from recorded probe data
    _plot_verification_fields(sim, source_center)

    logger.info("Empty FDTD source verification complete.")

def _plot_verification_fields(sim, source_center):
    """Plot recorded E field time series at the probe point(s)."""
    if not hasattr(sim, 'field_data') or not sim.field_data:
        logger.warning("No probe field data was recorded. Cannot generate verification plots.")
        return

    for point_key, data in sim.field_data.items():
        if not data:
            continue

        times = [entry[0] for entry in data]
        ex = [entry[1] for entry in data]
        ey = [entry[2] for entry in data]
        ez = [entry[3] for entry in data]

        plt.figure(figsize=(8, 5))
        plt.plot(times, ex, label='Ex', alpha=0.8)
        plt.plot(times, ey, label='Ey', alpha=0.8)
        plt.plot(times, ez, label='Ez', alpha=0.8)
        plt.xlabel('Time (a.u.)')
        plt.ylabel('Electric Field (a.u.)')
        plt.title(f'Electric Field Time Series at {point_key}\n(Source verification)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        filename = "verify_source.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved verification plot: {filename}")

        # Also print some stats for the dominant component
        max_e = max(max(abs(np.array(ex))), max(abs(np.array(ey))), max(abs(np.array(ez))))
        logger.info(f"  Max |E| at probe {point_key}: {max_e:.6e} a.u.")
