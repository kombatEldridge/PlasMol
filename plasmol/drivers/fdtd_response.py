# drivers/fdtd_response.py
import copy
import logging
import matplotlib.pyplot as plt
import numpy as np

from plasmol.classical.simulation import SIMULATION
from classical.sources import MEEPSOURCE   # your custom source builder

logger = logging.getLogger("main")


def run(params):
    """
    Main entry point for fdtd_response driver (Chen 2010 Figure 1 replication).
    """
    logger.info("=== Starting FDTD Response Driver (Chen 2010 Figure 1) ===")

    # ====================== FIRST RUN (X-Polarization) ======================
    logger.info("Running simulation for X-polarization...")
    sim_x = SIMULATION(params)
    sim_x.run()

    field_data_x = getattr(sim_x, 'field_data', {})
    logger.info(f"X-polarization run complete. field_data keys: {list(field_data_x.keys())}")

    # ====================== SECOND RUN (Y-Polarization) ======================
    logger.info("Preparing Y-polarization run...")

    params_y = copy.copy(params)
    params_y.__dict__ = dict(params.__dict__)

    # Ensure probe_points are carried over
    params_y.probe_points = getattr(params, 'probe_points', None)
    params_y.has_additional_parameters = getattr(params, 'has_additional_parameters', True)

    params_y.plasmon_source_component = "y"
    if hasattr(params_y, 'images_dir_name'):
        params_y.images_dir_name = "fdtd_y_pol"

    # Rebuild source object for Y-polarization
    params_y.plasmon_source_object = MEEPSOURCE(
        source_type=params_y.plasmon_source_type.lower().strip(),
        source_center=params_y.plasmon_source_center,
        source_size=params_y.plasmon_source_size,
        component=params_y.plasmon_source_component.lower().strip(),
        amplitude=params_y.plasmon_source_amplitude,
        is_integrated=params_y.plasmon_source_is_integrated,
        **{k: v for k, v in params_y.plasmon_source_additional_parameters.items()}
    )

    logger.info("Running simulation for Y-polarization...")
    sim_y = SIMULATION(params_y)
    sim_y.run()

    field_data_y = getattr(sim_y, 'field_data', {})
    logger.info(f"Y-polarization run complete. field_data keys: {list(field_data_y.keys())}")

    # ====================== PRINT RESULTS ======================
    print("\n" + "="*70)
    print("FIELD DATA FROM X-POLARIZATION RUN")
    print("="*70)
    if field_data_x:
        for key, data in field_data_x.items():
            print(f"Probe {key}: {len(data)} time steps recorded")
            if data:
                print(f"  First point: t={data[0][0]:.4f} au, Ex={data[0][1]:.6e}")
    else:
        print("No field_data found in X run!")

    print("\n" + "="*70)
    print("FIELD DATA FROM Y-POLARIZATION RUN")
    print("="*70)
    if field_data_y:
        for key, data in field_data_y.items():
            print(f"Probe {key}: {len(data)} time steps recorded")
            if data:
                print(f"  First point: t={data[0][0]:.4f} au, Ey={data[0][2]:.6e}")
    else:
        print("No field_data found in Y run!")

    print("\n" + "="*70)
    logger.info("FDTD Response Driver finished (two runs complete).")

    # ====================== PLOT FIELD DATA ======================
    plot_probe_fields(field_data_x, "X", "fdtd_x_probes.png")
    plot_probe_fields(field_data_y, "Y", "fdtd_y_probes.png")


def plot_probe_fields(field_data, pol_label, filename):
    """Plot time-domain E field for all probe points."""
    if not field_data:
        logger.warning(f"No data to plot for {pol_label}-polarization")
        return

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(field_data)))

    for i, (key, data) in enumerate(field_data.items()):
        if not data:
            continue
        t = [row[0] for row in data]
        if pol_label == "X":
            e = [row[1] for row in data]   # Ex
            label = f"Ex at {key}"
        else:
            e = [row[2] for row in data]   # Ey
            label = f"Ey at {key}"

        plt.plot(t, e, label=label, color=colors[i], linewidth=1.5)

    plt.xlabel("Time (a.u.)")
    plt.ylabel(f"E{pol_label.lower()} (a.u.)")
    plt.title(f"Electric Field at Probe Points — {pol_label}-Polarization")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved: {filename}")