# drivers/fdtd_response.py
"""
Fixed version of fdtd_response.py

Key fixes:
- params now safely picklable (via __getstate__ added to PARAMS class)
- _run_and_save ALWAYS recreates plasmon_source_object (and nanoparticle if present)
  so we never rely on the original Swig objects being present after unpickling.
- Works for both single-node (ProcessPoolExecutor) and MPI paths.
"""

import copy
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ====================== MPI SUPPORT ======================
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    has_mpi = True
except (ImportError, ModuleNotFoundError):
    rank = 0
    nprocs = 1
    comm = None
    has_mpi = False

from plasmol.classical.simulation import SIMULATION
from plasmol.classical.sources import MEEPSOURCE   # fixed import

logger = logging.getLogger("main")


def _run_and_save(params, pol_label):
    """Run one polarization and save ONLY a filename. Nothing Meep-related is returned."""
    logger.info(f"[{pol_label}] Starting simulation in child process...")

    p = copy.deepcopy(params)          # now safe because __getstate__ stripped Swig objects

    # === ALWAYS recreate Meep objects (source + nanoparticle) ===
    # This is the key change: we no longer depend on the pickled version having them.

    # 1. Recreate / override plasmon source
    if getattr(p, 'has_plasmon_source', False):
        component = "y" if pol_label == "Y" else p.plasmon_source_component
        if pol_label == "Y":
            p.plasmon_source_component = "y"
            if hasattr(p, 'images_dir_name'):
                p.images_dir_name = "fdtd_y_pol"

        p.plasmon_source_object = MEEPSOURCE(
            source_type=p.plasmon_source_type.lower().strip(),
            source_center=p.plasmon_source_center,
            source_size=p.plasmon_source_size,
            component=component,
            amplitude=p.plasmon_source_amplitude,
            is_integrated=p.plasmon_source_is_integrated,
            **{k: v for k, v in getattr(p, 'plasmon_source_additional_parameters', {}).items()}
        )

    # 2. Recreate nanoparticle if the section was present (same for X and Y)
    if getattr(p, 'has_nanoparticle', False) and not hasattr(p, 'nanoparticle'):
        # Recreate exactly as PARAMS._attribute_formation does
        try:
            import importlib
            materials = importlib.import_module("meep.materials")
            mat = getattr(materials, p.nanoparticle_material)
            p.nanoparticle = mp.Sphere(
                radius=p.nanoparticle_radius,
                center=mp.Vector3(*p.nanoparticle_center),
                material=mat
            )
        except Exception as e:
            logger.error(f"Failed to recreate nanoparticle in child process: {e}")
            raise

    sim = SIMULATION(p)
    sim.run()
    field_data = getattr(sim, 'field_data', {})

    filename = f"field_data_{pol_label.lower()}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(field_data, f)

    logger.info(f"[{pol_label}] Finished → saved {filename}")
    return filename                     # <-- ONLY a string, no Swig objects


def run(params):
    if rank == 0:
        logger.info("=== Starting FDTD Response Driver (Chen 2010 Figure 1) ===")
        if has_mpi and nprocs > 1:
            logger.info(f"MPI mode – {nprocs} processes → X/Y run in parallel on separate groups")
        else:
            logger.info("Single-node mode (Mac) – X/Y run in parallel using separate processes")

    field_data_x = {}
    field_data_y = {}

    if has_mpi and nprocs >= 2:
        # MPI path – also benefits from the new _run_and_save logic
        color = rank % 2
        local_comm = comm.Split(color=color, key=rank)
        local_rank = local_comm.Get_rank()

        if color == 0:
            if local_rank == 0:
                logger.info("Group X: running X-polarization...")
            sim_x = SIMULATION(params)
            sim_x.run()
            field_data_x = getattr(sim_x, 'field_data', {})
            if local_rank == 0:
                logger.info("Group X complete.")
        else:
            if local_rank == 0:
                logger.info("Group Y: running Y-polarization...")
            params_y = copy.deepcopy(params)   # safe thanks to __getstate__
            params_y.plasmon_source_component = "y"
            if hasattr(params_y, 'images_dir_name'):
                params_y.images_dir_name = "fdtd_y_pol"
            params_y.plasmon_source_object = MEEPSOURCE(
                source_type=params_y.plasmon_source_type.lower().strip(),
                source_center=params_y.plasmon_source_center,
                source_size=params_y.plasmon_source_size,
                component="y",
                amplitude=params_y.plasmon_source_amplitude,
                is_integrated=params_y.plasmon_source_is_integrated,
                **{k: v for k, v in getattr(params_y, 'plasmon_source_additional_parameters', {}).items()}
            )
            sim_y = SIMULATION(params_y)
            sim_y.run()
            field_data_y = getattr(sim_y, 'field_data', {})
            if local_rank == 0:
                logger.info("Group Y complete.")

        if local_rank == 0:
            if color == 0:
                comm.send(field_data_x, dest=0, tag=100)
            else:
                comm.send(field_data_y, dest=0, tag=101)
        if rank == 0:
            field_data_x = comm.recv(source=MPI.ANY_SOURCE, tag=100)
            field_data_y = comm.recv(source=MPI.ANY_SOURCE, tag=101)

    else:
        # Mac / single-node multiprocessing – now safe
        with ProcessPoolExecutor(max_workers=2) as executor:
            future_x = executor.submit(_run_and_save, params, "X")
            future_y = executor.submit(_run_and_save, params, "Y")

            for future in as_completed([future_x, future_y]):
                filename = future.result()          # <-- only a string
                with open(filename, "rb") as f:
                    data = pickle.load(f)
                if "x" in filename:
                    field_data_x = data
                else:
                    field_data_y = data

    # ====================== FINAL OUTPUT ======================
    if rank == 0:
        print_results(field_data_x, field_data_y)
        plot_probe_fields(field_data_x, "X", "fdtd_x_probes.png")
        plot_probe_fields(field_data_y, "Y", "fdtd_y_probes.png")
        logger.info("FDTD Response Driver finished (both polarizations complete).")


def print_results(field_data_x, field_data_y):
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


def plot_probe_fields(field_data, pol_label, filename):
    if not field_data:
        logger.warning(f"No data to plot for {pol_label}-polarization")
        return

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(field_data)))

    for i, (key, data) in enumerate(field_data.items()):
        if not data:
            continue
        t = [row[0] for row in data]
        e_idx = 1 if pol_label == "X" else 2
        e = [row[e_idx] for row in data]
        label = f"E{pol_label.lower()} at {key}"

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