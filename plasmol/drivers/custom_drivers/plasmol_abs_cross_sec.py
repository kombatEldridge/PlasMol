# plasmol/drivers/plasmol_abs_cross_sec.py
import copy
import logging
import os
import pickle
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from pathlib import Path

import matplotlib.pyplot as plt
import meep as mp
import numpy as np

from plasmol.classical.simulation import SIMULATION
from plasmol.classical.sources import MEEPSOURCE
from plasmol.utils.logging import setup_logging
from plasmol.quantum.molecule import MOLECULE
from plasmol.utils.csv import init_csv

logger = logging.getLogger("main")


class PrefixFilter(logging.Filter):
    """Adds a prefix to every log record from that child process."""
    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    def filter(self, record):
        record.msg = f"[{self.prefix}] {record.msg}"
        return True

def distance(pos: list):
    return float(np.linalg.norm([pos[0], pos[1], pos[2]]))

def _run_one_case(params, case: str, incident_flux_data=None):
    label = case.upper()
    log_file = getattr(params, 'log', None)
    setup_logging(verbose=1, log_file=log_file)
    prefix_filter = PrefixFilter(label)
    root_logger = logging.getLogger()
    root_logger.addFilter(prefix_filter)

    import sys
    from plasmol.utils.logging import PRINTLOGGER
    logger = logging.getLogger()
    sys.stdout = PRINTLOGGER(logger, logging.INFO)
    sys.stderr = PRINTLOGGER(logger, logging.WARNING)

    logger.info(f"Starting {label} simulation...")
    
    nfrq = 50
    p = copy.deepcopy(params)

    # Recreate plasmon NP
    r = distance(p.plasmol_molecule_position) + 0.005 # adding padding just in case
    if not case == "empty":
        mat_name = p.nanoparticle_dict["material"]
        p.nanoparticle_material = p._load_meep_material(mat_name)
        p.nanoparticle = mp.Sphere(
            radius=getattr(p, 'nanoparticle_radius'),
            center=mp.Vector3(*getattr(p, 'nanoparticle_center')),
            material=p.nanoparticle_material
        )
        logger.info(f"Nanoparticle geometry recreated")
        setattr(p, 'molecule', MOLECULE(p))
        # just to prevent any racetime things
        p.field_e_filepath = f"{Path(p.field_e_filepath).with_suffix('')}_{case}.csv"
        p.field_p_filepath = f"{Path(p.field_p_filepath).with_suffix('')}_{case}.csv"
        init_csv(p.field_e_filepath, "Electric Field intensity in atomic units")
        init_csv(p.field_p_filepath, "Molecule's Polarizability Field intensity in atomic units")
        logger.debug(f"Field files successfully initialized: {p.field_e_filepath} and {p.field_p_filepath}")
        logger.info(f"Molecule created")
    else:
        p.has_molecule = False
    
    # Recreate plasmon source
    freq = getattr(p, 'plasmon_source_additional_parameters').get('frequency')
    fwidth = getattr(p, 'plasmon_source_additional_parameters').get('fwidth')
    
    p.plasmon_source_object = MEEPSOURCE(
        source_type=getattr(p, 'plasmon_source_type').lower().strip(),
        source_center=getattr(p, 'plasmon_source_center'),
        source_size=getattr(p, 'plasmon_source_size'),
        component=getattr(p, 'plasmon_source_component'),
        is_integrated=getattr(p, 'plasmon_source_is_integrated'),
        **{k: v for k, v in getattr(p, 'plasmon_source_additional_parameters', {}).items()}
    )

    sim = SIMULATION(p)

    # Define the six flux boxes
    boxes = {}
    for name, center, size in [
        ("x1", mp.Vector3(x=-r), mp.Vector3(0, 2*r, 2*r)),
        ("x2", mp.Vector3(x=+r), mp.Vector3(0, 2*r, 2*r)),
        ("y1", mp.Vector3(y=-r), mp.Vector3(2*r, 0, 2*r)),
        ("y2", mp.Vector3(y=+r), mp.Vector3(2*r, 0, 2*r)),
        ("z1", mp.Vector3(z=-r), mp.Vector3(2*r, 2*r, 0)),
        ("z2", mp.Vector3(z=+r), mp.Vector3(2*r, 2*r, 0)),
    ]:
        boxes[name] = sim.simulation.add_flux(freq, fwidth, nfrq, mp.FluxRegion(center=center, size=size))

    # --- Case-specific logic ---
    if case == "empty":
        incident_flux_data = {}
        sim.run()
        freqs = mp.get_flux_freqs(boxes["x1"])
        flux0 = mp.get_fluxes(boxes["x1"])
        for name, flux_region in boxes.items():
            incident_flux_data[name] = sim.simulation.get_flux_data(flux_region)
        logger.info("Empty run finished")
        root_logger.removeFilter(prefix_filter)   # ← crucial
        logging.getLogger().removeFilter(prefix_filter)  # also remove from "main" logger if you added it there
        return {"incident_flux_data": incident_flux_data, "freqs": freqs, "flux0": flux0}
    elif case == "scatt":
        for k, b in boxes.items():
            sim.simulation.load_minus_flux_data(b, incident_flux_data[k])
        sim.run()
        logger.info("Scattering simulation completed.")
        scatt_fluxes = {}
        for name, _ in boxes.items():
            scatt_fluxes[name] = mp.get_fluxes(boxes[name])
        scatt_flux = (
            np.asarray(scatt_fluxes["x1"])
            - np.asarray(scatt_fluxes["x2"])
            + np.asarray(scatt_fluxes["y1"])
            - np.asarray(scatt_fluxes["y2"])
            + np.asarray(scatt_fluxes["z1"])
            - np.asarray(scatt_fluxes["z2"])
        )
        logger.info("Scattering run finished")
        return scatt_flux
    elif case == "abs":
        sim.run()
        logger.info("Absorption simulation completed.")
        abs_fluxes = {}
        for name, _ in boxes.items():
            abs_fluxes[name] = mp.get_fluxes(boxes[name])
        abs_flux = (
            np.asarray(abs_fluxes["x1"])
            - np.asarray(abs_fluxes["x2"])
            + np.asarray(abs_fluxes["y1"])
            - np.asarray(abs_fluxes["y2"])
            + np.asarray(abs_fluxes["z1"])
            - np.asarray(abs_fluxes["z2"])
        )
        logger.info("Absorption run finished")
        return abs_flux



def run(params):
    logger.info("=== Starting Cross-Section Driver ===")

    results = {}

    # 1. Run empty simulation
    logger.info("→ Running EMPTY simulation (incident flux)...")
    empty_result = _run_one_case(params, "empty")
    incident_flux_data = empty_result["incident_flux_data"]

    # 2. Run scattering + absorption in parallel
    with ProcessPoolExecutor(max_workers=2) as executor:
        future_scatt = executor.submit(_run_one_case, params, "scatt", incident_flux_data)
        future_abs   = executor.submit(_run_one_case, params, "abs", incident_flux_data)

        for future in as_completed([future_scatt, future_abs]):
            case = "scatt" if future == future_scatt else "abs"
            try:
                results[case] = future.result()
                logger.info(f"{case.upper()} run completed")
            except Exception as e:
                logger.error(f"{case.upper()} run failed: {e}")

    intensity = np.asarray(empty_result["flux0"])/(2 * getattr(params, 'nanoparticle_radius'))**2
    scatt_cross_section = np.divide(results["scatt"], intensity)
    np_abs_cross_section = np.divide(results["abs"], intensity)
    scatt_eff = scatt_cross_section*(-1)/(np.pi * getattr(params, 'nanoparticle_radius')**2)
    abs_eff = np_abs_cross_section/(np.pi*getattr(params, 'nanoparticle_radius')**2)

    wavelengths = 1/np.asarray(empty_result["freqs"])
    combined_array = np.column_stack((wavelengths, abs_eff, scatt_eff))
    output_filename = 'output_arrays.txt'
    np.savetxt(output_filename, combined_array, delimiter='\t', header='Wavelengths\tAbs\tScatt', comments='')

    data = pd.read_csv(output_filename, delimiter='\t')
    wavelengths = data["Wavelengths"] * 1000
    abs_eff = data["Abs"]
    scatt_eff = data["Scatt"]
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, abs_eff, label='Absorption Efficiency')
    plt.plot(wavelengths, scatt_eff, label='Scattering Efficiency')
    plt.plot(wavelengths, abs_eff + scatt_eff, label='Total Efficiency')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Efficiency')
    plt.title('Absorption and Scattering Efficiencies')
    plt.legend()
    plt.grid(True)
    plt.savefig('abs_scatt_efficiencies.png')
    plt.close()

    logger.info(f"Arrays saved to {output_filename}")
    logger.info("Absorption cross section driver finished.")
