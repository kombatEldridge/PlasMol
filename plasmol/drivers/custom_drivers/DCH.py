# drivers/custom_drivers/DCH.py
import os
import logging
import numpy as np
from plasmol.quantum.molecule import MOLECULE
from plasmol.drivers.quantum import run as run_quantum
from plasmol.utils.checkpoint import add_dch_mo_occ_checkpoint
from plasmol.utils.plotting import plot_dch_mo_occupations


def run(params):
    logger = logging.getLogger("main")

    # Survey mode: build the molecule, report MO atom contributions, then exit.
    if params.check_mo_contrib_by_atom:
        params.has_dch = False
        molecule = MOLECULE(params)
        for mo_idx in params.mo_removal_index_dict.keys():
            nmo = np.asarray(molecule.mf.mo_coeff).shape[-1]
            if mo_idx >= nmo:
                raise ValueError(f"MO index {mo_idx} is out of range (molecule has {nmo} MOs, 0-based).")
            _mo_atom_contribution(molecule, mo_idx)
        logger.info("DCH MO contribution survey complete; exiting before propagation.")
        return

    # MO occupation CSV is initialized in MOLECULE after neutral SCF once
    # dch_log_indices (0 .. LUMO+1) are known. dch_watch_indices only selects
    # which logged series appear in the final plot.

    if getattr(params, 'has_checkpoint', False):
        add_dch_mo_occ_checkpoint(params, params.dch_mo_occ_filepath)

    try:
        run_quantum(params)
    except Exception as e:
        logger.error(f"Error occurred while running quantum simulation: {e}")
        raise
    finally:
        base, _ = os.path.splitext(params.dch_mo_occ_filepath)
        plot_indices = getattr(params, 'dch_watch_indices', None)
        if plot_indices is not None:
            logger.debug(f"Plotting DCH hole occupations for MO indices: {plot_indices}")
        else:
            logger.info("Plotting DCH hole occupations for all logged MO indices.")
        plot_dch_mo_occupations(
            params.dch_mo_occ_filepath,
            output_image_path=base,
            indices=plot_indices,
            filter_by_amplitude=getattr(params, 'dch_filter_by_amplitude', False),
            amplitude_threshold=getattr(params, 'dch_amplitude_threshold', 0.2),
        )



def _mo_atom_contribution(molecule, mo_idx, threshold=0.01):
    """Print which atoms contribute most to a specific MO"""
    C = molecule.mf.mo_coeff
    if getattr(molecule, 'is_open_shell', False) and np.asarray(C).ndim == 3:
        c = C[0][:, mo_idx]
    else:
        c = C[:, mo_idx]
    pop_ao = c * (molecule.S @ c)
    logger = logging.getLogger("main")
    logger.info(f"=== MO {mo_idx+1} (index {mo_idx}) contributions ===")
    logger.info("Atom          Contribution (%)")

    ao_labels = molecule.mol.ao_labels()
    atom_pop = {}
    for i, label in enumerate(ao_labels):
        if pop_ao[i] > threshold:
            atom = label.split()[1]
            atom_pop[atom] = atom_pop.get(atom, 0) + pop_ao[i]

    for atom in sorted(atom_pop, key=atom_pop.get, reverse=True):
        percent = atom_pop[atom] * 100
        logger.info(f"{atom:4s}          {percent:6.2f}%")
