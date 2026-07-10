# drivers/custom_drivers/DCH.py
import os
import logging
from plasmol.quantum.molecule import MOLECULE
from plasmol.drivers.quantum import run as run_quantum
from plasmol.utils.checkpoint import add_dch_mo_occ_checkpoint
from plasmol.utils.csv import init_csv
from plasmol.utils.plotting import plot_dch_mo_occupations


def run(params):
    logger = logging.getLogger("main")

    # Survey mode: build the molecule, report MO atom contributions, then exit.
    if params.check_mo_contrib_by_atom:
        params.has_dch = False
        molecule = MOLECULE(params)
        for mo_idx in params.mo_index_list:
            nmo = molecule.mf.mo_coeff.shape[1]
            if mo_idx >= nmo:
                raise ValueError(f"MO index {mo_idx} is out of range (molecule has {nmo} MOs, 0-based).")
            _mo_atom_contribution(molecule, mo_idx)
        logger.info("DCH MO contribution survey complete; exiting before propagation.")
        return

    if not params.resumed_from_checkpoint:
        header = ['Timestamps (au)']
        for inx in params.dch_watch_indices:
            header.append(f'MO index {inx}')
        init_csv(
            params.dch_mo_occ_filepath,
            f"Time dependent MO occupations for the following MO indices: {params.dch_watch_indices}",
            header=header,
        )
        logger.debug(f"DCH MO occupation file initialized: {params.dch_mo_occ_filepath}")
    else:
        if params.dch_mo_occ_filepath and os.path.exists(params.dch_mo_occ_filepath):
            logger.debug(f"Resuming DCH MO occupation tracking from existing {params.dch_mo_occ_filepath}")
        else:
            header = ['Timestamps (au)']
            for inx in params.dch_watch_indices:
                header.append(f'MO index {inx}')
            init_csv(
                params.dch_mo_occ_filepath,
                f"Time dependent MO occupations for the following MO indices: {params.dch_watch_indices}",
                header=header,
            )
            logger.warning("Resumed DCH run but no mo_occ CSV was restored from the checkpoint; started a fresh occupation file at current time.")

    if getattr(params, 'has_checkpoint', False):
        add_dch_mo_occ_checkpoint(params, params.dch_mo_occ_filepath)

    try:
        run_quantum(params)
    except Exception as e:
        logger.error(f"Error occurred while running quantum simulation: {e}")
        raise
    finally:
        base, _ = os.path.splitext(params.dch_mo_occ_filepath)
        plot_dch_mo_occupations(params.dch_mo_occ_filepath, output_image_path=base)



def _mo_atom_contribution(molecule, mo_idx, threshold=0.01):
    """Print which atoms contribute most to a specific MO"""
    c = molecule.mf.mo_coeff[:, mo_idx]
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
