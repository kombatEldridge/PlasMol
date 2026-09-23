# drivers/custom_drivers/core_hole.py
import logging
import numpy as np
from plasmol.quantum.geometry import rotate_molecule_geometry
from plasmol.quantum.molecule import MOLECULE


def run(params):
    """Survey per-atom contributions for listed MOs, then exit (no propagation).

    Sudden SCH/DCH belongs under ``molecule.core_hole`` with driver
    ``quantum``, ``absorption``, or ``plasmol``.
    """
    logger = logging.getLogger("main")

    params.has_core_hole = False
    rotate_molecule_geometry(params)
    molecule = MOLECULE(params)
    mo_dict = getattr(params, 'mo_removal_index_dict', None) or {}
    for mo_idx in mo_dict.keys():
        nmo = np.asarray(molecule.mf.mo_coeff).shape[-1]
        if mo_idx >= nmo:
            raise ValueError(
                f"MO index {mo_idx} is out of range (molecule has {nmo} MOs, 0-based)."
            )
        _mo_atom_contribution(molecule, mo_idx)
    logger.info("Core-hole MO contribution survey complete; exiting before propagation.")


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
