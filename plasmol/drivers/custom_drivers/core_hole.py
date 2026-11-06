# drivers/custom_drivers/core_hole.py
import logging
from pathlib import Path

import numpy as np
from plasmol.quantum.geometry import rotate_molecule_geometry
from plasmol.quantum.molecule import MOLECULE

SURVEY_FILE = "mo_survey.txt"


def run(params):
    """Survey per-atom, per-shell contributions for listed MOs, then exit.

    Sudden SCH/DCH belongs under ``molecule.core_hole`` with driver
    ``quantum``, ``absorption``, or ``plasmol``. There is no propagation.
    """
    logger = logging.getLogger("main")

    params.has_core_hole = False
    rotate_molecule_geometry(params)
    molecule = MOLECULE(params)
    mo_dict = getattr(params, 'mo_removal_index_dict', None) or {}
    rows = ["mo_index\tatom_index\telement\tshell\tcontribution_percent"]
    for mo_idx in mo_dict.keys():
        nmo = np.asarray(molecule.mf.mo_coeff).shape[-1]
        if mo_idx >= nmo:
            raise ValueError(
                f"MO index {mo_idx} is out of range (molecule has {nmo} MOs, 0-based)."
            )
        rows.extend(_mo_atom_contribution(molecule, mo_idx))
    path = Path(SURVEY_FILE)
    path.write_text("\n".join(rows) + "\n")
    logger.info(f"Wrote {path.resolve()}")
    logger.info("Core-hole MO contribution survey complete; exiting before propagation.")


def _mo_atom_contribution(molecule, mo_idx, threshold=0.01):
    """Mulliken weight of each PySCF shell on each atom for one MO.

    ``ao_labels(fmt=False)`` is ``(atom_id, symbol, nl, component)``.
    Angular components of one ``nl`` (``2px``/``2py``/``2pz``, or spherical
    m) are summed. Shells with weight at or below ``threshold`` are omitted.
    """
    C = molecule.mf.mo_coeff
    if getattr(molecule, 'is_open_shell', False) and np.asarray(C).ndim == 3:
        c = C[0][:, mo_idx]
    else:
        c = C[:, mo_idx]
    pop_ao = np.asarray(c * (molecule.S @ c), dtype=float)
    labels = molecule.mol.ao_labels(fmt=False)
    if len(labels) != pop_ao.shape[0]:
        raise ValueError(
            f"AO label count ({len(labels)}) does not match the MO coefficient "
            f"length ({pop_ao.shape[0]})."
        )

    shell_pop = {}
    for pop, label in zip(pop_ao, labels):
        atom_id, symbol, shell, _component = label
        key = (int(atom_id), str(symbol), str(shell))
        shell_pop[key] = shell_pop.get(key, 0.0) + float(pop)

    logger = logging.getLogger("main")
    logger.info(f"=== MO {mo_idx+1} (index {mo_idx}) contributions ===")
    logger.info(f"{'Index':<8}{'Atom':<8}{'Shell':<8}Contribution (%)")

    ranked = sorted(
        ((key, pop) for key, pop in shell_pop.items() if pop > threshold),
        key=lambda item: (-item[1], item[0]),
    )
    rows = []
    for (atom_id, symbol, shell), pop in ranked:
        percent = pop * 100
        logger.info(f"{atom_id:<8}{symbol:<8}{shell:<8}{percent:6.2f}%")
        rows.append(f"{mo_idx}\t{atom_id}\t{symbol}\t{shell}\t{percent:.2f}")
    return rows
