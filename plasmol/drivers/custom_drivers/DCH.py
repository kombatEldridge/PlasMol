# drivers/custom_drivers/DCH.py
import logging
from plasmol.quantum.molecule import MOLECULE
from plasmol.drivers.quantum import run as run_quantum

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
            mo_atom_contribution(molecule, mo_idx)
        logger.info("DCH MO contribution survey complete; exiting before propagation.")
        return
    
    run_quantum(params)


def mo_atom_contribution(molecule, mo_idx, threshold=0.01):
    """Print which atoms contribute most to a specific MO"""
    c = molecule.mf.mo_coeff[:, mo_idx]
    pop_ao = c * (molecule.S @ c)
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
