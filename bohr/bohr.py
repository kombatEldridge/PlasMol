import input_parser
from pyscf import gto
import wavefunction
import numpy as np
import os
from mu import calculate_ind_dipole
import logging
import matrix_handler as mh
import sys
from volume import get_volume

# TODO: should probably move this to code that runs once only on init
def run(inputfile, dt, eArr):
    import options
    options = options.OPTIONS()

    molecule, method, basis = input_parser.read_input(inputfile,options)
    options.molecule = molecule

    #Format molecule string as required by PySCF
    atoms = molecule["atoms"]
    pyscf_molecule = "" 
    for index, atom in enumerate(atoms):
        pyscf_molecule += " " + atom
        pyscf_molecule += " " + str(molecule["coords"][atom+str(index+1)][0])
        pyscf_molecule += " " + str(molecule["coords"][atom+str(index+1)][1])
        pyscf_molecule += " " + str(molecule["coords"][atom+str(index+1)][2])
        if index != (len(atoms)-1):
            pyscf_molecule += ";"

    # Added B3LYP_WITH_VWN5 = True
    # to /Users/bldrdge1/.conda/envs/meep/lib/python3.11/site-packages/pyscf/__config__.py
    pyscf_mol = gto.M(atom = pyscf_molecule, 
                      basis  = basis["name"], 
                      unit   = 'B', 
                      charge = int(options.charge), 
                      spin   = int(options.spin), 
                      cart   = options.cartesian)
    pyscf_mol.set_common_origin(molecule["com"])
    pyscf_mol.verbose = 0
    pyscf_mol.max_memory = options.memory
    pyscf_mol.build()

    wfn = wavefunction.RKS(pyscf_mol)
    rks_energy = wfn.compute(options)
    
    D_ao_0 = wfn.D[0]
    D_mo_0 = wfn.C[0].T @ wfn.S @ D_ao_0 @ wfn.S @ wfn.C[0]
    mh.set_D_mo_0(D_mo_0)

    F_ao_0 = wfn.F[0]
    F_mo_0 = wfn.S @ wfn.C[0] @ F_ao_0 @ wfn.C[0].T @ wfn.S 
    mh.set_F_mo_0(F_mo_0)

    if method["propagator"] == 'rk4':
        from rk4 import propagate_density_matrix
    elif method["propagator"] == 'magnus':
        from magnus_2nd import propagate_density_matrix

    volume = get_volume(molecule["coords"])

    induced_dipole_matrix = calculate_ind_dipole(propagate_density_matrix, dt, eArr, wfn) / volume
        
    # Should be [p_x, p_y, p_z] where p is the dipole moment
    return induced_dipole_matrix 