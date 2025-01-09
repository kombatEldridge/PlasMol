import input_parser
from pyscf import gto
import wavefunction
import numpy as np
import os
from rk4 import rk4_ind_dipole as ind_dipole

def run(inputfile, 
        dt,
        directionCalculationSim,
        directionCalculationBohr,
        Ex=None, 
        Ey=None, 
        Ez=None):
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

    rks_wfn = wavefunction.RKS(pyscf_mol)
    rks_energy = rks_wfn.compute(options)

    output = np.zeros((3, 3))
    mapDirToDig = {'x': 0, 'y': 1, 'z': 2}
    eArr = [Ex, Ey, Ez]

    if method["propagator"] == 'rk4':
        from rk4 import rk4_ind_dipole as ind_dipole
    elif method["propagator"] == 'magnus':
        from magnus import magnus_ind_dipole as ind_dipole
    elif method["propagator"] == 'ptg':
        from ptg import ptg_ind_dipole as ind_dipole

    for simDir in directionCalculationSim:
        for bohrDir in directionCalculationBohr:
            mu = ind_dipole(mapDirToDig[simDir], mapDirToDig[bohrDir], rks_wfn, eArr[mapDirToDig[simDir]], dt)
            if abs(mu) >= method["resplimit"]:
                output[mapDirToDig[simDir], mapDirToDig[bohrDir]] = float(mu)

    filtered_output = np.sum(output, axis=1).tolist()
    return filtered_output