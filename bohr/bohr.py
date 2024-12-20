import input_parser
from pyscf import gto
import wavefunction
import numpy as np
import os

def JK(wfn, D):
    pot = wfn.jk.get_veff(wfn.ints_factory, 2*D)
    Fa = wfn.T + wfn.Vne + pot
    return Fa

def ind_dipole(direction1, direction2, wfn, eField, dt, storage_file='D_ao.npy'):
    # Load D_ao from the storage file if it exists; otherwise initialize
    if os.path.exists(storage_file):
        D_ao = np.load(storage_file)
    else:
        D_ao = wfn.D[0]

    D_ao = wfn.D[0]
    D_ao_init = wfn.D[0]
    D_mo = wfn.C[0].T @ wfn.S.T @ D_ao @ wfn.S @ wfn.C[0]

    # RK4 method to evolve the density matrix over a time step dt
    # Step 1: Calculate the Fock matrix in the AO basis at the initial time
    Ft_ao = JK(wfn, D_ao) - wfn.mu[direction1] * eField[0]
    Ft_mo = wfn.C[0].T @ Ft_ao @ wfn.C[0]
    k1 = (-1j * (Ft_mo @ D_mo - D_mo @ Ft_mo))

    # Step 2: Estimate the density matrix at half the time step using k1
    temp_D_mo = D_mo + 0.5 * k1 * dt
    temp_D_ao = wfn.C[0] @ temp_D_mo @ wfn.C[0].T
    Ft_ao = JK(wfn, temp_D_ao) - wfn.mu[direction1] * eField[1]
    Ft_mo = wfn.C[0].T @ Ft_ao @ wfn.C[0]
    k2 = (-1j * (Ft_mo @ temp_D_mo - temp_D_mo @ Ft_mo))

    # Step 3: Estimate the density matrix again at half the time step using k2
    temp_D_mo = D_mo + 0.5 * k2 * dt
    temp_D_ao = wfn.C[0] @ temp_D_mo @ wfn.C[0].T
    Ft_ao = JK(wfn, temp_D_ao) - wfn.mu[direction1] * eField[1]
    Ft_mo = wfn.C[0].T @ Ft_ao @ wfn.C[0]
    k3 = (-1j * (Ft_mo @ temp_D_mo - temp_D_mo @ Ft_mo))

    # Step 4: Estimate the density matrix at the full time step using k3
    temp_D_mo = D_mo + k3 * dt
    temp_D_ao = wfn.C[0] @ temp_D_mo @ wfn.C[0].T
    Ft_ao = JK(wfn, temp_D_ao) - wfn.mu[direction1] * eField[2]
    Ft_mo = wfn.C[0].T @ Ft_ao @ wfn.C[0]
    k4 = (-1j * (Ft_mo @ temp_D_mo - temp_D_mo @ Ft_mo))

    # Combine k1, k2, k3, and k4 to get the final evolved density matrix
    D_mo = D_mo + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    D_ao = wfn.C[0] @ D_mo @ wfn.C[0].T

    np.save(storage_file, D_ao)

    # Calculate the induced dipole moment in the direction2
    mu = np.trace(wfn.mu[direction2] @ D_ao) - np.trace(wfn.mu[direction2] @ D_ao_init)
    return mu.real

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

    for simDir in directionCalculationSim:
        for bohrDir in directionCalculationBohr:
            mu = ind_dipole(mapDirToDig[simDir], mapDirToDig[bohrDir], rks_wfn, eArr[mapDirToDig[simDir]], dt)
            if abs(mu) >= method["resplimit"]:
                output[mapDirToDig[simDir], mapDirToDig[bohrDir]] = float(mu)

    filtered_output = np.sum(output, axis=1).tolist()
    return filtered_output