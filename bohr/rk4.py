import numpy as np
import os
from bohr.bohr import JK

def rk4_ind_dipole(direction1, direction2, wfn, eField, dt, storage_file='D_ao_np1.npy', temp_storage_file='D_ao_np2.npy'):
    # Load D_ao from the storage file if it exists; otherwise initialize
    if os.path.exists(storage_file):
        D_ao = np.load(storage_file)
    else:
        D_ao = wfn.D[0]

    D_ao_init = D_ao
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

    if os.path.exists(temp_storage_file):
        temp_D_ao_load = np.load(temp_storage_file)
        np.save(storage_file, temp_D_ao_load)

    np.save(temp_storage_file, D_ao)


    # Calculate the induced dipole moment in the direction2
    mu = np.trace(wfn.mu[direction2] @ D_ao) - np.trace(wfn.mu[direction2] @ D_ao_init)
    return mu.real
