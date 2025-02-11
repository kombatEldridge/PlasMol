from volume import get_volume
import matrix_handler as mh
import numpy as np


def calculate_energy(wfn, D_mo_t_plus_dt):
    F_ao_t_plus_dt = mh.get_F_ao_t()
    D_ao_t_plus_dt = wfn.C[0] @ D_mo_t_plus_dt @ wfn.C[0].T
    E_el = 0.5 * np.trace(D_ao_t_plus_dt @ F_ao_t_plus_dt)  # Electronic energy
    E_tot = E_el + wfn.e_nuc  # Add nuclear repulsion energy
    return E_tot.real


def calculate_ind_dipole(propagate_density_matrix, dt, exc, wfn, D_mo_0):
    output = np.zeros(3)
    D_mo_t_plus_dt = propagate_density_matrix(dt, wfn, exc, D_mo_0)

    for dir in [0, 1, 2]: 
        # Repisky2015.pdf Eq. 22
        mu_i = np.trace(wfn.mu[dir] @ D_mo_t_plus_dt) - np.trace(wfn.mu[dir] @ D_mo_0)

        output[dir] = float(mu_i.real)

    # Should be [p_x, p_y, p_z] where p is the dipole moment in AU
    return output, D_mo_t_plus_dt

 
def run(dt, eArr, method, coords, wfn, D_mo_0):
    if method == 'rk4':
        from rk4 import propagate_density_matrix
    elif method == 'magnus':
        from magnus import propagate_density_matrix

    volume = get_volume(coords)
    induced_dipole_matrix, D_mo_t_plus_dt = calculate_ind_dipole(propagate_density_matrix, dt, eArr, wfn, D_mo_0)
    induced_dipole_matrix /= volume
    energy = calculate_energy(wfn, D_mo_t_plus_dt)

    # Should be [p_x, p_y, p_z] where p is the dipole moment in AU
    return induced_dipole_matrix, energy