from volume import get_volume
import matrix_handler as mh
import numpy as np


def calculate_ind_dipole(propagate_density_matrix, dt, exc, wfn, D_mo_0):
    output = np.zeros((3, 3))

    for j in [0, 1, 2]: 
        D_mo_t_plus_dt_j = propagate_density_matrix(dt, wfn, exc, D_mo_0, j)

        for i in [0, 1, 2]:
            # Repisky2015.pdf Eq. 22
            mu_ij = np.trace(wfn.mu[i] @ D_mo_t_plus_dt_j) - np.trace(wfn.mu[i] @ D_mo_0)
            output[i][j] = float(mu_ij.real)

    # Should be [p_x, p_y, p_z] where p is the dipole moment in AU
    return output


def run(dt, eArr, method, coords, wfn, D_mo_0):
    if method == 'rk4':
        from rk4 import propagate_density_matrix
    elif method == 'magnus':
        from magnus import propagate_density_matrix

    volume = get_volume(coords)
    induced_dipole_matrix = calculate_ind_dipole(propagate_density_matrix, dt, eArr, wfn, D_mo_0) / volume
    collapsed_output = np.sum(induced_dipole_matrix, axis=1)

    # Should be [p_x, p_y, p_z] where p is the dipole moment in AU
    return collapsed_output