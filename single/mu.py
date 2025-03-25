# mu.py
from volume import get_volume
import numpy as np

def calculate_ind_dipole(propagate_density_matrix, dt, molecule, field):
    output = np.zeros((3, 3))

    for j in [0, 1, 2]: 
        D_ao_t_plus_dt = propagate_density_matrix(dt, molecule, field)

        for i in [0, 1, 2]:
            # Repisky2015.pdf Eq. 22
            output[i][j] = 2 * float((np.trace(molecule.wfn.mu[i] @ D_ao_t_plus_dt) - np.trace(molecule.wfn.mu[i] @ molecule.wfn.D[0])).real)

    return output


def run(dt, molecule, field):
    # from magnus import propagate_density_matrix
    from magnus4 import propagate_density_matrix

    volume = get_volume(molecule.molecule["coords"])
    induced_dipole_matrix = calculate_ind_dipole(propagate_density_matrix, dt, molecule, field) / volume
    collapsed_output = np.sum(induced_dipole_matrix, axis=1)

    # Should be [p_x, p_y, p_z] where p is the dipole moment in au
    return collapsed_output