import numpy as np
import matrix_handler as mh

def calculate_ind_dipole(propagate_density_matrix, dt, exc, wfn):
    output = np.zeros(3)
    D_mo_0 = mh.get_D_mo_0()
    D_mo_dt = propagate_density_matrix(dt, wfn, exc)

    for dir in [0, 1, 2]: 
        # Repisky2015.pdf Eq. 22
        mu_i = np.trace(wfn.mu[dir] @ D_mo_dt) - np.trace(wfn.mu[dir] @ D_mo_0)

        output[dir] = float(mu_i.real)

    # Should be [p_x, p_y, p_z] where p is the dipole moment
    return output

 