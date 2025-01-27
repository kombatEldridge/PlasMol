from mu import calculate_ind_dipole
from volume import get_volume

def run(dt, eArr, method, coords, wfn, D_mo_0):
    if method == 'rk4':
        from rk4 import propagate_density_matrix
    elif method == 'magnus':
        from magnus import propagate_density_matrix

    # Take the dipole moment and divide it by the volume of the molecule
    # according to https://en.wikipedia.org/wiki/Polarization_density#Definition

    # Dividing it gives the Polarization density 
    # https://en.wikipedia.org/wiki/Current_density#Polarization_and_magnetization_currents

    volume = get_volume(coords)
    induced_dipole_matrix = calculate_ind_dipole(propagate_density_matrix, dt, eArr, wfn, D_mo_0) / volume
        
    # Should be [p_x, p_y, p_z] where p is the dipole moment
    return induced_dipole_matrix 