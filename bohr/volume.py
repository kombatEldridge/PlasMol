import math

def get_volume(xyz):
    # Define van der Waals radii (Å)
    vdw_radii = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52}

    volume = 0
    for atom in xyz:
        element = atom[0]
        if element in vdw_radii:
            radius = vdw_radii[element]
            volume += (4 / 3) * math.pi * (radius ** 3)
        else:
            print(f"Warning: No van der Waals radius for {element}")
        
    # Dividing by 0.14818471 Å³ will set the volume to atomic units.
    return volume / 0.14818471