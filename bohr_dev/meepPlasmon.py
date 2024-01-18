import numpy as np
import meep as mp
import sys
import time
from meep.materials import Au_JC_visible as Au
import bohr

inputfile = sys.argv[1]

# -------- MEEP INIT -------- #

r = 0.025

wvl_min = 0.400 
wvl_max = 0.800 
frq_min = 1/wvl_max
frq_max = 1/wvl_min
frq_cen = 0.5*(frq_min+frq_max)
dfrq = frq_max-frq_min
nfrq = 50

resolution = 1000

dpml = 0.01

pml_layers = [mp.PML(thickness=dpml)]

symmetries = [mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)]

s = 0.1
cell_size = mp.Vector3(s, s, s)

half_frame = s*resolution/2

# Gaussian-shaped planewave
sources = [
    mp.Source(
        mp.GaussianSource(frq_cen, fwidth=dfrq, is_integrated=True),
        center=mp.Vector3(-0.5*s+dpml),
        size=mp.Vector3(0, s, s),
        component=mp.Ez,
    )
]

# We'll put mol at 0.005 um off NP surface in x dir
mol_pos_x = r + 0.005
mol_pos_y = 0
mol_pos_z = 0

mol_dim_x = 0.001
mol_dim_y = 0.001

# cadmium telluride (CdTe) from D.T.F. Marple, J. Applied Physics, Vol. 35, pp. 539-42 (1964)
# ref: https://refractiveindex.info/?shelf=main&book=CdTe&page=Marple
# wavelength range: 0.86 - 2.5 μm

# CdTe_range = mp.FreqRange(min=um_scale / 2.5, max=um_scale / 0.86)

# CdTe_frq1 = 1 / (0.6049793384901669 * um_scale)
# CdTe_gam1 = 0
# CdTe_sig1 = 1.53

# CdTe_susc = [
#     mp.LorentzianSusceptibility(frequency=CdTe_frq1, gamma=CdTe_gam1, sigma=CdTe_sig1)
# ]

# CdTe = mp.Medium(
#     epsilon=5.68, E_susceptibilities=CdTe_susc, valid_freq_range=CdTe_range
# )

geometry = [mp.Sphere(radius=r,
                      center=mp.Vector3(),
                      material=Au), 
            mp.Sphere(radius=0.0005,
                      center=mp.Vector3(mol_pos_x, mol_pos_y, mol_pos_z), 
                      material=)]

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    sources=sources,
    symmetries=symmetries,
    geometry=geometry,
    default_material=mp.Medium(index=1.33)
)

Ez_arr = []

def getSlice(sim):
    global Ez_arr
    print("Getting slice")
    slice = sim.get_array(component=mp.Ez, 
                          center=mp.Vector3(mol_pos_x, mol_pos_y, mol_pos_z), 
                          size=mp.Vector3(mol_dim_x, mol_dim_y))
    slice = np.mean(slice, axis=0)
    Ez_arr.append(slice)
    return 0

def callRTTDDFT(sim):
    global Ez_arr
    print("Calling RT-TDDFT Code")
    bohr.run(inputfile, Ez_arr)
    Ez_arr = []
    return 0

sim.run(mp.at_every(0.1, getSlice), mp.at_every(0.3, callRTTDDFT), until_after_sources=50)

