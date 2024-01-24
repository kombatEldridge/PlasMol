import numpy as np
import meep as mp
import sys
import time
from meep.materials import Au_JC_visible as Au
import eps_fit_lorentzian
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
frq_range = mp.FreqRange(min=frq_min, max=frq_max)

resolution = 1000

dpml = 0.01

pml_layers = [mp.PML(thickness=dpml)]

symmetries = [mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)]

s = 0.1
cell_size = mp.Vector3(s, s, s)

half_frame = s*resolution/2

dipArr = [0]

chirp = lambda t: dipArr[t]

# Gaussian-shaped planewave
sources = [
    mp.Source(
        mp.GaussianSource(frq_cen, fwidth=dfrq, is_integrated=True),
        center=mp.Vector3(-0.5*s+dpml),
        size=mp.Vector3(0, s, s),
        component=mp.Ez,
    ), 
    mp.Source(
        mp.CustomSource(src_func=chirp),
        center=mp.Vector3(mol_pos_x, mol_pos_y, mol_pos_z),
    )
]

# We'll put mol at 0.005 um off NP surface in x dir
mol_pos_x = r + 0.005
mol_pos_y = 0
mol_pos_z = 0

mol_dim_x = 0.001
mol_dim_y = 0.001

geometry = [mp.Sphere(radius=r,
                      center=mp.Vector3(),
                      material=Au)]

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
    dipArr.append(0)
    return 0

def callRTTDDFT(sim):
    global Ez_arr
    print("Calling RT-TDDFT Code")
    dipArr[-1] = bohr.run(inputfile, Ez_arr)
    Ez_arr = []
    return 0

sim.run(mp.at_every(0.1, getSlice), 
        mp.at_every(0.3, callRTTDDFT), 
        until_after_sources=50)

