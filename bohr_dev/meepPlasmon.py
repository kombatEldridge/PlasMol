import os
import numpy as np
import meep as mp
import sys
from gif import make_gif
from meep.materials import Au_JC_visible as Au
import bohr
from scipy import constants

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

# We'll put mol at 0.010 um off NP surface in x dir
np_pos = mp.Vector3(0, 0, 0)
mol_pos = mp.Vector3(r + 0.010, 0, 0)

# Unused right now, but defines volume of molecule
# mol_dim = mp.Vector3(0.001, 0.001, 0.001)

# positive x dir goes left to right
# positive y dir goes top to down
# positive z dir goes ???
half_frame = s*resolution/2
mol_framex = (mol_pos.x + s/2)*resolution
mol_framey = (mol_pos.y + s/2)*resolution
mol_framez = (mol_pos.z + s/2)*resolution
np_framex = (np_pos.x + s/2)*resolution
np_framey = (np_pos.y + s/2)*resolution
np_framez = (np_pos.z + s/2)*resolution


# This gets called every half time step (FDTD nuance)
def chirpx(t):
    return dipArr[0][str(round(t, decimals))] if dipArr[0][str(round(t, decimals))] != 0 else 0
def chirpy(t):
    return dipArr[1][str(round(t, decimals))] if dipArr[1][str(round(t, decimals))] != 0 else 0
def chirpz(t):
    return dipArr[2][str(round(t, decimals))] if dipArr[2][str(round(t, decimals))] != 0 else 0

sources = [
    # mp.Source(
    #     mp.ContinuousSource(frequency=100, is_integrated=True), # freq=10 is 100nm
    #     center=mp.Vector3(-0.5*s+dpml),
    #     size=mp.Vector3(0, s, s),
    #     component=mp.Ez
    # ),
    mp.Source(
        mp.GaussianSource(frq_cen, fwidth=dfrq, is_integrated=True),
        center=mp.Vector3(-0.5*s+dpml),
        size=mp.Vector3(0, s, s),
        component=mp.Ez,
    ), 
    mp.Source(
        mp.CustomSource(src_func=chirpx, is_integrated=True),
        center=mol_pos,
        component=mp.Ex, 
    ),
    mp.Source(
        mp.CustomSource(src_func=chirpy, is_integrated=True),
        center=mol_pos,
        component=mp.Ey
    ),
    mp.Source(
        mp.CustomSource(src_func=chirpz, is_integrated=True),
        center=mol_pos,
        component=mp.Ez
    )
]

geometry = [mp.Sphere(radius=r,
                      center=np_pos,
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

# one time unit is 3.33 fs
dT = sim.Courant/resolution

initDIP = 0

# init as nested-dict
dipArr = {}
for i in np.arange(0,3):
    dipArr[i] = {}
    dipArr[i][str(0*dT)] = initDIP
    dipArr[i][str(0.5*dT)] = initDIP
    dipArr[i][str(1*dT)] = initDIP

num1_str = str(dT/2)
decimal_index = num1_str.find('.')
decimals = len(num1_str) - decimal_index - 1

E_arr = [[],[],[]]
components = [mp.Ex, mp.Ey, mp.Ez]

def getSlice(sim):
    global E_arr
    print("Getting slice: ", sim.meep_time(), " meep time")
    for i in np.arange(0,3):
        E_arr[i].append(np.mean(sim.get_array(component=components[i], center=mol_pos, size=mp.Vector3(1E-20, 1E-20, 1E-20))))
        dipArr[i][str(round(sim.meep_time() + (0.5*dT), decimals))] = initDIP
        dipArr[i][str(round(sim.meep_time() + dT,       decimals))] = initDIP
    return 0

def callBohr(sim):
    global E_arr
    # Convert E_arr values to SI
    # one E field unit in meep is I (current) / a (length unit) / epsi_0 (vacuum permit) / c (speed of light) N/C
    # one E field unit in bohr is 0.51422082E12 N/C
    for i in np.arange(0,3):
        E_arr[i] = np.array(E_arr[i]) * (1 / 1e-6 / constants.epsilon_0 / constants.c) / 0.51422082e12

    # If this is happening at the atomic scale, 
    # one time unit in meep is 3.33333333E-15 s or 333.33333E-17 s
    # one time unit in bohr is 2.4188843265857E-17 s
    dTbohr = dT * (333.3333333333333/2.4188843265857)

    molResponse = [0,0,0]

    print("Calling Bohr: ", sim.meep_time(), " meep time")
    for i in np.arange(0,3):
        molResponse[i] = 0 if np.mean(E_arr[i]) < 1e-12 else bohr.run(inputfile, E_arr[0], E_arr[1], E_arr[2], dTbohr)[i]
        dipArr[i][str(round(sim.meep_time() + (0.5*dT), decimals))] = molResponse[i]
        dipArr[i][str(round(sim.meep_time() + dT,       decimals))] = molResponse[i]

    print("molResponse: ", molResponse, "\n")
    E_arr = [[],[],[]]
    return 0

script_name = "testingGifOutput"
sim.use_output_directory(script_name)

def clear_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file_name in files:
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        print(f"All files in {directory_path} have been deleted.")
    except Exception as e:
        print(f"An error occurred: {e}")

clear_directory(script_name)

# intensityMin = -0.0001
# intensityMax = 0.0001

sim.run(
    mp.at_every(dT, getSlice),
    mp.at_every(3*dT, callBohr), 
    mp.at_every(10*dT, mp.output_png(mp.Ez, f"-X 10 -Y 10 -z {half_frame} -Zc RdBu")),
    # mp.at_every(10*dT, mp.output_png(mp.Ez, f"-X 10 -Y 10 -m {intensityMin} -M {intensityMax} -z {half_frame} -Zc RdBu")),
    # mp.at_every(10*dT, mp.output_png(mp.Ex, f"-X 10 -Y 10 -m {intensityMin} -M {intensityMax} -x {mol_framex} -Zc RdBu")),
    until=900*dT)

make_gif(script_name)
