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

half_frame = s*resolution/2

# This gets called every half time step (FDTD nuance)
# chirp = lambda t: dipArr[str(round(t, decimals))]

def chirp(t):
    if(dipArr[str(round(t, decimals))] == 0):
        return 0
    else:
        return dipArr[str(round(t, decimals))]

# We'll put mol at 0.005 um off NP surface in x dir
mol_pos_x = r + 0.005
mol_pos_y = 0
mol_pos_z = 0

mol_dim_x = 0.001
mol_dim_y = 0.001
mol_dim_z = 0.001

# Gaussian-shaped planewave
sources = [
    mp.Source(
        mp.ContinuousSource(frequency=100, is_integrated=True), # freq=10 is 100nm
        center=mp.Vector3(-0.5*s+dpml),
        size=mp.Vector3(0, s, s),
        component=mp.Ez
    ),
    # mp.Source(
    #     mp.GaussianSource(frq_cen, fwidth=dfrq, is_integrated=True),
    #     center=mp.Vector3(-0.5*s+dpml),
    #     size=mp.Vector3(0, s, s),
    #     component=mp.Ez,
    # ), 
    mp.Source(
        mp.CustomSource(src_func=chirp),
        center=mp.Vector3(mol_pos_x, mol_pos_y, mol_pos_z),
        component=mp.Ez
    )
]

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

# one time unit is 3.33 fs
dT = sim.Courant/resolution

initDIP = 0

# init as dict
dipArr = {}
dipArr[str(0*dT)] = initDIP
dipArr[str(0.5*dT)] = initDIP
dipArr[str(1*dT)] = initDIP

num1_str = str(dT/2)
decimal_index = num1_str.find('.')
decimals = len(num1_str) - decimal_index - 1

Ez_arr = []

def getSlice(sim):
    global Ez_arr
    print("Getting slice: ", sim.meep_time(), " meep time")
    slice = sim.get_array(component=mp.Ez, 
                          center=mp.Vector3(mol_pos_x, mol_pos_y, mol_pos_z),
                          size=mp.Vector3(1E-20, 1E-20, 1E-20))
    slice = np.mean(slice)
    Ez_arr.append(slice)
    dipArr[str(round(sim.meep_time() + (0.5*dT), decimals))] = initDIP
    dipArr[str(round(sim.meep_time() + dT,       decimals))] = initDIP
    return 0

def callRTTDDFT(sim):
    global Ez_arr
    # Convert Ez_arr values to SI
    # one E field unit in meep is I (current) / a (length unit) / epsi_0 (vacuum permit) / c (speed of light) N/C
    # one E field unit in bohr is 0.51422082E12 N/C
    Ez_arr = np.array(Ez_arr) * (1 / 1e-6 / constants.epsilon_0 / constants.c) / 0.51422082e12

    # If this is happening at the atomic scale, 
    # one time unit in meep is 3.33333333E-15 s or 333.33333E-17 s
    # one time unit in bohr is 2.4188843265857E-17 s
    dTbohr = dT * (333.3333333333333/2.4188843265857)
    print(dT * 3.33e-15)
    print(dTbohr * 2.4188843265857e-17)

    print("Calling RT-TDDFT Code: ", sim.meep_time(), " meep time")
    print("Ez_arr: ", Ez_arr)
    molResponse = 0 if np.mean(Ez_arr) < 1e-9 else bohr.run(inputfile, Ez_arr, dTbohr)
    print("molResponse: ", molResponse, "\n")
    dipArr[str(round(sim.meep_time() + (0.5*dT), decimals))] = molResponse
    dipArr[str(round(sim.meep_time() + dT,       decimals))] = molResponse
    Ez_arr = []
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

sim.run(
    mp.at_every(dT, getSlice),
    mp.at_every(3*dT, callRTTDDFT), 
    mp.at_every(10*dT, mp.output_png(mp.Ez, f"-X 10 -Y 10 -R -z {half_frame} -Zc RdBu")),
    until=900*dT)

make_gif(script_name)

