# fourier.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

init_w_eV = float(0)
final_w_eV = float(35)
step_w_eV = float(0.1)
nw = int(final_w_eV/step_w_eV)
freq_eV = np.arange(init_w_eV, final_w_eV, step_w_eV)

def fourier(ofn, fn):
    data           = np.load(ofn, allow_pickle=True)
    dipole         = data["dipole_moment"]
    time           = data["time_points"]
    dt             = float(data["dt"])
    print(f"{ofn} with {data['basis']} basis, using RT-TD{data['method']} with {len(time)} time points.")

    damp = 0.010
    print("Damping factor gamma: %f",damp)

    abs_real = [[],[],[]]
    abs_imag = [[],[],[]]

    for axis in (0, 1, 2):
        print(f"Starting direction { {0:'x', 1:'y', 2:'z'}[axis] }")
        for step_num in range(nw):
            w = freq_eV[step_num]/27.21138 # converted to au
            S = 0j
            for k in range(len(time)):
                S += dipole[axis][k] * np.exp(-1j * w * time[k]) * np.exp(-damp * time[k]) * dt

            abs_real[axis].append(S.real)
            abs_imag[axis].append(S.imag)
            
    print("Fourier transform done!")
    for i in range(3):
        abs_real[i] = np.array(abs_real[i])
        abs_imag[i] = np.array(abs_imag[i])

    np.savez(fn, abs_imag)
    return abs_imag

fourier("rttddft.npz", "transformed-rttddft.npz")
rttddft = np.load("transformed-rttddft.npz")
rttddft = rttddft['arr_0']

# nwchem
data_nwchem = pd.read_fwf('nwchem-lr-tddft.txt').values
f_nwchem = data_nwchem[:,0]
y_nwchem = data_nwchem[:,1]

from scipy.constants import speed_of_light, physical_constants
SPEED_OF_LIGHT = speed_of_light/physical_constants["atomic unit of velocity"][0]

def absorption(imag):
    fullsum = imag[0] + imag[1] + imag[2]
    return - 4 * np.pi * freq_eV / 3 / SPEED_OF_LIGHT * fullsum

abs_rttddft = absorption(rttddft)

fig = plt.figure(figsize=(14, 8))
plt.plot(freq_eV, abs_rttddft/max(abs(abs_rttddft)), color='blue', label='RT-TDDFT')
plt.plot(f_nwchem, y_nwchem/max(y_nwchem), '--', color='green', label='LR-TDDFT')
plt.xlabel('Angular frequency ω (eV)')
plt.ylabel('Absorption')
plt.title('Fourier Transform of μ(t)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()