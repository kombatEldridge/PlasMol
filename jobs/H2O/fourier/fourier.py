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

# fourier("h2o-0.1-4k-earlykick-smallkick.npz", "fourier-h2o-0.1-4k-earlykick-smallkick.npz")
# fourier("h2o-0.1-4k-earlykick.npz", "fourier-h2o-0.1-4k-earlykick.npz")
# fourier("h2o-0.1-4k-smallkick.npz", "fourier-h2o-0.1-4k-smallkick.npz")
# fourier("h2o-0.1-4k.npz", "fourier-h2o-0.1-4k.npz")
# fourier("h2o-0.01-4k.npz", "fourier-h2o-0.01-4k.npz")
# fourier("h2o-0.1-10k.npz", "fourier-h2o-0.1-10k.npz")
# fourier("h2o-0.1-4k-rk4.npz", "fourier-h2o-0.1-4k-rk4.npz")
# fourier("h2o-0.1-4k-earlykick-bigkick-newmu.npz", "fourier-h2o-0.1-4k-earlykick-bigkick-newmu.npz")
# fourier("h2o-0.01-4k-earlykick-bigkick-newmu-rk4.npz", "fourier-h2o-0.01-4k-earlykick-bigkick-newmu-rk4.npz")
# fourier("npz-files/h2o-0.1-4k-earliestkick.npz", "fourier-files/fourier-h2o-0.1-4k-earliestkick.npz")

fourierh2o014kearlykicksmallkick = np.load("fourier-files/fourier-h2o-0.1-4k-earlykick-smallkick.npz")
fourierh2o014kearlykick = np.load("fourier-files/fourier-h2o-0.1-4k-earlykick.npz")
fourierh2o0110ksmallkick = np.load("fourier-files/fourier-h2o-0.1-10k-smallkick.npz")
fourierh2o014k = np.load("fourier-files/fourier-h2o-0.1-4k.npz")
fourierh2o00115k = np.load("fourier-files/fourier-h2o-0.01-1.5k.npz")
fourierh2o0110k = np.load("fourier-files/fourier-h2o-0.1-10k.npz")
fourierh2o014krk4 = np.load("fourier-files/fourier-h2o-0.1-4k-rk4.npz")
fourierh2o014kearlykickbigkicknewmu = np.load("fourier-files/fourier-h2o-0.1-4k-earlykick-bigkick-newmu.npz")
fourierh2o0014kearlykickbigkicknewmurk4 = np.load("fourier-files/fourier-h2o-0.01-4k-earlykick-bigkick-newmu-rk4.npz")
fourierh2o0014kearliestkick = np.load("fourier-files/fourier-h2o-0.1-4k-earliestkick.npz")

fourierh2o014kearlykicksmallkick = fourierh2o014kearlykicksmallkick['arr_0']
fourierh2o014kearlykick = fourierh2o014kearlykick['arr_0']
fourierh2o0110ksmallkick = fourierh2o0110ksmallkick['arr_0']
fourierh2o014k = fourierh2o014k['arr_0']
fourierh2o00115k = fourierh2o00115k['arr_0']
fourierh2o0110k = fourierh2o0110k['arr_0']
fourierh2o014krk4 = fourierh2o014krk4['arr_0']
fourierh2o014kearlykickbigkicknewmu = fourierh2o014kearlykickbigkicknewmu['arr_0']
fourierh2o0014kearlykickbigkicknewmurk4 = fourierh2o0014kearlykickbigkicknewmurk4['arr_0']
fourierh2o0014kearliestkick = fourierh2o0014kearliestkick['arr_0']

# Bohr
data_bohr = pd.read_fwf('fourier-files/bohr-lr-tddft.txt').values
f_bohr = data_bohr[:,0]
y_bohr = data_bohr[:,1]

# nwchem
data_nwchem = pd.read_fwf('fourier-files/nwchem-lr-tddft.txt').values
f_nwchem = data_nwchem[:,0]
y_nwchem = data_nwchem[:,1]

# nwchem
data_nwchem_2 = pd.read_fwf('fourier-files/nwchem-lr-tddft-2.txt').values
f_nwchem_2 = data_nwchem_2[:,0]
y_nwchem_2 = data_nwchem_2[:,1]

# nwchem
data_nwchem_b3lyp = pd.read_fwf('fourier-files/nwchem-lr-tddft-b3lyp.txt').values
f_nwchem_b3lyp = data_nwchem_b3lyp[:,0]
y_nwchem_b3lyp = data_nwchem_b3lyp[:,1]

# pyscf
data_pyscf = pd.read_fwf('fourier-files/pyscf-lr-tddft.txt').values
f_pyscf = data_pyscf[:,0]
y_pyscf = data_pyscf[:,1]

# pyscf_dft
data_pyscf_dft = pd.read_fwf('fourier-files/pyscf-lr-tddft-dft.txt').values
f_pyscf_dft = data_pyscf_dft[:,0]
y_pyscf_dft = data_pyscf_dft[:,1]

# pyscf_dft_4
data_pyscf_dft_4 = pd.read_fwf('fourier-files/pyscf-lr-tddft-dft-grid4.txt').values
f_pyscf_dft_4 = data_pyscf_dft_4[:,0]
y_pyscf_dft_4 = data_pyscf_dft_4[:,1]

# pyscf_dft_sym
data_pyscf_dft_sym = pd.read_fwf('fourier-files/pyscf-lr-tddft-dft-sym.txt').values
f_pyscf_dft_sym = data_pyscf_dft_sym[:,0]
y_pyscf_dft_sym = data_pyscf_dft_sym[:,1]

# pyscf_dft_b3lyp
data_pyscf_b3lyp = pd.read_fwf('fourier-files/pyscf-lr-tddft-dft-b3lyp.txt').values
f_pyscf_b3lyp = data_pyscf_b3lyp[:,0]
y_pyscf_b3lyp = data_pyscf_b3lyp[:,1]

from scipy.constants import speed_of_light, physical_constants
SPEED_OF_LIGHT = speed_of_light/physical_constants["atomic unit of velocity"][0]

def absorption(imag):
    fullsum = imag[0] + imag[1] + imag[2]
    return - 4 * np.pi * freq_eV / 3 / SPEED_OF_LIGHT * fullsum

fourierh2o014kearlykicksmallkick = absorption(fourierh2o014kearlykicksmallkick)
fourierh2o014kearlykick = absorption(fourierh2o014kearlykick)
fourierh2o0110ksmallkick = absorption(fourierh2o0110ksmallkick)
fourierh2o014k = absorption(fourierh2o014k)
fourierh2o00115k = absorption(fourierh2o00115k)
fourierh2o0110k = absorption(fourierh2o0110k)
fourierh2o014krk4 = absorption(fourierh2o014krk4)
fourierh2o014kearlykickbigkicknewmu = absorption(fourierh2o014kearlykickbigkicknewmu)
fourierh2o0014kearlykickbigkicknewmurk4 = absorption(fourierh2o0014kearlykickbigkicknewmurk4)
fourierh2o0014kearliestkick = absorption(fourierh2o0014kearliestkick)

fig = plt.figure(figsize=(14, 8))

# plt.plot(freq_eV, fourierh2o014kearlykicksmallkick/max(fourierh2o014kearlykicksmallkick), color='red', label='Resolution: 0.1; Length: 4k au; Kick at 0.1 au; Intensity: 5e-6 au')
# plt.plot(freq_eV, fourierh2o014kearlykick/max(fourierh2o014kearlykick), color='black', label='RT-TDDFT (PlasMol)') # Resolution: 0.1 au; Length: 4k au; Kick at 0.1 au; Intensity: 5e-5 au
# plt.plot(freq_eV, fourierh2o0110ksmallkick/max(fourierh2o0110ksmallkick), color='yellow', label='Resolution: 0.1; Length: 10k au; Small Kick')
# plt.plot(freq_eV, fourierh2o014k/max(fourierh2o014k), color='green', label='Resolution: 0.1 au; Length: 4k au')
# plt.plot(freq_eV, fourierh2o00115k/max(fourierh2o00115k), color='blue', label='Resolution: 0.01; Length: 1.5k au; Magnus2')
# plt.plot(freq_eV, fourierh2o0110k/max(fourierh2o0110k), color='indigo', label='Resolution: 0.1; Length: 10k au')
# plt.plot(freq_eV, fourierh2o014krk4/max(fourierh2o014krk4), color='lightgrey', label='Resolution: 0.1; Length: 4k au; RK4')
# plt.plot(freq_eV, fourierh2o014kearlykickbigkicknewmu/max(fourierh2o014kearlykickbigkicknewmu), color='violet', label='Resolution: 0.1 au; Length: 4k au; Kick at 0.1 au; Intensity: 1e-4 au')
# plt.plot(freq_eV, fourierh2o0014kearlykickbigkicknewmurk4/max(fourierh2o0014kearlykickbigkicknewmurk4), color='green', label='Resolution: 0.01 au; Length: 4k au; RK4')
plt.plot(freq_eV, fourierh2o0014kearliestkick/max(fourierh2o0014kearliestkick), color='green', label='RT-TDDFT (PlasMol) Early Kick') # Resolution: 0.1 au; Length: 4k au; Kick at 0.1 au; Intensity: 5e-5 au

# plt.plot(f_bohr, y_bohr/max(y_bohr), '--', color='red', label='Bohr spectrum')
# plt.plot(f_nwchem, y_nwchem/max(y_nwchem), '--', color='green', label='nwchem spectrum')
plt.plot(f_nwchem_2, y_nwchem_2/max(y_nwchem_2), '--', color='blue', label='LR-TDDFT (nwchem)')
# plt.plot(f_nwchem_b3lyp, y_nwchem_b3lyp/max(y_nwchem_b3lyp), '--', color='pink', label='nwchem spectrum b3lyp')
# plt.plot(f_pyscf, y_pyscf/max(y_pyscf), '.', color='red', label='LR-TDDFT (pyscf)')
# plt.plot(f_pyscf_b3lyp, y_pyscf_b3lyp/max(y_pyscf_b3lyp), color='black', label='pyscf spectrum b3lyp')
# plt.plot(f_pyscf_dft, y_pyscf_dft/max(y_pyscf_dft), color='blue', label='pyscf dft spectrum')
# plt.plot(f_pyscf_dft_4, y_pyscf_dft_4/max(y_pyscf_dft_4), color='pink', label='pyscf dft spectrum grid = 4')
# plt.plot(f_pyscf_dft_sym, y_pyscf_dft_sym/max(y_pyscf_dft_sym), color='black', label='pyscf dft spectrum C2v sym')

plt.xlabel('Angular frequency ω (eV)', fontsize=16)
plt.ylabel('Absorption', fontsize=16)
plt.title('Absorption Spectrum of Water', fontsize=20)
plt.grid(True)
plt.legend(fontsize=16)
plt.tight_layout()
# plt.savefig('EarliestKick.png', dpi=600)
plt.show()