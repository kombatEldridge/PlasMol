# fourier.py
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .. import constants

logger = logging.getLogger("main")

init_w_eV = float(0)
final_w_eV = float(35)
step_w_eV = float(0.1)
nw = int(final_w_eV/step_w_eV)
freq_eV = np.arange(init_w_eV, final_w_eV, step_w_eV)

def fold(file_x, file_y, file_z):
    def read_dipole_component(filename, column):
        df = pd.read_csv(filename, delimiter=',', skiprows=1, header=None, names=['Timestamps (au)', 'X Values', 'Y Values', 'Z Values'], comment='#')
        df = df.dropna()
        
        # Extract columns and convert to NumPy arrays explicitly
        time = np.array(df['Timestamps (au)'].values[1:], dtype=float)
        dipole = np.array(df[column].values[1:], dtype=float)
        
        # Return as NumPy arrays
        return np.array(time), np.array(dipole)

    # Load components
    tx, dx = read_dipole_component(file_x, 'X Values')
    ty, dy = read_dipole_component(file_y, 'Y Values')
    tz, dz = read_dipole_component(file_z, 'Z Values')
    
    # Find the length of the shortest file (assuming timesteps are same)
    min_length = min(len(tx), len(ty), len(tz))

    # Trim all arrays to the shortest length
    tx = tx[:min_length]
    ty = ty[:min_length]
    tz = tz[:min_length]
    dx = dx[:min_length]
    dy = dy[:min_length]
    dz = dz[:min_length]

    # Check consistency
    if not (np.allclose(tx, ty) and np.allclose(tx, tz)):
        raise ValueError("Time points do not match across files!")

    # Build time_points and dipole_moment array
    time_points = tx
    dipole_moment = np.vstack([dx, dy, dz])  # Shape: (3, N)

    return time_points, dipole_moment


def fourier(time, dipole, file):
    dt = time[1] - time[0]

    damp = 0.010
    logger.debug("Damping factor gamma: %f",damp)

    abs_real = [[],[],[]]
    abs_imag = [[],[],[]]

    for axis in (0, 1, 2):
        logger.info(f"Starting Fourier transform of direction { {0:'x', 1:'y', 2:'z'}[axis] }")
        for step_num in range(nw):
            w = freq_eV[step_num]/27.21138 # converted to au
            S = 0j
            for k in range(len(time)):
                S += dipole[axis][k] * np.exp(-1j * w * time[k]) * np.exp(-damp * time[k]) * dt

            abs_real[axis].append(S.real)
            abs_imag[axis].append(S.imag)
            
    logger.info("Fourier transform done!")
    for i in range(3):
        abs_real[i] = np.array(abs_real[i])
        abs_imag[i] = np.array(abs_imag[i])

    np.savez(file, abs_imag)
    return abs_imag


def absorption(imag):
    fullsum = imag[0] + imag[1] + imag[2]
    return - 4 * np.pi * freq_eV / 3 / constants.C_AU * fullsum


def transform(file_x, file_y, file_z,  pField_Transform_path, eV_spectrum_path):
    time_points, dipole_moment = fold(file_x, file_y, file_z)
    abs_imag = fourier(time_points, dipole_moment, pField_Transform_path)
    abs = absorption(abs_imag)

    fig = plt.figure(figsize=(14, 8))
    plt.plot(freq_eV, abs/max(abs), color='green', label='Spectrum')
    plt.xlabel('Angular frequency ω (eV)', fontsize=16)
    plt.ylabel('Absorption', fontsize=16)
    plt.title('Absorption Spectrum of Water', fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(eV_spectrum_path, dpi=600)
    plt.show()