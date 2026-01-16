# utils/fourier.py
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plasmol import constants

logger = logging.getLogger("main")

def fold(file_x, file_y, file_z):
    def read_dipole_component(filename, column):
        # Read CSV, skip comment lines and use first non-comment row as header
        df = pd.read_csv(filename, delimiter=',', header=0, comment='#')
        # Extract time and requested dipole component as NumPy arrays
        time = np.array(df['Timestamps (au)'].values, dtype=float)
        dipole = np.array(df[column].values, dtype=float)
        logger.debug(f"Loaded {len(time)} points from {filename}")
        return time, dipole

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


def fourier(time, dipole, file, damp):
    dt = time[1] - time[0]

    logger.debug("Damping factor gamma: %f", damp)

    abs_real = [[], [], []]
    abs_imag = [[], [], []]
    freqs_out = []

    # Calculate frequencies once, as they are the same for all axes
    freqs_au = np.fft.fftfreq(len(time), d=dt) * 2 * np.pi
    freqs_ev = freqs_au * 27.211386
    mask = (freqs_ev >= 0) & (freqs_ev <= 50)    # Create mask for frequencies between 0 and 50 eV

    for axis in (0, 1, 2):
        logger.info(f"Starting Fourier transform of direction { {0:'x', 1:'y', 2:'z'}[axis] }")
        dipole_windowed = dipole[axis] * np.exp(-damp * time)
        S = np.fft.fft(dipole_windowed) * dt 
        abs_real[axis] = S.real[mask]
        abs_imag[axis] = S.imag[mask]

    # Keep only frequencies in the 0 to 50 eV range
    freqs_out = freqs_ev[mask]

    logger.info("Fourier transform done!")
    for i in range(3):
        abs_real[i] = np.array(abs_real[i])
        abs_imag[i] = np.array(abs_imag[i])

    np.savez(file, abs_imag=abs_imag, freqs=freqs_out)
    return abs_imag, freqs_out

def absorption(imag, freqs):
    fullsum = imag[0] + imag[1] + imag[2]
    return - 4 * np.pi * freqs / 3 / constants.C_AU * fullsum

def transform(file_x, file_y, file_z, pField_Transform_path, eV_spectrum_path, damp=0.01):
    time_points, dipole_moment = fold(file_x, file_y, file_z)
    abs_imag, freqs = fourier(time_points, dipole_moment, pField_Transform_path, damp)
    abs = absorption(abs_imag, freqs)

    fig = plt.figure(figsize=(14, 8))
    plt.plot(freqs, abs/max(abs), color='green', label='Spectrum')
    plt.xlabel('Angular frequency ω (eV)', fontsize=16)
    plt.ylabel('Absorption', fontsize=16)
    plt.title('Absorption Spectrum of Water', fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(eV_spectrum_path, dpi=600)
    # plt.show()