# fourier.py
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
This script reads a CSV file containing time-domain dipole signals (X, Y, Z)
and computes an absorption spectrum in wavelength (nm) with Gaussian broadening.
It supports two possible time units:
  - Timestamps (au): atomic units (1 au = 2.418884e-17 s)
  - Timestamps (fs): femtoseconds (1 fs = 1e-15 s)

Usage:
  python fourier_transform_absorption.py input.csv [--sigma SIGMA] [--npoints N]

Arguments:
  input.csv      Path to the input CSV file with headers:
                   either ['Timestamps (au)', 'X Values', 'Y Values', 'Z Values']
                   or       ['Timestamps (fs)', 'X Values', 'Y Values', 'Z Values']

Options:
  --sigma SIGMA  Gaussian broadening width (in nm), default=10
  --npoints N    Number of points in the wavelength grid, default=2000

Output:
  Displays a plot of broadened absorption spectra for X, Y, Z components
  in the range 0-2000 nm.
"""
# Constants
AU_TO_S = 2.418884e-17      # atomic unit of time to seconds
FS_TO_S = 1e-15              # femtoseconds to seconds
C_M_S = 299792458            # speed of light in m/s
C_NM_S = C_M_S * 1e9         # speed of light in nm/s


def load_time_series(csv_path):
    df = pd.read_csv(csv_path, comment='#')
    if 'Timestamps (au)' in df.columns:
        t = df['Timestamps (au)'].values * AU_TO_S
    elif 'Timestamps (fs)' in df.columns:
        t = df['Timestamps (fs)'].values * FS_TO_S
    else:
        raise ValueError("CSV must contain 'Timestamps (au)' or 'Timestamps (fs)'")
    signals = {
        'X': df['X Values'].values,
        'Y': df['Y Values'].values,
        'Z': df['Z Values'].values
    }
    return t, signals


def compute_spectrum(t, signal):
    # detrend (remove mean)
    sig = signal - np.mean(signal)
    n = len(t)
    dt = t[1] - t[0]
    # FFT
    fft_vals = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, d=dt)
    # ignore DC
    mask = freqs > 0
    freqs = freqs[mask]
    intensities = np.abs(fft_vals[mask])**2
    # convert to wavelength (nm)
    wavelengths = C_NM_S / freqs
    return wavelengths, intensities


def gaussian_broaden(wl, I, wl_grid, sigma):
    # Build matrix of Gaussians: shape (len(I), len(wl_grid))
    diff = wl[:, None] - wl_grid[None, :]
    gauss = np.exp(-0.5 * (diff / sigma)**2)
    # Weighted sum over peaks
    return np.dot(I, gauss)


def fourier(filename, sigma=10, npoints=2000, output_image='spectrum.png'):
    # Load data
    t, signals = load_time_series(filename)

    # Compute individual spectra
    spectra = {}
    for axis, sig in signals.items():
        wl, I = compute_spectrum(t, sig)
        # Limit to wavelengths <= 2000 nm
        mask = wl <= 2000.0
        spectra[axis] = (wl[mask], I[mask])

    # Create common wavelength grid
    wl_min = min(wl.min() for wl, _ in spectra.values())
    wl_grid = np.linspace(wl_min, 2000.0, npoints)

    # Apply Gaussian broadening
    broads = {}
    for axis, (wl_vals, I_vals) in spectra.items():
        broads[axis] = gaussian_broaden(wl_vals, I_vals, wl_grid, sigma)

    # Plot
    plt.figure(figsize=(8, 5))
    for axis, I_b in broads.items():
        plt.plot(wl_grid, I_b, label=f"{axis}-dipole")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (arb. units)")
    plt.title("Absorption Spectrum with Gaussian Broadening")
    plt.xlim(0, 2000)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"Saved spectrum to {output_image}")


if __name__ == "__main__":
    args = sys.argv
    fourier(sys.argv[1])
