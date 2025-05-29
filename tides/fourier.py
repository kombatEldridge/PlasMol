# fourier.py
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
AU_TO_S = 2.418884e-17      # atomic unit of time to seconds
FS_TO_S = 1e-15              # femtoseconds to seconds
C_M_S = 299792458            # speed of light in m/s
C_NM_S = C_M_S * 1e9         # speed of light in nm/s


def load_time_series(csv_path):
    """
    Load time series data from a CSV file.

    Reads a CSV file with timestamps in either atomic units or femtoseconds,
    converting them to seconds, and extracts X, Y, Z component values.

    Parameters:
    csv_path : str
        Path to the CSV file.

    Returns:
    tuple
        (t, signals) where t is an np.ndarray of times in seconds, and signals is a dict
        with 'X', 'Y', 'Z' keys mapping to np.ndarray of corresponding values.
    """
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
    """
    Compute the Fourier spectrum of a time series signal.

    Performs FFT on the detrended signal, converting frequencies to wavelengths in nanometers.

    Parameters:
    t : np.ndarray
        Time values in seconds.
    signal : np.ndarray
        Signal values corresponding to the time points.

    Returns:
    tuple
        (wavelengths, intensities) where wavelengths is an np.ndarray in nanometers,
        and intensities is an np.ndarray of squared FFT magnitudes.
    """
    # detrend (remove mean)
    sig = signal - np.mean(signal)
    n = len(t)
    dt = np.diff(t)
    if not np.allclose(dt, dt[0], rtol=1e-10):
        raise ValueError("Time series must have uniform spacing for FFT")
    # FFT
    fft_vals = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, d=dt[0])
    # ignore DC
    mask = freqs > 0
    freqs = freqs[mask]
    intensities = np.abs(fft_vals[mask])**2
    # convert to wavelength (nm)
    wavelengths = C_NM_S / freqs
    return wavelengths, intensities


def gaussian_broaden(wl, I, wl_grid, sigma):
    """
    Apply Gaussian broadening to a spectrum.

    Broadens the spectrum intensities onto a specified wavelength grid using a Gaussian function.

    Parameters:
    wl : np.ndarray
        Original wavelengths of the spectrum in nanometers.
    I : np.ndarray
        Original intensities of the spectrum.
    wl_grid : np.ndarray
        Target wavelength grid in nanometers for broadening.
    sigma : float
        Standard deviation of the Gaussian in nanometers.

    Returns:
    np.ndarray
        Broadened intensities on wl_grid.
    """
    # Build matrix of Gaussians: shape (len(I), len(wl_grid))
    diff = wl[:, None] - wl_grid[None, :]
    gauss = np.exp(-0.5 * (diff / sigma)**2)
    # Weighted sum over peaks
    return np.dot(I, gauss)


def fourier(filename, sigma=10, npoints=2000, output_image='spectrum.png'):
    """
    Compute and plot the absorption spectrum from a time series CSV file.

    Loads time series data, computes the Fourier spectrum for X, Y, Z components,
    applies Gaussian broadening, plots the result, and saves both the plot and data.

    Parameters:
    filename : str
        Path to the input CSV file with time series data.
    sigma : float, optional
        Standard deviation for Gaussian broadening in nanometers (default 10).
    npoints : int, optional
        Number of points in the wavelength grid (default 2000).
    output_image : str, optional
        Path to save the output spectrum image (default 'spectrum.png').

    Returns:
    None
    """
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
    
    # Find the global maximum across all broadened spectra
    max_I = max(np.max(broads['X']), np.max(broads['Y']), np.max(broads['Z']))
                
    # Plot
    plt.figure(figsize=(8, 5))
    for axis, I_b in broads.items():
        plt.plot(wl_grid, I_b / max_I, label=f"{axis}-dipole")
    ax = plt.gca()
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Intensity") 
    plt.title("Absorption Spectrum with Gaussian Broadening")
    plt.xlim(wl_min, 2000)
    wl_left, wl_right = ax.get_xlim()
    plt.legend()

    # === Add top axis in eV ===
    HC_eVnm = 1239.841984
    def nm_to_eV(wl_nm):   return HC_eVnm / np.maximum(wl_nm, 1e-10) 
    def eV_to_nm(E_eV):    return HC_eVnm / np.maximum(E_eV, 1e-10)

    eV_left, eV_right = nm_to_eV(wl_left), nm_to_eV(wl_right)

    secax = ax.secondary_xaxis('top', functions=(nm_to_eV, eV_to_nm))
    secax.set_xlabel('Energy (eV)')

    ticks = [0.5, 1, 2, 3, 4, 5, 10]
    labels = [str(int(tick)) if tick == int(tick) else str(tick) for tick in ticks]
    secax.set_xticks(ticks)
    secax.set_xticklabels(labels)
    secax.set_xlim(eV_left, eV_right)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"Saved spectrum to {output_image}")
    
    # Export spectrum data to CSV
    output_csv = filename.replace('.csv', '_spectrum.csv')
    df = pd.DataFrame({
        'Wavelength (nm)': wl_grid,
        'Intensity_X': broads['X'] / max_I,
        'Intensity_Y': broads['Y'] / max_I,
        'Intensity_Z': broads['Z'] / max_I
    })
    df.to_csv(output_csv, index=False)
    print(f"Exported spectrum data to {output_csv}")


if __name__ == "__main__":
    args = sys.argv
    fourier(sys.argv[1])
