# fourier.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def fourier(polarizability_csv, output_image='spectrum.png'):
    # Load CSV (assumes columns: 'Timestamps (fs)', 'X Values', 'Y Values', 'Z Values')
    df = pd.read_csv(polarizability_csv, comment='#')
    
    time = df['Timestamps (fs)'].values * 1e-15  # Convert fs to seconds
    signal_x = df['X Values'].values
    signal_y = df['Y Values'].values
    signal_z = df['Z Values'].values

    # Time step (assumes uniform spacing)
    dt = time[1] - time[0]

    # Perform FFT
    fft_vals_x = np.fft.fft(signal_x)
    fft_vals_y = np.fft.fft(signal_y)
    fft_vals_z = np.fft.fft(signal_z)
    freqs = np.fft.fftfreq(len(time), d=dt)  # In Hz

    # Keep only positive frequencies
    mask = freqs > 0
    freqs = freqs[mask]
    fft_vals_x = fft_vals_x[mask]
    fft_vals_y = fft_vals_y[mask]
    fft_vals_z = fft_vals_z[mask]

    # Convert frequencies to wavelengths in nm
    c = 2.99792458e8  # speed of light in m/s
    wavelengths_nm = (c / freqs) * 1e9  # Convert to nm

    # Sort by wavelength for better plotting (descending wavelength = ascending freq)
    sort_indices = np.argsort(wavelengths_nm)
    wavelengths_nm = wavelengths_nm[sort_indices]
    fft_x = np.abs(fft_vals_x[sort_indices])
    fft_y = np.abs(fft_vals_y[sort_indices])
    fft_z = np.abs(fft_vals_z[sort_indices])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths_nm, fft_x, label='X polarization')
    plt.plot(wavelengths_nm, fft_y, label='Y polarization')
    plt.plot(wavelengths_nm, fft_z, label='Z polarization')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Magnitude')
    plt.title('Absorption Spectrum')
    plt.legend()
    plt.grid(True)
    plt.xlim(wavelengths_nm.min(), wavelengths_nm.max())  # Optional, ensures full range shown
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"Saved spectrum to {output_image}")
