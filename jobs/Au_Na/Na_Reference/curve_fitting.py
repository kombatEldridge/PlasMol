import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from pathlib import Path

# ========================== CONFIG ==========================
csv_file = "spectrum.csv"
prominence = 0.05
energy_range_ev = (1.0, 6.0)
# ===============================================================

print(f"Loading CSV: {csv_file}")
df = pd.read_csv(csv_file, sep=',')

energy_ev = df['Frequency'].values
absorption = df['Absorption'].values

mask = (energy_ev >= energy_range_ev[0]) & (energy_ev <= energy_range_ev[1])
energy_ev = energy_ev[mask]
spectrum = absorption[mask]

print(f"Fitting Absorption")

peaks_idx, _ = find_peaks(spectrum, prominence=prominence * spectrum.max(), distance=8)
peak_energies = energy_ev[peaks_idx]

print(f"\nDetected {len(peak_energies)} peak(s) at: {peak_energies.round(3)} eV\n")

def multi_lorentzian(x, *params):
    offset = params[-1]
    y = offset
    for i in range(0, len(params)-1, 3):
        E0, gamma, A = params[i:i+3]
        y += A * ((gamma / 2)**2) / ((x - E0)**2 + (gamma / 2)**2)
    return y

p0 = []
for i, E_guess in enumerate(peak_energies):
    idx = peaks_idx[i]
    height_guess = spectrum[idx] * 1.1
    gamma_guess = 0.12
    p0.extend([E_guess, gamma_guess, height_guess])
p0.append(0.0)  # offset

lower_bounds = [0] * len(p0)
upper_bounds = [np.inf] * len(p0)

# Force gamma to be at least ~3× your energy step (physically reasonable)
for i in range(0, len(p0)-1, 3):          # gamma parameters
    lower_bounds[i+1] = 0.06              # ← change to 0.10 if you prefer slightly narrower

# Optional: allow tiny negative offset if needed
lower_bounds[-1] = -0.1

popt, _ = curve_fit(multi_lorentzian, energy_ev, spectrum, p0=p0,
                    bounds=(lower_bounds, upper_bounds), maxfev=10000)

offset = popt[-1]
print(f"Fitted offset: {offset:.4f}")
print(f"Peak height above baseline: {1.9211 + offset:.4f}  ← should be ≈ 1.0")
print(f"Mean energy step size: {np.mean(np.diff(energy_ev)):.4f} eV")

peaks = []
for i in range(0, len(popt)-1, 3):
    E, gamma, A = popt[i:i+3]
    peaks.append((E, gamma, A))

peak_list = []
for E, gamma, A in peaks:
    lambda_nm = 1239.84193 / E
    peak_list.append((E, gamma, A, lambda_nm))

# Sort by wavelength ascending (smallest λ first)
peak_list.sort(key=lambda x: x[3])

print("=== ALL FITTED PEAKS ===")
print("Peak |   E (eV)   |  λ (nm)   |   γ (eV)   |   Amp    ")
print("-" * 70)

for idx, (E, gamma, A, lambda_nm) in enumerate(peak_list):
    print(f"{idx:4d} | {E:8.4f}   | {lambda_nm:7.1f}   | {gamma:8.4f}   | {A:8.4f} ")

print("-" * 70)

plt.figure(figsize=(11, 6))
plt.plot(energy_ev, spectrum, 'b-', label=f'Raw Absorption', lw=2)

x_fit = np.linspace(energy_ev.min(), energy_ev.max(), 2000)
y_fit = multi_lorentzian(x_fit, *popt)
plt.plot(x_fit, y_fit, 'k--', label='Total fit', lw=2.2)

offset = popt[-1]
for idx, (E, gamma, A, lambda_nm) in enumerate(peak_list):
    y_comp = A * ((gamma / 2)**2) / ((x_fit - E)**2 + (gamma / 2)**2)
    plt.plot(x_fit, y_comp + offset, '--', lw=1.6, label=f'Peak {idx} (λ={lambda_nm:.0f} nm)')

# add the fitted parameters as text in the plot
for idx, (E, gamma, A, lambda_nm) in enumerate(peak_list):
    plt.text(E + 1, A - .1, f'Energy: {E:.2f} eV', ha='center', va='bottom')
    plt.text(E + 1, A - .15, f'Gamma: {gamma:.2f} eV', ha='center', va='bottom')

plt.xlabel('Energy (eV)')
plt.ylabel('Absorption (arb. units)')
plt.title(f'Absorption Fit — {Path(csv_file).stem}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('peak_fit.png', dpi=400)