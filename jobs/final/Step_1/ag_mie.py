# Mie absorption of a bare Ag sphere in water (n = 1.33).
# Uses meep.materials.Ag, the broadband Rakic fit, not Ag_visible.
# The target is the trans-thioindigo LR-TDDFT root at 416.6 nm.
#
#   conda activate meep
#   python jobs/final/Step_1/ag_mie.py

from pathlib import Path

import matplotlib.pyplot as plt
import meep.materials as mm
import miepython
import numpy as np

HERE = Path(__file__).resolve().parent
N_ENV = 1.33
TARGET_NM = 416.6
WAVE_NM = np.arange(300.0, 600.0 + 0.5, 0.5)
# The series uses 35 nm. 40 nm is the earlier comparison.
SPECTRUM_NM = (35.0, 40.0)
DIAMETERS_NM = np.arange(10.0, 80.0 + 0.5, 0.5)


def refractive_index(material, wavelength_nm):
    """Complex index from a Meep Lorentz–Drude sum. Frequency unit is 1/μm."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    freq = 1000.0 / wavelength_nm
    eps = np.empty(wavelength_nm.shape, dtype=complex)
    for i, f in enumerate(np.atleast_1d(freq)):
        eps[i] = material.epsilon(float(f))[0, 0]
    index = np.sqrt(eps)
    index = np.where(np.imag(index) < 0.0, -index, index)
    return index


def absorption(material, diameter_nm, wavelength_nm):
    """Absorption efficiency and cross section (nm²) of one sphere."""
    m = refractive_index(material, wavelength_nm)
    qext, qsca, _qback, _g = miepython.efficiencies(
        m, diameter_nm, wavelength_nm, n_env=N_ENV
    )
    qabs = np.real(qext - qsca)
    cabs = qabs * np.pi * (0.5 * diameter_nm) ** 2
    return qabs, cabs


def interior_peak(cabs):
    """Index of the strongest interior local maximum, else the window maximum."""
    slope = np.diff(cabs)
    interior = np.where((slope[:-1] > 0.0) & (slope[1:] <= 0.0))[0] + 1
    if len(interior):
        return int(interior[np.argmax(cabs[interior])])
    return int(np.argmax(cabs))


def main():
    rows = ["diameter_nm\twavelength_nm\tqabs\tcabs_nm2"]
    curves = {}
    print(f"material Ag   n = {N_ENV}   dye root {TARGET_NM:.1f} nm")
    print(f"{'d (nm)':>8} {'peak (nm)':>10} {'Qabs':>10} {'Cabs (nm^2)':>14} {'at dye':>10}")
    for diameter in SPECTRUM_NM:
        qabs, cabs = absorption(mm.Ag, diameter, WAVE_NM)
        curves[diameter] = cabs
        peak = interior_peak(cabs)
        at_dye = int(np.argmin(np.abs(WAVE_NM - TARGET_NM)))
        print(
            f"{diameter:8.1f} {WAVE_NM[peak]:10.1f} "
            f"{qabs[peak]:10.3f} {cabs[peak]:14.1f} "
            f"{100.0 * cabs[at_dye] / cabs[peak]:9.0f}%"
        )
        for wavelength, q, c in zip(WAVE_NM, qabs, cabs):
            rows.append(f"{diameter:.1f}\t{wavelength:.1f}\t{q:.8e}\t{c:.8e}")
    spectrum_file = HERE / "ag_mie_spectrum.txt"
    spectrum_file.write_text("\n".join(rows) + "\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    for diameter, cabs in curves.items():
        ax.plot(WAVE_NM, cabs, label=f"{diameter:.1f} nm")
    ax.axvline(TARGET_NM, color="0.25", ls="--", lw=1, label=f"thioindigo root ({TARGET_NM:.0f} nm)")
    ax.set_xlim(WAVE_NM[0], WAVE_NM[-1])
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"Absorption cross section (nm$^2$)")
    ax.set_title(f"Mie absorption of a bare Ag sphere (Meep Ag, n = {N_ENV})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    spectrum_figure = HERE / "ag_mie_spectrum.png"
    fig.savefig(spectrum_figure, dpi=150)
    plt.close(fig)

    # Ag_visible is recorded only so the size difference is reproducible.
    # It is not the material for d35.
    size_rows = ["material\tdiameter_nm\tpeak_nm\tcabs_nm2"]
    peaks = {}
    materials = (
        ("Ag", mm.Ag, WAVE_NM),
        ("Ag_visible", mm.Ag_visible, WAVE_NM[WAVE_NM >= 400.0]),
    )
    for name, material, wave in materials:
        peak_nm = []
        for diameter in DIAMETERS_NM:
            _qabs, cabs = absorption(material, diameter, wave)
            i = interior_peak(cabs)
            peak_nm.append(wave[i])
            size_rows.append(f"{name}\t{diameter:.1f}\t{wave[i]:.2f}\t{cabs[i]:.6e}")
        peaks[name] = np.array(peak_nm)
    size_file = HERE / "ag_mie_size.txt"
    size_file.write_text("\n".join(size_rows) + "\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(DIAMETERS_NM, peaks["Ag"], label="Ag")
    ax.plot(DIAMETERS_NM, peaks["Ag_visible"], label="Ag_visible")
    ax.axhline(TARGET_NM, color="0.25", ls="--", lw=1, label=f"thioindigo root ({TARGET_NM:.0f} nm)")
    ax.set_xlabel("Diameter (nm)")
    ax.set_ylabel("Mie absorption peak (nm)")
    ax.set_title(f"Ag sphere in water (n = {N_ENV})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    size_figure = HERE / "ag_mie_size.png"
    fig.savefig(size_figure, dpi=150)
    print(f"Wrote {spectrum_file}")
    print(f"Wrote {spectrum_figure}")
    print(f"Wrote {size_file}")
    print(f"Wrote {size_figure}")


if __name__ == "__main__":
    main()
