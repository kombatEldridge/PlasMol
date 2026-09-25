# Linear-response TDDFT for neutral trans-thioindigo.
# Geometry is the PBE0/6-311G* structure in opt.xyz.
# The excitation is the campaign functional, not the geometry functional:
# LC-ωPBE, μ = 0.34272, Cartesian 6-311G*.
#
#   conda activate meep
#   python jobs/final/Step_1/Molecules/transthioindigo/lr_tddft.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft, gto

HERE = Path(__file__).resolve().parent
GEOM = HERE / "opt.xyz"
NSTATES = 30
HA_TO_EV = 27.211386
# Same half-width on every root. Not the real-time Fourier gamma.
HWHM_EV = 0.15
WINDOW_EV = (1.5, 6.0)
# Benzene solution maximum, not the gas-phase root this calculation produces.
EXPT_EV = 1239.84193 / 543.0


def load_geometry(path):
    """Atom list in the same window PlasMol uses: coordinates start on line 3."""
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    natom = None
    for line in lines:
        parts = line.split()
        if len(parts) == 1 and parts[0].isdigit():
            natom = int(parts[0])
    if natom is None:
        raise ValueError(f"No atom count in {path}")
    atoms = []
    for line in lines[2:2 + natom]:
        symbol, x, y, z = line.split()[:4]
        atoms.append((symbol, (float(x), float(y), float(z))))
    if len(atoms) != natom:
        raise ValueError(f"Expected {natom} atoms in {path}, found {len(atoms)}")
    return atoms


def uniform_spectrum(energies_ev, strengths, grid, hwhm):
    """Sum of identical Lorentzians. Each line has area equal to its oscillator strength."""
    spectrum = np.zeros_like(grid, dtype=float)
    for energy, strength in zip(energies_ev, strengths):
        if strength == 0.0:
            continue
        spectrum += strength * (hwhm / np.pi) / ((grid - energy) ** 2 + hwhm ** 2)
    return spectrum


def main():
    mol = gto.M(
        atom=load_geometry(GEOM),
        basis="6-311G*",
        cart=True,
        charge=0,
        spin=0,
        unit="Angstrom",
        verbose=4,
    )
    mf = dft.RKS(mol)
    mf.xc = "HYB_GGA_XC_LC_WPBE"
    mf.omega = 0.34272
    mf.max_memory = 10000
    mf.kernel()

    td = mf.TDDFT()
    td.nstates = NSTATES
    td.max_memory = 10000
    td.kernel()
    strengths = np.asarray(td.oscillator_strength(gauge="length"), dtype=float)
    energies_ev = np.asarray(td.e, dtype=float) * HA_TO_EV
    wavelengths_nm = np.where(energies_ev > 0.0, 1239.84193 / energies_ev, np.nan)

    rows = ["state\tenergy_eV\twavelength_nm\toscillator_strength"]
    print(f"{'state':>5} {'E (eV)':>10} {'λ (nm)':>10} {'f':>12}")
    for i, (energy, wavelength, strength) in enumerate(
        zip(energies_ev, wavelengths_nm, strengths), start=1
    ):
        print(f"{i:5d} {energy:10.4f} {wavelength:10.1f} {strength:12.6f}")
        rows.append(f"{i}\t{energy:.6f}\t{wavelength:.4f}\t{strength:.8e}")
    table = HERE / "lr_thioindigo_states.txt"
    table.write_text("\n".join(rows) + "\n")

    grid = np.linspace(WINDOW_EV[0], WINDOW_EV[1], 2000)
    spectrum = uniform_spectrum(energies_ev, strengths, grid, HWHM_EV)
    wavelengths = 1239.84193 / grid
    spectrum_rows = ["energy_eV\twavelength_nm\tintensity"]
    for energy, wavelength, intensity in zip(grid, wavelengths, spectrum):
        spectrum_rows.append(f"{energy:.6f}\t{wavelength:.4f}\t{intensity:.8e}")
    spectrum_file = HERE / "lr_thioindigo_spectrum.txt"
    spectrum_file.write_text(
        f"# uniform Lorentzian, HWHM {HWHM_EV} eV, area of each line = oscillator strength\n"
        + "\n".join(spectrum_rows)
        + "\n"
    )

    in_window = (energies_ev >= WINDOW_EV[0]) & (energies_ev <= WINDOW_EV[1])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, spectrum, color="C0", label=f"uniform Lorentzian, HWHM {HWHM_EV} eV")
    ax.vlines(
        energies_ev[in_window], 0.0, strengths[in_window],
        color="0.55", lw=1, label="roots",
    )
    ax.axvline(EXPT_EV, color="0.35", ls="--", lw=1, label="benzene λmax (543 nm)")
    ax.set_xlim(*WINDOW_EV)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Intensity (oscillator strength per eV)")
    ax.set_title("trans-thioindigo LR-TDDFT, LC-ωPBE, 6-311G*")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figure = HERE / "lr_thioindigo_spectrum.png"
    fig.savefig(figure, dpi=150)
    print(f"Wrote {table}")
    print(f"Wrote {spectrum_file}")
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
