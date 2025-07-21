# Welcome to PlasMol Documentation

PlasMol is a Python-based simulation framework for combining plasmonics (via Meep) with real-time time-dependent density functional theory (RT-TDDFT) simulations. It enables hybrid simulations of electromagnetic fields interacting with quantum molecular systems, useful for applications in plasmon-enhanced spectroscopy, nanophotonics, and quantum chemistry.

## Key Features
- **Simulation Modes**: Pure Meep (electromagnetics), pure RT-TDDFT (quantum), or hybrid PlasMol.
- **Quantum Components**: Molecule handling with PySCF, electric field interactions, density matrix propagation (step, magnus2, rk4 methods), checkpointing, and HOMO-LUMO analysis.
- **Meep Integration**: Custom sources (continuous, Gaussian, chirped, pulse), symmetries, PML boundaries, and HDF5 output for visualizations.
- **Utilities**: CSV handling for fields, Fourier transforms for spectra, plotting (e.g., electric vs. polarization fields), GIF generation, and logging.
- **Input Handling**: CLI arguments, input file parsing (Meep and Quantum sections), parameter merging.

## Quick Start
1. Install dependencies: `pip install pyscf meep numpy matplotlib pandas scipy`.
2. Run a simulation: `python -m src.main -f path/to/input.in -v`.

For more, see [Installation](installation.md) and [Usage](usage.md).

## Project Structure
- `src/`: Core codebase with modular directories (drivers/, quantum/, meep/, input/, utils/).
- Organized for scalability, with relative imports and package `__init__.py` files.

[TODO: Add project logo or screenshot of a simulation output here.]