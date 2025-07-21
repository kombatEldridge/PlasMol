# API Reference

This page documents key classes and functions. Based on codebase structure.

## quantum/ Module
### `MOLECULE(params)`
- Initializes molecule with PySCF, handles SCF, Fock matrix, dipole calculations.
- Methods: `get_F_orth()`, `calculate_mu()`, `get_homo_lumo_coefficient()`.

### `ELECTRICFIELD(times, params)`
- Builds electric fields (pulse or kick shapes).

### Propagators (quantum/propagators/)
- `step.propagate(params, molecule, exc)`: Step method.
- `magnus2.propagate()`: Magnus2 with predictor-corrector.
- `rk4.propagate()`: Runge-Kutta 4.

## meep Module
### `Simulation(params, molecule=None)`
- Runs Meep simulation, handles sources, PML, symmetries.
- Methods: `chirpx(t)`, `getElectricField(sim)`, `callBohr(sim, eField)`.

### Sources
- `ContinuousSource(...)`, `GaussianSource(...)`, etc.

## Drivers
- `drivers.meep.run(params)`: Runs Meep simulation.
- `drivers.rttddft.run(params)`: Runs RT-TDDFT, supports multi-threading for transforms.
- `drivers.plasmol.run(params)`: Hybrid run.

## Utils
- `utils.csv.initCSV(filename, comment)`, `updateCSV(...)`.
- `utils.plotting.show_eField_pField(eFieldFile, pFieldFile)`.
- `utils.fourier.transform(...)`: Fourier transform and absorption spectrum.

[TODO: Add full parameter lists, return types, or use Sphinx for auto-gen docs.]
