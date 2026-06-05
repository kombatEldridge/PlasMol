# Contributing Guide

Thank you for your interest in PlasMol! Contributions of all kinds are welcome — bug fixes, new features, documentation improvements, example inputs, and especially **test cases**.

## How to Contribute

1. **Fork** the repository on GitHub.
2. **Create a feature branch** (`git checkout -b my-new-feature`).
3. **Make your changes** (see extension points below).
4. **Test** your changes (run existing tutorials + any new tests).
5. **Submit a Pull Request** with a clear description of what was changed and why.

## Most Needed Contributions (as of v1.1.0)

- **Test suite** — Currently the biggest gap. Even simple regression tests for the quantum propagators, JSON validation, and end-to-end hybrid runs would be extremely valuable.
- **More nanoparticle shapes** — Support for rods, shells, dimers, etc. in the classical module.
- **Additional electric field shapes** — Both classical (chirped pulses, custom waveforms) and quantum.
- **New custom drivers** — Especially for SERS enhancement mapping, hot-electron dynamics, or plexciton spectra.
- **Documentation & examples** — More tutorial JSON files, Jupyter notebooks, and real-world research examples.
- **Performance / GPU** — Exploration of PySCF + Meep GPU paths or surrogate models for the quantum step.

## Adding New Features — Recommended Workflow

### 1. New JSON Parameter

- Add a row to `plasmol/utils/input/struct.py:param_defs`.
- Add validation logic in `PARAMS._validate_all()`.
- Handle the attribute in `PARAMS._attribute_formation()` if it needs special processing.
- Document it in `usage.md`.

### 2. New Propagator

- Implement `propagate_xxx(**params, molecule, exc)` in `quantum/propagators/`.
- Add it to the `propagator_map` in `params.py`.
- Update validation and `--describe` table.
- Add a short description in `api-reference.md` and `methodology.md`.

### 3. New Custom Driver

- Create `plasmol/drivers/custom_drivers/my_driver.py` with a top-level `def run(params):`.
- Register it in `plasmol/drivers/__init__.py:get_driver()`.
- Document usage in `tutorials.md` and `usage.md`.

### 4. New Observable / Measurable

- Add a method to `quantum/molecule.py` (e.g. `calculate_custom_observable()`).
- Call it inside `quantum/propagation.py` after the dipole calculation.
- Write results using `utils/csv.py:update_csv`.
- (Optional) Add visualization in `utils/plotting.py`.

### 5. New Classical Source or Geometry

- Extend `classical/sources.py` (MEEPSOURCE class).
- For geometry, extend the handling inside `SIMULATION.__init__`.
- Update JSON schema and validation.

## Code Style & Quality

- Follow PEP 8.
- Use type hints where reasonable.
- Keep functions focused and well-documented (Google or NumPy style docstrings).
- Run `black` and `flake8` (or the dev requirements) before submitting a PR.
- All new functionality should be accompanied by at least a minimal test or a working tutorial JSON.

## Questions?

Open an issue on GitHub or email the maintainer (bldrdge1@memphis.edu). We are happy to discuss design decisions before you invest a lot of time.

**Thank you for helping make PlasMol better for the plasmonics and quantum chemistry communities!**