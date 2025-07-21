# Usage Guide

PlasMol is run via the command-line interface (CLI) from `src/main.py`. It supports three simulation types: Meep (electromagnetics), Quantum (RT-TDDFT), and PlasMol (hybrid).

## CLI Arguments
```bash
python -m src.main -f <input_file> [options]
```
- `-f, --input <file>`: Path to the input file (required).
- `-l, --log <file>`: Log file name.
- `-v, --verbose`: Increase verbosity (up to -vv` for debug).
- `-r, --restart`: Restart simulation (deletes previous outputs).

## Input File Format
Input files are divided into sections like `start meep` / `end meep`, `start quantum` / `end quantum`, and `start settings` / `end settings`. See examples in `templates/`.

Example snippet:
```
start settings
dt = 0.01
t_end = 100.0
eField_path = eField.csv
end settings

start meep
[simulation parameters like cellLength, resolution]
end meep

start quantum
[basis, xc, geometry, etc.]
end quantum
```

## Running Simulations
- **Pure Meep**:
  ```bash
  python -m src.main -f templates/template-meep.in -v
  ```
  Outputs electric fields, HDF5 images, GIF (if enabled).

- **Pure RT-TDDFT**:
  ```bash
  python -m src.main -f templates/template-quantum.in -v
  ```
  polarization fields, spectra, checkpoints.

- **Hybrid PlasMol**:
  ```bash
  python -m src.main -f templates/template-plasmol.in -vv
  ```
  Combines both, with molecule-field interactions.

## Output Files
- CSV: eField.csv, pField.csv for fields.
- Plots: output.png (eV_spectrum.png for spectra).
- Checkpoint: .npz for resuming.
- GIF: Simulation visualizations if HDF5 enabled.

## Advanced Options
- Propagators: Set in quantum input (step, magnus2, rk4).
- Sources: Continuous, Gaussian, chirped, pulse in meep section.
- Restart: Use `-r` to clear old files and resume from checkpoint.

[TODO: Add example input files or screenshots of output.]
