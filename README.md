# PlasMol

PlasMol is a simulation framework that integrates finite-difference time-domain (FDTD) electromagnetic simulations using Meep with real-time time-dependent density functional theory (RT-TDDFT) calculations using PySCF. It enables the study of plasmon-molecule interactions, pure electromagnetic simulations, or standalone quantum simulations. The project supports various propagators for RT-TDDFT, checkpointing, data visualization, and spectrum analysis via Fourier transforms.

## Key Features

- **Hybrid Simulations (PlasMol Mode)**: Combines Meep FDTD for plasmonic structures with RT-TDDFT for molecular responses, allowing self-consistent coupling between electromagnetic fields and molecular polarization.
- **Standalone Modes**:
  - Quantum-only: RT-TDDFT simulations with propagators like step, Magnus2, or Runge-Kutta 4 (RK4).
  - Meep-only: Pure FDTD simulations for electromagnetic wave propagation.
- **Electric Field Sources**: Supports continuous, Gaussian, chirped, and pulse sources with customizable parameters (e.g., wavelength, frequency, peak time).
- **Propagators for RT-TDDFT**: Options include 'step', 'magnus2' (with predictor-corrector convergence), and 'rk4'.
- **Checkpointing and Restarting**: Save and resume simulations using `.npz` checkpoint files.
- **Data Output and Visualization**: Exports electric and polarization fields to CSV; generates plots of fields over time and absorption spectra.
- **Fourier Transform Analysis**: Computes absorption spectra from polarization data in multi-directional simulations.
- **Material Support**: Predefined materials like Au (gold) and Ag (silver) for plasmonic objects.
- **Symmetries and PML Boundaries**: Configurable symmetries and perfectly matched layers (PML) for efficient simulations.
- **GIF Generation**: Optional output of simulation frames as HDF5 and compilation into GIFs for visualization.

## Technologies Used

- **Python**: Version 3.x (core language).
- **Meep**: For FDTD electromagnetic simulations.
- **PySCF**: For quantum chemistry calculations, including RT-TDDFT.
- **NumPy & SciPy**: Matrix operations, linear algebra, and exponentials.
- **Pandas**: Data handling for CSV I/O.
- **Matplotlib**: Plotting fields and spectra.
- **Other Libraries**: PIL (for GIF creation), threading (for parallel directional simulations).
- **Constants and Units**: Handles conversions between atomic units (au), femtoseconds (fs), nanometers (nm), etc.

Note: The project assumes a pre-configured environment with these libraries (no internet access for installations during runtime).

## Prerequisites

- Python 3.x or higher.
- Required Python packages (install via pip if needed):
  ```
  pip install meep pyscf numpy scipy pandas matplotlib pillow
  ```
- For Meep materials: Ensure Meep is compiled with material libraries (e.g., for Au and Ag).
- Sufficient computational resources: Simulations can be memory- and CPU-intensive, especially in hybrid mode.
- Optional: Graphviz or similar for visualizing project structure (not required for running).

Potential error: If Meep or PySCF is not installed correctly, you may encounter import errors. Ensure your environment matches the listed versions.

## Detailed Step-by-Step Installation Guide

1. **Clone the Repository**:
   ```
   git clone https://github.com/yourusername/PlasMol.git
   cd PlasMol
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   - Install core libraries:
     ```
     pip install meep pyscf numpy scipy pandas matplotlib pillow
     ```
   - Note: Meep may require additional system dependencies (e.g., HDF5, MPI). Refer to [Meep documentation](https://meep.readthedocs.io/en/latest/Installation/) for platform-specific instructions.
   - PySCF may need BLAS/LAPACK for performance; install via `pip install pyscf[blas]`.

4. **Verify Installation**:
   - Run a simple test:
     ```
     python -c "import meep; import pyscf; print('Installation successful!')"
     ```
   - If errors occur (e.g., missing modules), reinstall the affected package.

5. **Environment Variables**:
   - Set `PYTHONPATH` to include the project root if running scripts directly:
     ```
     export PYTHONPATH=$PYTHONPATH:/path/to/PlasMol
     ```
   - For logging: Optionally set `LOG_LEVEL` (e.g., `export LOG_LEVEL=DEBUG`) to control verbosity.

6. **Common Issues**:
   - **Error: "No module named 'meep'"**: Ensure Meep is installed in your active environment.
   - **Permission Issues**: Run with `sudo` if needed, but prefer virtual environments.
   - **Memory Errors**: Reduce simulation resolution or time steps for large systems.

## Usage Examples

PlasMol is run via the command line, providing an input file. Use `-v` for verbosity, `-r` for restart (deletes output files), and `-l` for logging to a file.

### Basic Command
```
python src/main.py -f path/to/input_file.inp
```

### Example 1: Hybrid PlasMol Simulation
Input file (`input.inp`):
```
start settings
dt 0.01
t_end 100.0
eField_path eField.csv
end settings

start meep
start simulation
cellLength 10
pmlThickness 1.0
resolution 20
surroundingMaterialIndex 1.0
eFieldCutOff 1e-6
end simulation
start molecule
center 0 0 0
end molecule
start source
sourceType gaussian
sourceCenter 0 0 0
sourceSize 1 1 1
frequency 0.1
width 0.2
component z
end source
start object
material Au
radius 0.5
center 0 0 0
end object
end meep

start quantum
start rttddft
basis sto-3g
xc b3lyp
charge 0
spin 0
propagator magnus2
maxiter 10
pc_convergence 1e-8
units angstrom
end rttddft
start geometry
H 0 0 0
H 0 1 0
end geometry
end quantum
```
Run: `python src/main.py -f input.inp -v -r`

Output: Generates `eField.csv`, `pField.csv`, plots, and optionally a GIF.

### Example 2: Quantum-Only (RT-TDDFT)
Omit the `meep` section in the input file. Use `shape pulse` or `kick` in the quantum source block.

### Example 3: Meep-Only
Omit the `quantum` section.

### CLI Options
- `-f/--input`: Required path to input file.
- `-l/--log`: Log file name.
- `-v/--verbose`: Increase verbosity (e.g., `-vv` for debug).
- `-r/--restart`: Delete existing output files and restart.

### Error Messages
- "No 'dt' value given in settings file": Ensure `dt` and `t_end` are specified.
- "Simulation failed: Unsupported propagator": Choose 'step', 'rk4', or 'magnus2'.
- "Checkpoint file not found": Verify `chkfile_path` exists for restarts.

## Configuration (if applicable)

Configuration is done via the input file with sections: `settings`, `meep`, `quantum`.

- **Settings Block**: `dt` (time step in au), `t_end` (end time in au), `eField_path` (output CSV).
- **Meep Block**: Sub-blocks for `simulation` (cell size, PML, resolution), `source` (type, params), `object` (material, shape), `molecule` (position), `hdf5` (for GIF output).
- **Quantum Block**: `rttddft` (basis, xc, charge, spin, propagator), `geometry` (atomic coordinates), optional `source` for field shape.
- **Files**: `chkfile` (path and frequency for checkpoints), output paths for fields and spectra.

For multi-directional transforms, set `transform true` in quantum section.

## Project Structure Explanation

```
PlasMol/
├── src/
│   ├── __init__.py          # Package init with version/author
│   ├── constants.py         # Physical constants and unit conversions
│   ├── drivers/             # Simulation drivers
│   │   ├── __init__.py
│   │   ├── meep.py          # Meep-only driver
│   │   ├── plasmol.py       # Hybrid PlasMol driver
│   │   └── rttddft.py       # Quantum-only driver with threading for directions
│   ├── input/               # Input parsing and params
│   │   ├── __init__.py
│   │   ├── cli.py           # CLI argument parser
│   │   ├── params.py        # PARAMS class for merging inputs
│   │   └── parser.py        # Parses input file sections
│   ├── main.py              # Entry point: parses args, runs simulation
│   ├── meep/                # Meep-specific modules
│   │   ├── __init__.py
│   │   ├── simulation.py    # Meep simulation class with custom sources
│   │   └── sources.py       # Source classes (Continuous, Gaussian, etc.)
│   ├── quantum/             # Quantum modules
│   │   ├── __init__.py
│   │   ├── chkfile.py       # Checkpoint save/load
│   │   ├── electric_field.py# Electric field generation
│   │   ├── molecule.py      # Molecule class with PySCF integration
│   │   ├── propagation.py   # Propagation wrapper
│   │   └── propagators/     # RT-TDDFT propagators
│   │       ├── __init__.py
│   │       ├── magnus2.py   # Magnus2 propagator
│   │       ├── rk4.py       # RK4 propagator
│   │       └── step.py      # Step propagator
│   └── utils/               # Utilities
│       ├── __init__.py
│       ├── csv.py           # CSV init/update/read
│       ├── fourier.py       # Fourier transform for spectra
│       ├── gif.py           # GIF creation from frames
│       ├── logging.py       # Custom logger for stdout
│       └── plotting.py      # Field plotting
└── README.md                # This file
```

## Contributing Guidelines

(Not explicitly found in the codebase. If contributing, follow standard practices: fork the repo, create a feature branch, submit a pull request with clear descriptions. Ensure code style matches (e.g., PEP8) and add tests if possible.)

## License Information

(Not found in the codebase. Assume open-source under MIT License unless specified otherwise. Check for a LICENSE file in the repository.)