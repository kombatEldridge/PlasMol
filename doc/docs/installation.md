# Installation Guide

PlasMol requires Python 3.8+ and several scientific libraries. Follow these steps to set up the environment.

## Prerequisites
- Python 3.8 or higher.
- Git (for cloning the repository).
- Optional: Conda or virtualenv for isolated environments.

## Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/PlasMol.git  # [TODO: Replace with actual repo URL]
cd PlasMol
```

## Step 2: Create a Virtual Environment
Using virtualenv:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

Or with Conda:
```bash
conda create -n plasmol python=3.12
conda activate plasmol
```

## Step 3: Install Dependencies
PlasMol uses:
- PySCF for quantum calculations.
- Meep for electromagnetics.
- NumPy, SciPy, Pandas, Matplotlib for data handling and plotting.

Install via pip:
```bash
pip install pyscf meep numpy scipy pandas matplotlib logging argparse
# Optional for advanced features: torch (ML), biopython (if bio-related), etc.
# [TODO: Add any other dependencies from your requirements.txt if available]
```

## Step 4: Install PlasMol as a Package (Optional)
To make it importable system-wide:
```bash
pip install -e .
```
This assumes a `setup.py` or `pyproject.toml` in the root (e.g., for editable install).

## Step 5: Verify Installation
Run a test:
```bash
python -m src.main --help
```
If you see the CLI help message, it's working.

## Common Issues
- **Meep Installation**: On some systems, Meep requires additional setup (e.g., `conda install -c conda-forge meep` for binaries).
- **PySCF Errors**: Ensure OpenBLAS or MKL is installed for performance.
- **Import Errors**: Run as a module (`python -m src.main`) from the project root to resolve relative imports.

[TODO: Add platform-specific notes, e.g., for macOS/Linux/Windows.]