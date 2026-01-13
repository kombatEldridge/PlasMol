# Installation Guide

PlasMol requires Python 3.8+ and several scientific libraries. Follow these steps to set up the environment.

## Prerequisites

- Python 3.8 or higher.
- Git (for cloning the repository).
- Optional: Conda or virtualenv for isolated environments.

## Step 0: Install MEEP

Please visit the [installation page](https://meep.readthedocs.io/en/master/Installation/) for MEEP to install it for use in PlasMol.

## Step 1: Clone the Repository

```bash
git clone https://github.com/kombatEldridge/PlasMol.git  # [TODO: Replace with actual repo URL]
cd PlasMol
```

## Step 2: Create a Virtual Environment

If a virtual/conda environment for MEEP was not used for install, you can set up one here.

Using virtualenv:

```bash
python -m venv plasmol
source plasmol/bin/activate  # On Windows: env\Scripts\activate
```

Or with Conda:

```bash
conda create -n plasmol python=3.12
conda activate plasmol
```

## Step 3: Install Dependencies

PlasMol uses:

- Meep for electromagnetics (should already be installed).
- PySCF for quantum calculations.
- NumPy, SciPy, Pandas, Matplotlib for data handling and plotting.

Install via pip:

```bash
pip install pyscf numpy scipy pandas matplotlib logging argparse
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
python -m plasmol.main --help
```

If you see the CLI help message, it's working.
