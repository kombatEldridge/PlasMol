# Installation Guide

## Prerequisites

- Python ≥ 3.8
- Git
- Meep (via conda is strongly recommended)
- Optional but recommended: `conda` or `venv` for environment isolation

## Step-by-Step Installation

### 1. Install Meep

```bash
conda create -n plasmol python=3.12
conda activate plasmol
conda install -c conda-forge pymeep
```

Verify:
```bash
python -c "import meep as mp; print(mp.__version__)"
```

### 2. Clone PlasMol

```bash
git clone https://github.com/kombatEldridge/PlasMol.git
cd PlasMol
```

### 3. Install Python dependencies & PlasMol

```bash
pip install -e .
```

This installs:
- pyscf, numpy, scipy, pandas, matplotlib
- rich (for `--describe` table)
- All other runtime requirements

For development (linting, testing, docs):
```bash
pip install -r requirements-dev.txt
```

### 4. Verify Installation

```bash
python -m plasmol.main --help
python -m plasmol.main --describe | head -30
```

You should see the CLI help and a beautiful parameter table.
