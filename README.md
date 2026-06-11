# PlasMol: Simulating Plasmon-Molecule Interactions

![PlasMol Logo](doc/docs/PlasMol.png)

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://github.com/kombatEldridge/PlasMol/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://www.python.org/downloads/)
[![GitHub Issues](https://img.shields.io/github/issues/kombatEldridge/PlasMol.svg)](https://github.com/kombatEldridge/PlasMol/issues)
[![GitHub Stars](https://img.shields.io/github/stars/kombatEldridge/PlasMol.svg?style=social)](https://github.com/kombatEldridge/PlasMol/stargazers)

**Current version: v1.1.0** (June 2026)

Read the full documentation: [https://kombateldridge.github.io/PlasMol/](https://kombateldridge.github.io/PlasMol/)

PlasMol is an open-source Python package for simulating plasmon-molecule interactions. It tightly couples classical Finite-Difference Time-Domain (FDTD) electromagnetics via [Meep](https://meep.readthedocs.io/) with quantum Real-Time Time-Dependent Density Functional Theory (RT-TDDFT) via [PySCF](https://pyscf.org/). The bidirectional coupling—classical field drives quantum propagation, induced dipole feeds back into FDTD—is handled automatically, enabling studies of nanoparticle-molecule systems such as plasmon-enhanced spectroscopy and SERS.

## Simulation Modes

PlasMol supports three primary modes (additional workflows are available through [custom drivers](doc/docs/custom_drivers.md)):

1. **Classical FDTD** — Spherical nanoparticle simulations (e.g., Au/Ag spheres) with custom sources, symmetries, PML, optional field imaging/GIFs, and absorption/scattering cross-section calculations.
2. **Quantum RT-TDDFT** — Isolated molecule simulations with absorption spectra (Fourier workflow), MO energy comparisons, [Lopata-style](https://pubs.acs.org/doi/abs/10.1021/ct400569s) CAP broadening, and checkpoint/restart support.
3. **Full Hybrid PlasMol** — Self-consistent NP + molecule simulations where the classical electric field drives quantum propagation and the induced molecular dipole is fed back as a point source in FDTD.

The driver is inferred from your JSON input (`molecule` only → quantum, `plasmon` only → classical, both → plasmol), or you can set `"driver"` explicitly in `settings`.

## Key Features (v1.1.0)

- **JSON input format** with validation and the `--describe` CLI flag for exploring every supported parameter.
- **Custom drivers** for Fourier absorption spectra, MO comparison, NP/plasmon cross-sections, and user-defined workflows.
- **Lopata CAP broadening** (static and dynamic) with automatic tuning of LRC parameters and vacuum level.
- **Checkpoint/restart** for long quantum simulations.
- **Propagators**: Magnus2 (default), RK4, or step for RT-TDDFT.
- **Outputs**: CSVs for fields and dipoles, PNG/GIF field evolution, absorption spectra, MO diagrams, and checkpoints.

## Quick Start

### Installation

PlasMol requires Python 3.9+, Meep, and PySCF. Installing Meep via conda is strongly recommended. See the [Installation Guide](doc/docs/installation.md) for full details.

```bash
conda create -n plasmol python=3.12
conda activate plasmol
conda install -c conda-forge pymeep

# Verify Meep
python -c "import meep as mp; print(mp.__version__)"

git clone https://github.com/kombatEldridge/PlasMol.git
cd PlasMol
pip install -e .
```

For development (linting, testing, docs):

```bash
pip install -r requirements-dev.txt
```

Verify the install:

```bash
python -m plasmol.main --help
python -m plasmol.main --describe | head -30
```

### Basic Usage

PlasMol is controlled by a single **JSON input file**. Example templates live in `templates/`.

```bash
python -m plasmol.main -f templates/template-plasmol-MAIN.json -vv -l plasmol.log
```

**Common CLI options:**

| Flag | Description |
|------|-------------|
| `-f`, `--input` | Path to JSON input file |
| `-l`, `--log` | Log file path (default: terminal) |
| `-v`, `-vv` | Verbosity (`-v` = INFO, `-vv` = DEBUG) |
| `--describe` | Print all supported parameters with types, defaults, and units |

Explore the full parameter schema:

```bash
python -m plasmol.main --describe
```

### Example: Hybrid PlasMol Input

```json
{
  "settings": {
    "dt": 0.1,
    "t_end": 400
  },
  "plasmon": {
    "simulation": {
      "cell_length": 0.1,
      "pml_thickness": 0.01,
      "surrounding_material_index": 1.33
    },
    "source": {
      "type": "continuous",
      "center": [-0.04, 0, 0],
      "size": [0, 0.1, 0.1],
      "component": "z",
      "additional_parameters": { "frequency": 5.0 }
    },
    "nanoparticle": {
      "material": "Au_JC_visible",
      "radius": 0.03,
      "center": [0, 0, 0]
    },
    "molecule": {
      "position": [0, 0, 0],
      "back_propagation": true
    }
  },
  "molecule": {
    "geometry": [
      {"atom": "O", "coord": [0.0, 0.0, -0.1302]},
      {"atom": "H", "coord": [1.4891, 0.0, 1.0332]},
      {"atom": "H", "coord": [-1.4891, 0.0, 1.0332]}
    ],
    "geometry_units": "bohr",
    "basis": "6-31g",
    "xc": "pbe0",
    "propagator": { "type": "magnus2" }
  },
  "files": {
    "field_e_filepath": "field_e.csv",
    "field_p_filepath": "field_p.csv"
  }
}
```

For step-by-step walkthroughs (classical, quantum, absorption spectra, hybrid, MO comparison, cross-sections), see [Tutorials](doc/docs/tutorials.md).

## Documentation

| Guide | Description |
|-------|-------------|
| [Installation](doc/docs/installation.md) | Environment setup and verification |
| [Usage](doc/docs/usage.md) | JSON schema, all parameters, validation rules |
| [Tutorials](doc/docs/tutorials.md) | Hands-on examples for every major workflow |
| [Theory & Methodology](doc/docs/methodology.md) | Hybrid FDTD–RT-TDDFT coupling loop |
| [Custom Drivers](doc/docs/custom_drivers.md) | Building and registering custom workflows |
| [Contributing](doc/docs/contributing.md) | Extension points, code style, PR process |
| [About](doc/docs/about.md) | Version history, citation, contact |

## Contributing

Contributions are welcome—bug fixes, new features, documentation, example inputs, and especially **test cases**. The project most needs help with test suites, additional nanoparticle shapes, new electric field sources, and SERS-related custom drivers. See the [Contributing Guide](doc/docs/contributing.md) and open an issue or PR on [GitHub](https://github.com/kombatEldridge/PlasMol).

## Citation

There is no formal journal publication yet. If you use PlasMol in your work, please cite:

```bibtex
@software{PlasMol,
  author = {Brinton King Eldridge},
  title = {PlasMol: Simulating Plasmon-Molecule Interactions},
  url = {https://github.com/kombatEldridge/PlasMol},
  version = {1.1.0},
  year = {2026}
}
```

## License

[GPL-3.0 License](https://github.com/kombatEldridge/PlasMol/blob/main/LICENSE).

## Acknowledgments

Built on [Meep](https://meep.readthedocs.io/), [PySCF](https://pyscf.org/), NumPy, SciPy, Matplotlib, Pandas, and Rich.

- **Developer**: [Brinton King Eldridge](https://github.com/kombatEldridge) ([Google Scholar](https://scholar.google.com/citations?hl=en&user=8OgnrHMAAAAJ))
- **Advisors**: Dr. Daniel Nascimento ([Google Scholar](https://scholar.google.com/citations?hl=en&user=VVPFNW8AAAAJ)), Dr. Yongmei Wang ([Google Scholar](https://scholar.google.com/citations?hl=en&user=TLvIKj0AAAAJ))
- **Association**: University of Memphis

## Contact

- Email: [bldrdge1@memphis.edu](mailto:bldrdge1@memphis.edu)
- GitHub: [kombatEldridge](https://github.com/kombatEldridge)
- LinkedIn: [Brinton Eldridge](https://www.linkedin.com/in/brinton-eldridge/)