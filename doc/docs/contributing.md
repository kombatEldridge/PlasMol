# Contributing Guide

Welcome contributors! Follow these steps to contribute.

## Development Setup
- Fork the repo.
- Install dev deps: `pip install black pre-commit pytest`.
- Run `pre-commit install` for hooks.

## Code Organization
- `drivers/`: Simulation entry points (meep.py, etc.).
- `quantum/`: Molecule, propagation, propagators/.
- `meep/`: Simulation and sources.
- `input/`: Parsing and params.
- `utils/`: CSV, logging, plotting, etc.
- Use relative imports (e.g., `from ..quantum import MOLECULE`).

## Adding Features
- Create branch: `git checkout -b feature/new-prop`.
- Write tests in `tests/`.
- Update docs in `docs/`.
- Submit PR with changelog entry.

## Coding Standards
- PEP 8 with Black formatter.
- Type hints and docstrings.
- Tests: `pytest`.

[TODO: Add issue templates or specific guidelines.]
