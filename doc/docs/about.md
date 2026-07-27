# About PlasMol

**PlasMol** is a hybrid simulation framework for studying plasmon-molecule interactions. It combines the power of Meep (classical FDTD) with PySCF (quantum RT-TDDFT) in a tightly coupled, self-consistent loop.

## Current Version: v1.2.0 (July 2026)

Major improvements since v1.0.0:

- **Core-hole driver** (`core_hole`) for sudden SCH/DCH dynamics and MO hole tracking.
- Hybrid Fourier **parallel/perpendicular** polarization modes with vacuum \(E_{inc}\) deconvolution.
- Complete migration to modern **JSON input format** with validation and the `--describe` CLI flag.
- Support for **Lopata CAP broadening** (static & dynamic) and automatic tuning of LRC parameters / vacuum level.
- **Checkpoint / restart** capability for long quantum simulations.
- Multiple **custom drivers** (Fourier absorption spectra, MO comparison, NP/plasmon cross-section calculations).
- Improved multiprocessing safety, logging, and error messages.
- Many new parameters exposed via the single source-of-truth `param_defs` table.

## Releases

- **v1.2.0** (current) — Core-hole SCH/DCH, hybrid Fourier ∥/⊥ polarization, vacuum \(E_{inc}\) references; continues JSON schema, CAP, checkpointing, custom drivers.
- **v1.1.0** — JSON schema, CAP broadening, checkpointing, custom drivers, extensive validation.
- **v1.0.0** — Initial public release with block-style input files and "proof-of-concept" capabilities.

## Philosophy & Design Goals

PlasMol was created to enable research on plasmon-enhanced phenomena without having to glue together separate classical and quantum codes by hand. The bidirectional coupling (E-field → quantum propagation → induced dipole → back into FDTD) is handled automatically.

The codebase is intentionally **extensible**. Empty commented sections and clear extension points exist throughout so that researchers can add custom observables, sources, or post-processing with minimal friction.

## Citation

There is no formal journal publication yet. If you use PlasMol, please cite:

```bibtex
@software{PlasMol,
  author = {Brinton King Eldridge},
  title = {PlasMol: Simulating Plasmon-Molecule Interactions},
  url = {https://github.com/kombatEldridge/PlasMol},
  version = {1.2.0},
  year = {2026}
}
```

## Acknowledgments

- **Developer**: Brinton King Eldridge [[Google Scholar](https://scholar.google.com/citations?hl=en&user=8OgnrHMAAAAJ)]
- **Advisors**: Dr. Daniel Nascimento [[Google Scholar](https://scholar.google.com/citations?hl=en&user=VVPFNW8AAAAJ)], Dr. Yongmei Wang [[Google Scholar](https://scholar.google.com/citations?hl=en&user=TLvIKj0AAAAJ)]
- **Association**: University of Memphis

## Contact & Community

- Email: <bldrdge1@memphis.edu>
- GitHub: <https://github.com/kombatEldridge/PlasMol>
- Issues & Pull Requests are very welcome!

## License

GPL-3.0 (see LICENSE file).
