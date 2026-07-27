# Quantum driver

`driver: "quantum"` (or molecule-only input)

## Purpose

**Real-time TDDFT** of an isolated molecule driven by an analytic field (kick or pulse). No Meep.

## When to use

- Field-driven molecular dynamics of the density (not spectrum post-processing — use [fourier](fourier.md) for spectra)
- Checkpointed long pure-quantum runs

## Required JSON

| Section | Role |
| --------- | ------ |
| `settings` | `dt`, `t_end` |
| `molecule` | Geometry, basis, xc, propagator, **source** |
| `files` | Field CSVs, optional checkpoint |

Molecule source requires full kick/pulse fields (unlike Fourier minimal kick defaults).

## Typical outputs

- `field_e.csv` / `field_p.csv` — drive and induced dipole
- `output.png` (or `spectra_e_vs_p_filepath`) — E vs μ plot
- Optional `checkpoint.npz`

## Template

Copy and edit:

```bash
cp templates/template-quantum.json my_run.json
python -m plasmol.main -f my_run.json -vv -l run.log
```

Source (repo path): `templates/template-quantum.json`

## See also

- [All simulations](index.md)
- [Usage](../usage.md) — parameter reference
