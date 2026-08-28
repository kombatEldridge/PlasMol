# Absorption driver

`driver: "absorption"`

## Purpose

Compute **linear absorption spectra** by Fourier-transforming the time-domain response:

- **Quantum-only:** three δ-kicks → Im[μ(ω)]
- **Hybrid:** Meep + vacuum E_inc reference → Im[μ/E_inc]
- **Polarization:** `full` \| `parallel` \| `perpendicular`

## When to use

- Molecular absorption spectra (recommended path)
- Orientation-resolved hybrid spectra near an NP

## Required JSON

| Section | Role |
| --------- | ------ |
| `settings.driver` | `"absorption"` |
| `molecule` | Geometry + electronic structure |
| `molecule.source` | May be just `{"type": "kick"}` — other kick fields default under the absorption driver |
| `additional_parameters.absorption` | At least `spectrum_filepath` (or files spectra path) |
| `plasmon` | Optional; required for hybrid / ∥ / ⊥ modes |

## Key `absorption` keys

`gamma`, `tau`, `min_ev`, `max_ev`, `spectrum_filepath`, `npz_filepath`, `polarization`, `perp_component`, `field_e_ref_filepath`, `reference_only`

```{warning}
Hybrid absorption (any run with a `plasmon` section, including `parallel` / `perpendicular`) **cannot** be checkpointed. Molecule-only three-kick absorption can; see [Checkpointing](../checkpointing.md).
```

## Typical outputs

- Spectrum PNG + CSV
- Optional `absorption.npz`
- Per-direction `x_dir/` / `y_dir/` / `z_dir/` CSVs (`polarization: full` only)
- Job-root `field_e.csv` / `field_p.csv` / `field_e_ref.csv` for `parallel` / `perpendicular`

## Theory

- [Fourier Spectra methodology](../methodology/fourier.md)

## Template

Copy and edit:

```bash
cp templates/template-absorption.json my_run.json
python -m plasmol.main -f my_run.json -vv -l run.log
```

Source (repo path): `templates/template-absorption.json`

## See also

- [All simulations](index.md)
- [Usage](../usage.md) — parameter reference
- [Checkpointing](../checkpointing.md) — molecule-only absorption only
