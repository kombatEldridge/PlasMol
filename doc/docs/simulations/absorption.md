# Absorption driver

`driver: "absorption"`

## Purpose

Compute **linear absorption spectra** by Fourier-transforming the time-domain response:

- **Quantum-only:** three δ-kicks → Im[μ(ω)]
- **Hybrid:** Meep + vacuum E_inc reference → Im[μ/E_inc]
- **Polarization:** `full` (no NP) \| `parallel` \| `perpendicular` \| `single`

## When to use

- Molecular absorption spectra (recommended path)
- Orientation-resolved hybrid spectra near an NP

## Required JSON

| Section | Role |
| --------- | ------ |
| `settings.driver` | `"absorption"` or `{"name": "absorption", ...}` |
| `molecule` | Geometry + electronic structure; optional `core_hole` for sudden SCH/DCH spectra |
| `molecule.source` | May be just `{"type": "kick"}` — other kick fields default under the absorption driver |
| Driver keys | At least `spectrum_filepath` (or files spectra path) |
| `plasmon` | Optional; required for hybrid / ∥ / ⊥ / `single` modes |

## Key `settings.driver` keys

`gamma`, `tau`, `min_ev`, `max_ev`, `spectrum_filepath`, `npz_filepath`, `polarization`, `perp_component`, `field_e_ref_filepath`, `reference_only`, `observables` (`cross_section` / `dissipative_power` / `A_raw`; default `["cross_section"]`)

```{warning}
Hybrid absorption (any run with a `plasmon` section, including `parallel` / `perpendicular` / `single`) **cannot** be checkpointed. Molecule-only three-kick absorption can; see [Checkpointing](../checkpointing.md).
```

## Typical outputs

- Spectrum PNG + CSV (default `observables: ["cross_section"]`; extra names are `{stem}_{observable}.png`)
- Optional `absorption.npz`
- Per-direction `x_dir/` / `y_dir/` / `z_dir/` CSVs (`polarization: full` only)
- `fields/field_e.csv` / `fields/field_p.csv` / `fields/field_e_ref.csv` for hybrid NP+molecule (`parallel` / `perpendicular` / `single`)

## Theory

- [Fourier Spectra methodology](../fourier.md)
- [Hybrid absorption observables](../observables.md) — `cross_section`, `dissipative_power`, `A_raw`

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
- [Hybrid absorption observables](../observables.md)
- [Checkpointing](../checkpointing.md) — molecule-only absorption only
