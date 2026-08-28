# Fourier driver

`driver: "fourier"`

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
| `settings.driver` | `"fourier"` |
| `molecule` | Geometry + electronic structure |
| `molecule.source` | May be just `{"type": "kick"}` — other kick fields default under Fourier |
| `additional_parameters.fourier` | At least `spectrum_filepath` (or files spectra path) |
| `plasmon` | Optional; required for hybrid / ∥ / ⊥ modes |

## Key `fourier` keys

`gamma`, `tau`, `min_ev`, `max_ev`, `spectrum_filepath`, `npz_filepath`, `polarization`, `perp_component`, `field_e_ref_filepath`, `reference_only`

## Typical outputs

- Spectrum PNG + CSV
- Optional `fourier.npz`
- Per-direction `x_dir/` / `y_dir/` / `z_dir/` CSVs (full mode)

## Theory

- [Fourier Spectra methodology](../methodology/fourier.md)

## Template

Copy and edit:

```bash
cp templates/template-fourier.json my_run.json
python -m plasmol.main -f my_run.json -vv -l run.log
```

Source (repo path): `templates/template-fourier.json`

## See also

- [All simulations](index.md)
- [Usage](../usage.md) — parameter reference
