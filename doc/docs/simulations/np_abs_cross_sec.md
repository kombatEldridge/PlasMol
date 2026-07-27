# NP absorption cross-section driver

`driver: "np_abs_cross_sec"`

## Purpose

Classical **absorption, scattering, and extinction** efficiencies for a spherical nanoparticle using Meep flux boxes (Mie-type diagnostics).

## When to use

- NP plasmon resonance scans without a molecule
- Comparing materials / radii / host indices

## Required JSON

| Section | Role |
| --------- | ------ |
| `settings.driver` | `"np_abs_cross_sec"` |
| `plasmon` | Cell, broadband source, nanoparticle |
| `additional_parameters` | `n_flux_freqs`, `flux_padding`, optional `line_fit`, `decay_stop` |

## Typical outputs

- Efficiency spectra / arrays and optional Lorentzian peak fits

## Template

Copy and edit:

```bash
cp templates/template-np_abs_cross_sec.json my_run.json
python -m plasmol.main -f my_run.json -vv -l run.log
```

Source (repo path): `templates/template-np_abs_cross_sec.json`

## See also

- [All simulations](index.md)
- [Usage](../usage.md) — parameter reference
