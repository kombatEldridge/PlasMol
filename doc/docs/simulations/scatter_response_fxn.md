# Scatter response function driver

`driver: "scatter_response_fxn"`

## Purpose

Chen2010-style **scatter response** workflow: multiple polarized FDTD runs (with/without NP) and probe-point field recording for λ-dependent response analysis.

## When to use

- Reproducing multi-pol / vacuum-vs-NP classical response maps
- Collecting probe-point time series for offline post-processing

## Required JSON

| Section | Role |
| --------- | ------ |
| `settings.driver` | `"scatter_response_fxn"` |
| `plasmon` | Cell, source, optional NP |
| `additional_parameters.probe_points` | List of `[x,y,z]` (μm) sample locations |

## Typical outputs

- Probe field data / pickles under the job directory (see driver logging)

## Template

Copy and edit:

```bash
cp templates/template-scatter_response_fxn.json my_run.json
python -m plasmol.main -f my_run.json -vv -l run.log
```

Source (repo path): `templates/template-scatter_response_fxn.json`

## See also

- [All simulations](index.md)
- [Usage](../usage.md) — parameter reference
