# Tune driver

`driver: "tune"`

## Purpose

Automatically determine **LRC range-separation parameter** and/or **CAP ε₀** for a molecule. Exits after tuning (no RT-TDDFT propagation).

## When to use

- Before long Fourier runs with range-separated hybrids or CAP broadening
- When JSON uses `"lrc_parameter": "tune"` or `"eps0": "tune"` / `{TUNE}` in XC

## Required JSON

| Section | Role |
| --------- | ------ |
| `settings.driver` | **Must** be `"tune"` when using tune markers |
| `molecule` | Geometry, basis, XC |
| Markers | `"lrc_parameter": "tune"` and/or `cap.eps0: "tune"` |

## Typical outputs

- Log lines with recommended ω and/or ε₀ values to copy into production input

## Template

Copy and edit:

```bash
cp templates/template-tune.json my_run.json
python -m plasmol.main -f my_run.json -vv -l run.log
```

Source (repo path): `templates/template-tune.json`

## See also

- [All simulations](index.md)
- [Usage](../usage.md) — parameter reference
