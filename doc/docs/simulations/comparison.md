# Comparison driver

`driver: "comparison"`

## Purpose

**Ground-state SCF** comparisons across basis sets and XC functionals; produces MO energy diagrams (no time propagation).

## When to use

- Choosing basis/XC for production runs
- Publication-style MO level plots

## Required JSON

| Section | Role |
| --------- | ------ |
| `settings.driver` | `"comparison"` |
| `molecule` | Geometry (basis/xc may be placeholders when comparison supplies lists) |
| `additional_parameters.comparison` | `bases`, `xcs`, optional LRC map, plot limits, `dir_name` |

Incompatible with plasmon or Fourier sections.

## Typical outputs

- `mo_comparison/` (or `dir_name`) individual and grid MO plots

## Template

Copy and edit:

```bash
cp templates/template-comparison.json my_run.json
python -m plasmol.main -f my_run.json -vv -l run.log
```

Source (repo path): `templates/template-comparison.json`

## See also

- [All simulations](index.md)
- [Usage](../usage.md) — parameter reference
