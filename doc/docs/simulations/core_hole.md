# Core-hole driver

`driver: "core_hole"`

## Purpose

**Sudden single/double core-hole** initial conditions (SCH/DCH) with RT-TDDFT and MO hole-occupation tracking on the neutral MO basis.

## When to use

- Core-ionized dynamics, hole migration / occupation plots
- Survey which atoms contribute to a candidate core MO

## Required JSON

| Section | Role |
| --------- | ------ |
| `settings.driver` | `"core_hole"` |
| `molecule` | Neutral geometry; open-shell is forced after hole creation |
| `additional_parameters.mo_removal_index_dict` | e.g. `{"0": 2}` DCH, `{"0": 1}` SCH |
| `additional_parameters.core_hole_mo_occ_filepath` | Occupation CSV path |

Optional: `core_hole_watch_indices`, amplitude filter, `check_mo_contrib_by_atom` survey mode.

## Typical outputs

- `mo_occ.csv` and occupation plot PNG
- Standard field CSVs if a source is present

## Theory

- [Core-Hole Dynamics](../methodology/core_hole.md)

## Template

Copy and edit:

```bash
cp templates/template-core_hole.json my_run.json
python -m plasmol.main -f my_run.json -vv -l run.log
```

Source (repo path): `templates/template-core_hole.json`

## See also

- [All simulations](index.md)
- [Usage](../usage.md) — parameter reference
