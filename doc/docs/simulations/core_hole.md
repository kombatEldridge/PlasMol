# Core-hole driver

`driver: "core_hole"`

## Purpose

**Survey** which atoms contribute to candidate MOs (`check_mo_contrib_by_atom`), then exit. This driver does **not** ionize or propagate.

Sudden SCH/DCH is `molecule.core_hole` on a [quantum](quantum.md), [absorption](absorption.md), or [plasmol](plasmol.md) run. See [Core-Hole Dynamics](../core_hole.md).

## When to use

- Choose which MO to ionize before a production SCH/DCH run
- Print per-atom AO contributions for several MOs at once (more than two are allowed)

## Required JSON

| Section | Role |
| --------- | ------ |
| `settings.driver` | `"core_hole"` (no extra keys) |
| `molecule` | Neutral geometry, basis, xc |
| `molecule.core_hole.mo_removal_index_dict` | MOs to survey (electron counts ignored) |

## Typical outputs

- Log lines: `=== MO k (index k-1) contributions ===` with atom percentages

## Theory

- [Core-Hole Dynamics](../core_hole.md)

## Template

Copy and edit:

```bash
cp templates/template-core_hole.json my_run.json
python -m plasmol.main -f my_run.json -vv -l run.log
```

Source (repo path): `templates/template-core_hole.json`

## See also

- [All simulations](index.md)
- [Usage](../usage.md) — `molecule.core_hole` for production SCH/DCH
