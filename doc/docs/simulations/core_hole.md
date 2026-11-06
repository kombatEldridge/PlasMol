---
orphan: true
---

# Core-hole driver

`driver: "core_hole"`

## Purpose

**Survey** which atoms and PySCF shells contribute to candidate MOs, then exit. This driver does **not** ionize or propagate.

Sudden SCH/DCH is `molecule.core_hole` on a [quantum](quantum.md), [absorption](absorption.md), or [plasmol](plasmol.md) run. See [Core-Hole Dynamics](../core_hole.md).

## When to use

- Choose which MO to ionize before a production SCH/DCH run
- Print shell contributions on each atom for several MOs at once (more than two are allowed)

## Required JSON

| Section | Role |
| --------- | ------ |
| `settings.driver` | `"core_hole"` (no extra keys) |
| `molecule` | Neutral geometry, basis, xc |
| `molecule.core_hole.mo_removal_index_dict` | MOs to survey (electron counts ignored) |

## Typical outputs

- Log lines: `=== MO k (index k-1) contributions ===` with atom index, element, PySCF shell (`1s`, `2p`, `3d`, …), and percent
- `mo_survey.txt`, columns `mo_index`, `atom_index`, `element`, `shell`, `contribution_percent`

Angular components of one shell are added together (`2px` + `2py` + `2pz` is one `2p` row; spherical components likewise). A shell at or below 1% is omitted. The atom index is PySCF's 0-based index, so two atoms of the same element stay separate.

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

- [Other Drivers](other.md)
- [All drivers](index.md)
- [Usage](../usage.md) — `molecule.core_hole` for production SCH/DCH
