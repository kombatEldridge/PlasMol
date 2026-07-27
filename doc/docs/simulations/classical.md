# Classical driver

`driver: "classical"` (or plasmon-only input)

## Purpose

Run a **pure Meep FDTD** simulation: nanoparticle optics, empty-cell field evolution, optional field imaging/GIFs. No RT-TDDFT.

## When to use

- Cross-sections of spheres (see also [np_abs_cross_sec](np_abs_cross_sec.md))
- Testing sources, symmetries, and PML without a molecule
- Classical near-field maps

## Required JSON

| Section | Role |
| --------- | ------ |
| `settings` | `dt`, `t_end`; set `"driver": "classical"` or omit if only `plasmon` is present |
| `plasmon.simulation` | Cell, PML, medium index, symmetries |
| `plasmon.source` | Continuous / gaussian / custom Meep source |
| `plasmon.nanoparticle` | Optional sphere material/radius/center |
| `plasmon.images` | Optional PNG/GIF field dumps |

## Typical outputs

- Log of Meep progress
- Optional `plasmol-images/` frames and GIF if images are enabled

## Template

Copy and edit:

```bash
cp templates/template-classical.json my_run.json
python -m plasmol.main -f my_run.json -vv -l run.log
```

Source (repo path): `templates/template-classical.json`

## See also

- [All simulations](index.md)
- [Usage](../usage.md) — parameter reference
