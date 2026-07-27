# Verify source driver

`driver: "verify_source"`

## Purpose

**Empty-cell FDTD** check of the plasmon source: runs classical Meep with no molecule and plots the recorded E-field near the source.

## When to use

- Debugging custom or gaussian/continuous sources
- Confirming polarization and timing before expensive hybrid runs

## Required JSON

| Section | Role |
| --------- | ------ |
| `settings.driver` | `"verify_source"` |
| `plasmon.simulation` | Cell + PML |
| `plasmon.source` | Source to verify |

No nanoparticle or molecule is required (the driver forces a pure classical setup).

## Typical outputs

- `verify_source_*.png` field traces at the probe

## Template

Copy and edit:

```bash
cp templates/template-verify_source.json my_run.json
python -m plasmol.main -f my_run.json -vv -l run.log
```

Source (repo path): `templates/template-verify_source.json`

## See also

- [All simulations](index.md)
- [Usage](../usage.md) — parameter reference
