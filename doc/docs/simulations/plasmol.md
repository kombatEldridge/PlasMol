# Hybrid PlasMol driver

`driver: "plasmol"` (or both `plasmon` + `molecule` without a custom driver)

## Purpose

**Self-consistent hybrid** loop: Meep advances Maxwell fields; the local E at the molecule drives RT-TDDFT; the induced dipole can be fed back into FDTD (`back_propagation`).

## When to use

- Plasmon–molecule coupling, SERS-style geometries, hybrid time series
- Not the preferred path for absorption *spectra* (use [fourier](fourier.md) with a plasmon section)

## Required JSON

| Section | Role |
| --------- | ------ |
| `plasmon` | Cell, source, optional NP |
| `plasmon.molecule.position` | Sample / coupling point (μm) |
| `molecule` | Quantum system (no molecular source — field comes from Meep) |

Do **not** set both `plasmon.source` and `molecule.source`.

## Typical outputs

- Coupled `field_e.csv` / `field_p.csv` at the molecule site
- Optional field movies if `plasmon.images` is set

## Theory

- [Theory & Methodology](../methodology.md)
- [Effective polarizability](../methodology/effective_polarizability.md)

## Template

Copy and edit:

```bash
cp templates/template-plasmol.json my_run.json
python -m plasmol.main -f my_run.json -vv -l run.log
```

Source (repo path): `templates/template-plasmol.json`

## See also

- [All simulations](index.md)
- [Usage](../usage.md) — parameter reference
