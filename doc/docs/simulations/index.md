# Drivers

PlasMol is organized around **drivers**: self-contained workflows selected by `settings.driver` (or inferred from which top-level JSON sections you provide).

| Driver | Purpose | Template |
| -------- | --------- | ---------- |
| [classical](classical.md) | Pure Meep FDTD (nanoparticle / empty cell) | `template-classical.json` |
| [quantum](quantum.md) | Pure RT-TDDFT time propagation | `template-quantum.json` |
| [plasmol](plasmol.md) | Hybrid FDTD ↔ RT-TDDFT | `template-plasmol.json` |
| [absorption](absorption.md) | Absorption spectra (kick or hybrid deconvolution) | `template-absorption.json` |

Less common workflows (MO diagrams, core-hole survey, tuning, NP cross-sections, scatter response, and source checks) are on [Other Drivers](other.md).

## Driver selection

```bash
python -m plasmol.main -f templates/template-<driver>.json -vv
```

If `settings.driver` is omitted:

- only `molecule` → **quantum**
- only `plasmon` → **classical**
- both → **plasmol**

Custom / specialized workflows always set `"driver": "<name>"` explicitly.
