# Simulations

PlasMol is organized around **drivers**: self-contained workflows selected by `settings.driver` (or inferred from which top-level JSON sections you provide).

| Driver | Purpose | Template |
| -------- | --------- | ---------- |
| [classical](classical.md) | Pure Meep FDTD (nanoparticle / empty cell) | `template-classical.json` |
| [quantum](quantum.md) | Pure RT-TDDFT time propagation | `template-quantum.json` |
| [plasmol](plasmol.md) | Hybrid FDTD ↔ RT-TDDFT | `template-plasmol.json` |
| [fourier](fourier.md) | Absorption spectra (kick or hybrid deconvolution) | `template-fourier.json` |
| [core_hole](core_hole.md) | Sudden SCH/DCH + MO hole tracking | `template-core_hole.json` |
| [comparison](comparison.md) | Ground-state MO energy diagrams | `template-comparison.json` |
| [tune](tune.md) | Auto-tune LRC ω and/or CAP ε₀ | `template-tune.json` |
| [np_abs_cross_sec](np_abs_cross_sec.md) | NP absorption / scattering efficiencies | `template-np_abs_cross_sec.json` |
| [scatter_response_fxn](scatter_response_fxn.md) | Multi-pol scatter response (Chen2010-style) | `template-scatter_response_fxn.json` |
| [verify_source](verify_source.md) | Empty-cell source visualization | `template-verify_source.json` |

## Driver selection

```bash
python -m plasmol.main -f templates/template-<driver>.json -vv
```

If `settings.driver` is omitted:

- only `molecule` → **quantum**
- only `plasmon` → **classical**
- both → **plasmol**

Custom / specialized workflows always set `"driver": "<name>"` explicitly.

## Related docs

- [Usage](../usage.md) — full JSON schema
- [Tutorials](../tutorials.md) — walkthroughs
- [Custom Drivers](../custom_drivers.md) — registering new drivers
- [Theory & Methodology](../methodology.md)
