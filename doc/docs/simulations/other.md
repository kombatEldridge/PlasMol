# Other Drivers

Specialized workflows that are not part of the usual [classical](classical.md), [quantum](quantum.md), [hybrid](plasmol.md), or [absorption](absorption.md) runs. Select one by setting `settings.driver` to the name below.

| Driver | Purpose | Template |
| -------- | --------- | ---------- |
| [comparison](comparison.md) | Ground-state MO energy diagrams | `template-comparison.json` |
| [core_hole](core_hole.md) | Survey MO contributions by atom index and PySCF shell (SCH/DCH is `molecule.core_hole`) | `template-core_hole.json` |
| [tune](tune.md) | Auto-tune LRC ω and/or CAP ε₀ | `template-tune.json` |
| [np_abs_cross_sec](np_abs_cross_sec.md) | NP absorption / scattering efficiencies | `template-np_abs_cross_sec.json` |
| [scatter_response_fxn](scatter_response_fxn.md) | Multi-pol scatter response (Chen2010-style) | `template-scatter_response_fxn.json` |
| [verify_source](verify_source.md) | Empty-cell source visualization | `template-verify_source.json` |

## See also

- [Drivers](index.md) — main drivers and how a driver is chosen
- [Custom drivers](../custom_drivers.md) — register a workflow of your own
