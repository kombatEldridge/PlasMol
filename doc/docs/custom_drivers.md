# Custom Drivers

PlasMol routes every run through `get_driver(driver_str)` in `plasmol/drivers/__init__.py`. Built-in drivers cover classical FDTD, quantum RT-TDDFT, hybrid PlasMol, and several specialized workflows.

## Built-in drivers

| `settings.driver` | Module | Purpose |
| ------------------- | -------- | --------- |
| `classical` | `drivers/classical.py` | Pure Meep FDTD |
| `quantum` | `drivers/quantum.py` | Pure RT-TDDFT |
| `plasmol` | `drivers/plasmol.py` | Hybrid FDTD ↔ RT-TDDFT |
| `absorption` | `custom_drivers/absorption/` | Absorption spectra (kick or hybrid deconvolution; full/∥/⊥) |
| `core_hole` | `custom_drivers/core_hole.py` | Survey per-atom MO contributions (no propagation) |
| `comparison` | `custom_drivers/comparison.py` | MO energy diagrams across bases/XCs |
| `tune` | `custom_drivers/tune.py` | Auto-tune LRC ω and CAP ε₀ |
| `np_abs_cross_sec` | `custom_drivers/np_abs_cross_sec.py` | NP absorption/scattering efficiencies |
| `scatter_response_fxn` | `custom_drivers/scatter_response_fxn.py` | Chen2010-style scatter response |
| `verify_source` | `custom_drivers/verify_source.py` | Empty-cell source check |

If `settings.driver` is omitted, PlasMol **infers** the driver: molecule only → quantum, plasmon only → classical, both → plasmol. Specialized workflows should set `driver` explicitly.

## Registering your own driver

1. Create `plasmol/drivers/custom_drivers/my_driver.py` with a `run(params)` function.
2. Import and register it in `plasmol/drivers/__init__.py` inside `get_driver`.
3. Add any new JSON keys to `param_defs` in `plasmol/utils/struct.py` (paths under `settings.driver`, with a boolean gate such as `has_my_driver`, and list them in `DRIVER_PARAM_KEYS`).
4. Validate in `plasmol/utils/params_helpers/has_<gate>.py` (`check` / `form`) as needed.
5. Document the keys in [Usage](usage.md) and add unit tests under `tests/`.

```python
# drivers/__init__.py (sketch)
from plasmol.drivers.custom_drivers.my_driver import run as run_my_driver

def get_driver(driver_str):
    ...
    elif driver_str == "my_driver":
        return run_my_driver
```

## See also

- [Usage](usage.md) — JSON schema
- [Contributing](contributing.md) — style and PR process
- [Core-Hole Dynamics](core_hole.md) — `molecule.core_hole` sudden SCH/DCH; `core_hole` driver survey
- [Fourier Spectra](fourier.md) — `absorption` driver theory
- [Hybrid absorption observables](observables.md) — \(\sigma_m\), \(A_{\mathrm{diss}}\), \(A_{\mathrm{raw}}\)
