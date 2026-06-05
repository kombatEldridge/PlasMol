# Adding Custom Drivers

PlasMol's driver architecture is designed to be extensible. While the core provides `classical` (pure FDTD), `quantum` (pure RT-TDDFT), and `plasmol` (hybrid) drivers, you can implement **completely custom workflows** by writing your own `run(params)` function.

Custom drivers are perfect for:
- Replicating specific results or figures from research papers (see `scatter_response_fxn`)
- Computing specialized quantities such as absorption/scattering cross-sections (`np_abs_cross_sec`, `plasmol_abs_cross_sec`)
- Running parameter sweeps, optimization, or batch simulations (`comparison`, `fourier`)
- Implementing novel analysis pipelines (SERS enhancement mapping, hot-carrier dynamics, plexcitons, custom post-processing, etc.)
- Integrating PlasMol with external codes or experimental data

All existing "custom" drivers are located in `plasmol/drivers/custom_drivers/` and serve as excellent templates.

## Prerequisites

- Comfortable with PlasMol's [JSON input format](usage.md)
- Basic understanding of the `PARAMS` object (see [API Reference](api-reference.md))
- Python proficiency (and Meep/PySCF if your workflow uses classical or quantum components)

## Step 1: Create Your Custom Driver File

Create a new Python module inside the custom drivers directory:

```bash
touch plasmol/drivers/custom_drivers/my_awesome_driver.py
```

A minimal but complete structure looks like this:

```python
# plasmol/drivers/custom_drivers/my_awesome_driver.py
import logging
import copy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plasmol.utils.logging import setup_logging
from plasmol.utils.csv import init_csv, update_csv

logger = logging.getLogger("main")


def run(params):
    """
    Main entry point for the custom driver.
    
    `params` is a fully validated and populated PARAMS instance.
    All settings from your JSON (including those under additional_parameters)
    are available as attributes on this object (e.g. params.my_awesome_param).
    """
    logger.info("=== Starting My Awesome Custom Driver ===")
    
    # === 1. Read configuration ===
    my_param = getattr(params, 'my_awesome_param', 42)
    logger.info(f"Running with my_awesome_param = {my_param}")
    
    # Create output directory
    out_dir = Path(getattr(params, 'output_dir', 'my_awesome_outputs'))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # === 2. (Optional) Deep-copy params for sub-simulations ===
    # This is a common pattern when you need to run multiple variants
    p = copy.deepcopy(params)
    
    # === 3. Your custom logic goes here ===
    # Examples of what you might do:
    #
    # from plasmol.quantum.molecule import MOLECULE
    # molecule = MOLECULE(params)
    #
    # from plasmol.classical.simulation import SIMULATION
    # sim = SIMULATION(params)
    # sim.run()
    #
    # Or run multiple simulations in parallel with ProcessPoolExecutor
    # (see scatter_response_fxn.py and fourier.py for the PrefixFilter + setup_logging pattern)
    
    # === 4. Generate outputs (plots, CSVs, pickles, etc.) ===
    # Always write results to disk so the user has artifacts
    
    logger.info("=== My Awesome Custom Driver completed successfully ===")
```

### Best Practices Inside `run(params)`

- **Logging**: Always use `logger = logging.getLogger("main")`. It automatically respects `-v` / `-vv` and the `--log` file.
- **Child processes**: When using `ProcessPoolExecutor`, copy the `PrefixFilter` class and the `PRINTLOGGER` redirection pattern from `fourier.py` or `scatter_response_fxn.py`. This keeps child-process logs clean and prefixed.
- **Deep copying**: Use `copy.deepcopy(params)` before modifying `params` for different sub-runs. This prevents accidental state leakage.
- **Component instantiation**: If you need a `MOLECULE` or `SIMULATION` object, create it inside `run()` (or helper functions). The top-level `params` only auto-populates these when using the built-in drivers.
- **Error handling**: Wrap critical sections in try/except and log exceptions with `logger.error(..., exc_info=True)`.
- **Outputs**: Write everything to a dedicated folder (use a timestamp or params attribute). Good defaults: `field_e.csv`, `field_p.csv`, `spectrum.png`, etc.

## Step 2: Register the Driver in the Registry

Edit `plasmol/drivers/__init__.py`:

**Add the import** (group it with the other custom drivers):

```python
from plasmol.drivers.custom_drivers.my_awesome_driver import run as run_my_awesome_driver
```

**Add a branch in `get_driver(driver_str)`**:

```python
def get_driver(driver_str):
    if driver_str == 'classical':
        return run_classical
    ...
    elif driver_str == 'my_awesome_driver':
        return run_my_awesome_driver
    else:
        raise ValueError(
            f"Unknown driver: {driver_str}. "
            "Please add your custom driver to the drivers/__init__.py file."
        )
```

**Optionally** add `run_my_awesome_driver` to the `__all__` list.

After this change, the driver name `"my_awesome_driver"` becomes a valid value for `"settings": {"driver": "..."}` in your JSON files.

## Step 3: Register Parameters (Recommended)

If your driver accepts configuration through the JSON input, register the parameters in `plasmol/utils/input/struct.py` inside the `param_defs` list. This gives you:

- Automatic type validation
- Appearance in `python -m plasmol.main --describe`
- Sensible defaults
- Clean attribute access on the `params` object (`params.my_param` works directly)

### Example: Simple scalar parameter

Add near the other `additional_parameters` entries (at the bottom of `param_defs`):

```python
    ('my_awesome_param', ['additional_parameters', 'my_awesome_param'], False, 'has_custom', 42, None, (int, float), "Description of this awesome parameter", "arb. units"),
```

### Example: Grouped parameters (recommended for complex drivers)

```python
    # Section container (is_section_dict=True)
    ('my_awesome_dict', ['additional_parameters', 'my_awesome'], True, 'has_my_awesome', None, None, dict, None, None),
    
    # Leaf parameters
    ('my_awesome_param', ['additional_parameters', 'my_awesome', 'param'], False, 'has_my_awesome', 42, None, (int, float), "The main parameter", "units"),
    ('my_awesome_flag', ['additional_parameters', 'my_awesome', 'enable_foo'], False, 'has_my_awesome', False, None, bool, "Whether to enable feature Foo", None),
```

You can also add custom validation logic inside `PARAMS._validate_all()` (search for the `if self.has_custom:` block).

## Step 4: Test Your Driver

Create a minimal test JSON (you can put it in `templates/` or just use it locally):

```json
{
  "settings": {
    "dt": 0.1,
    "t_end": 50,
    "driver": "my_awesome_driver"
  },
  "additional_parameters": {
    "my_awesome_param": 100
  }
}
```

Run it:

```bash
python -m plasmol.main -f my_test.json -vv
```

You should see your log messages and any output files you wrote.

Iterate until everything works as expected.

## Step 5: Document and Share

- Add a short tutorial or example JSON in `tutorials.md`
- Mention the new driver in `usage.md` (under the custom driver section)
- If it introduces new concepts, update `api-reference.md`
- Consider contributing it back to the repository — custom drivers that solve real research problems are very welcome!

## Full Working Examples to Copy From

Look at the source code of these drivers for battle-tested patterns:

| Driver                    | What it demonstrates                          | Complexity |
|---------------------------|-----------------------------------------------|------------|
| `comparison.py`           | Ground-state SCF sweeps + rich matplotlib grid plots | Low       |
| `fourier.py`              | Three parallel directional simulations + FFT + damping | Medium    |
| `scatter_response_fxn.py`        | Four parallel FDTD runs with logging filters and pickle outputs | Medium-High |
| `np_abs_cross_sec.py`     | Flux-box calculations, curve fitting (scipy), multi-peak Lorentzian | High      |
| `plasmol_abs_cross_sec.py`| Hybrid version of the above (molecule + nanoparticle) | High      |

Any of them can serve as a starting template — just rename the file, change the `run` function, and register it.

## Troubleshooting

- **`ValueError: Unknown driver: ...`** — You forgot to add the `elif` branch in `get_driver()`.
- **`AttributeError: 'PARAMS' object has no attribute 'my_param'`** — The parameter is missing from your JSON **or** you didn't register it in `param_defs` (or the path is wrong).
- **Child process logs are missing or garbled** — Make sure you copied the `PrefixFilter` + `PRINTLOGGER` redirection code when using `ProcessPoolExecutor`.
- **Pickling / multiprocessing errors** — Large objects (Meep simulations, PySCF mean-field objects) cannot be pickled. Recreate them inside the worker function or pass only primitive data.
- **`has_custom` / driver validation errors** — Remember: when you specify `"driver"` in `settings`, you are responsible for providing all required sections (`molecule`, `plasmon`, `files`, etc.) that your code expects.

## Summary

Custom drivers turn PlasMol from "a simulation tool" into **a flexible research framework**. You keep the robust parameter parsing, logging, unit handling, and validation infrastructure while having complete freedom inside `run(params)`.

We look forward to seeing what you build!

---

*Happy coding! If you create something generally useful, consider opening a pull request.*