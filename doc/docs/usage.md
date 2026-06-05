# Usage

PlasMol is run from the command line and is controlled almost entirely by a single **JSON input file**. 

## Command-Line Interface (CLI)

```bash
python -m plasmol.main -f input.json [options]
```

**Options**:

- `-f`, `--input` : Path to JSON input file (will assume first file given is input if not specified).
- `-l`, `--log` : Path to log file (default: print to terminal).
- `-v`, `-vv` : Verbosity (`-v` = INFO (default), `-vv` = DEBUG).
- `--describe` : Print a rich table of **all** supported parameters (with types, defaults, descriptions, units) and exit. Extremely useful for exploring the schema.
- `--help` : Show CLI help.

Example:
```bash
python -m plasmol.main --describe
```

## Overall JSON Structure

The input file has five top-level keys (all optional except `settings`):

```json
{
  "settings": { ... },
  "plasmon": { ... },               // classical FDTD / nanoparticle
  "molecule": { ... },              // RT-TDDFT molecule
  "files": { ... },                 // output paths
  "additional_parameters": { ... }  // fourier, comparison, probe_points, custom drivers
}
```

Comments are supported in JSON using `#`, `--`, `%`, or `//` (they are stripped before parsing).

Values listed as `null` in the exerpts below are considered optional and are just stated to highlight them.

## 1. "settings"

This section is required.

```json
{
  "settings": {
    "dt": 0.1,
    "t_end": 400,
    "driver": null
  }
}
```

| Key | Type | Default | Description | Units |
|-----|------|---------|-------------|-------|
| `dt` | float | – | Time step | a.u. |
| `t_end` | float | – | Simulation end time | a.u. |
| `driver` | str or null | null | Force a specific driver name | – |

Driver is inferred automatically if not specified:

- Only `"molecule"` top-level key → `quantum` driver
- Only `"plasmon"` top-level key → `classical` driver
- Both `"molecule"` and `"plasmon"` top-level keys → `plasmol` driver
- If `"driver"` present in `"settings: {...}"` → forces a custom driver

## 2. "plasmon"

Contains everything needed for Meep FDTD simulations of nanoparticles (and the classical part of hybrid runs).

### 2.1 "simulation"
These are the general paramters necessary to run a MEEP simulation. More information on these parameters can be found in the [MEEP documentation](https://meep.readthedocs.io/).

```json
{
  "simulation": {
    "cell_length": 0.1,
    "cell_volume": [0.1, 0.1, 0.1],
    "pml_thickness": 0.01,
    "symmetries": ["Y", 1, "Z", -1],
    "surrounding_material_index": 1.33,
    "courant": 0.5,
  }
}
```

| Key | Type | Description | Default | Units |
|-----|------|---------|-------------|-------|
| `cell_length`                 | int or float | Simulation cubic box length | 0.1 | μm |
| `cell_volume`                 | list of int or float | Simulation box size; `cell_volume` overrides `cell_length` if both are given | – | μm |
| `pml_thickness`               | int or float | Perfectly matched layer thickness; recommended ≈ λ_max / 2 | 0.01 | μm |
| `symmetries`                  | list of str, int pairs | Pairs of `axis, phase`, e.g. `["Y", 1, "Z", -1]`; Axes = X/Y/Z, phase = ±1 | – | – |
| `surrounding_material_index`  | int or float | Refractive index of background medium | 1.33 | – |
| `courant`                     | int or float | Courant number for stability | 0.5 | – |


### 2.2 "source"

Defines the incident electromagnetic source within the FDTD simulation. Again, if a hybrid system is requested, this will be the only allowed incident electric field.

```json
{
  "source": {
    "type": "continuous",
    "center": [-0.04, 0, 0],
    "size": [0, 0.1, 0.1],
    "component": "z",
    "amplitude": 1.0,
    "is_integrated": true,
    "additional_parameters": {
      "frequency": 5.0,
      "wavelength": null,
      "start_time": 0,
      "end_time": 1e20,
      "width": 0,
      "fwidth": null,
      "slowness": 3.0,
      "cutoff": 5.0,
      "src_func": null
    }
  }
}
```

**Common fields**:

| Key | Type | Source Type | Description | Default | Units |
|-----|------|---------|----|-------------|-------|
| `type` | str | All | Type of preset electric field to add ("continuous", "gaussian", or <custom\>) | – | – |
| `center` | list of 3 floats | All | Center coordinates of the source | – | μm |
| `size` | list of 3 floats | All | Size of the source volume; for 2D/3D sources, set the propagation dimension size to 0 | – | μm |
| `component` | str | All | Electric field component the source acts on ("x", "y", "z") | – | – |
| `amplitude` | int or float | All | Overall amplitude multiplying the source | 1 | arb. |
| `is_integrated` | bool | All | Whether the source is integrated over time (dipole moment) | True | – |
| `additional_parameters.frequency` | int or float | All | Frequency of the source | – | 1/μm |
| `additional_parameters.wavelength` | int or float | `continuous` + `gaussian` | Frequency of the source; gets converted to frequency if given instead | – | μm |
| `additional_parameters.start_time` | int or float | All | The starting time for the source | 0 | t_meep |
| `additional_parameters.end_time` | int or float | All | The end time for the source | 1e20 | t_meep |
| `additional_parameters.width` | int or float | `continuous` + `gaussian`| Roughly, the temporal width of the smoothing | 0 | – |
| `additional_parameters.fwidth` | int or float | All |  frequency width is proportional to the inverse of the temporal width; equal to 1/width | inf | – |
| `additional_parameters.slowness` | int or float | `continuous` | Controls how far into the exponential tail of the tanh function the source turns on | 3.0 | – |
| `additional_parameters.cutoff` | int or float | `gaussian` | How many widths the current decays for before it is cut off and set to zero  | 5.0 | – |

If you want to provide a custom source, you'll need to go into `classical/sources.py` and inject the source function (`src_func`). Additionally the `"type"` must be the name of the function. 

Note: At v1.1.0, all other variables for your custom source function not stated above as supported (such as wavelength and frequency) must be hard coded into the `src_func` within `classical/sources.py`.

### 2.3 "nanoparticle"

This section specifies details about the singular NP in your simulation.

```json
{
  "nanoparticle": {
    "material": "Au_JC_visible",
    "radius": 0.03,
    "center": [0, 0, 0]
  }
}
```

| Key | Type | Description | Default | Units |
|-----|------|---------|-------------|-------|
| `material` | str | Name from `meep.materials` (e.g. `Au_JC_visible`, `Ag_JC_visible`) | – | – |
| `radius`   | int or float | Radius of Spherical NP | – | μm |
| `center`   | list of int or float | Center position of Spherical NP | [0, 0, 0] | μm |

Note: At v1.1.0, only **spherical** NPs are supported.

### 2.4 "images"

Generate PNG frames and optional GIF of |E| evolution.

```json
{
  "images": {
    "timesteps_between": 1,
    "additional_parameters": ["-m -3", "-M 10", "-Zc dkbluered", "-S 10"],
    "dir_name": "fdtd_frames",
    "make_gif": true
  }
}
```

| Key | Type | Description | Default | Units |
|-----|------|---------|-------------|-------|
| `timesteps_between` | int | Number of Meep timesteps between PNG frames | – | – |
| `additional_parameters` | list | Additional arguments passed to Meep's output_png (from h5topng) | ['-Zc dkbluered', '-S 10'] | – |
| `dir_name` | str | Directory name where PNG frames will be saved | plasmol-images | – |
| `make_gif` | bool | Automatically create animated GIF from the PNG frames after simulation | True | – |


### 2.5 "molecule"

This set is necessary to specify details about how the molecule will be treated within the MEEP framework.

```json
{
  "molecule": {
    "position": [0, 0, 0],
    "tolerance_field_e": 1e-20,
    "back_propagation": true
  }
}
```

| Key | Type | Description | Default | Units |
|-----|------|---------|-------------|-------|
| `position` | list of int or float | Location of the quantum molecule inside the Meep cell | – | μm |
| `tolerance_field_e` | int or float | Minimum E at molecule position before triggering quantum propagation (hybrid only) | 1e-20 | a.u. |
| `back_propagation` | bool | Whether to inject the molecular induced dipole back into Meep as a CustomSource | True | – |

## 3. "molecule"

Contains all parameters for the RT-TDDFT quantum simulation of the molecule. This section is used both for pure quantum runs and for the quantum part of hybrid `plasmol` runs.

### 3.1 Geometry & Electronic Structure

```json
{
  "geometry": [
    {"atom": "O", "coord": [0.0, 0.0, -0.1302]},
    {"atom": "H", "coord": [1.4891, 0.0, 1.0332]},
    {"atom": "H", "coord": [-1.4891, 0.0, 1.0332]}
  ],
  "geometry_units": "bohr",
  "charge": 0,
  "spin": 0,
  "basis": "6-31g",
  "xc": "pbe0",
  "lrc_parameter": null
}
```

| Key | Type | Description | Default | Units |
|-----|------|---------|-------------|-------|
| `geometry` | list of dicts or str | List of `{"atom": "...", "coord": [x,y,z]}` or path to a `.xyz` file | – | – |
| `geometry_units` | str | Units of the geometry coordinates ("bohr" or "angstrom") | – | – |
| `charge` | int | Total molecular charge | 0 | – |
| `spin` | int | Spin multiplicity minus one (0 = closed shell) | 0 | – |
| `basis` | str | Basis set name (e.g. `"6-31g"`, `"def2-tzvpp"`) | – | – |
| `xc` | str | Exchange-correlation functional (PySCF/Libxc name) | – | – |
| `lrc_parameter` | float or `"tune"` | Range-separation parameter μ (ω) for RSH functionals; use `"tune"` for automatic IP-tuning | – | a.u. |

For `geometry` entries given as `.xyz` files, they must follow this format:

 - First line: total number of atoms (optional)
 - Second line: molecule name or comment (optional)
 - All other lines: element symbol or atomic number, x, y, and z coordinates, separated by spaces or tabs


### 3.2 Propagator

```json
{
  "propagator": {
    "type": "magnus2",
    "pc_convergence": 1e-12,
    "max_iterations": 200
  },
  "hermiticity_tolerance": 1e-12
}
```

| Key | Type | Description | Default | Units |
|-----|------|---------|-------------|-------|
| `propagator.type` | str | Time-propagation algorithm ("magnus2", "rk4", or "step") | `"magnus2"` | – |
| `propagator.pc_convergence` | float | Predictor-corrector convergence threshold (Magnus2 only) | 1e-12 | a.u. |
| `propagator.max_iterations` | int | Maximum predictor-corrector iterations (Magnus2 only) | 200 | – |
| `hermiticity_tolerance` | float | Tolerance used when checking that matrices are Hermitian | 1e-12 | a.u. |

### 3.3 Quantum Source (pure quantum runs only)

When running a standalone RT-TDDFT simulation (no `"plasmon"` section), you must provide an incident electric field via this block.

```json
{
  "source": {
    "type": "pulse",
    "intensity": 0.001,
    "peak_time": 10,
    "width_steps": 50,
    "component": "z",
    "additional_parameters": {
      "wavelength": 0.5,
      "frequency": null
    }
  }
}
```

| Key | Type | Source Type | Description | Default | Units |
|-----|------|---------|----|-------------|-------|
| `type` | str | All | Shape of the external field ("pulse" or "kick") | – | – |
| `intensity` | float | All | Peak electric-field strength | – | a.u. |
| `peak_time` | float | All | Time at which the pulse/kick reaches maximum | – | a.u. |
| `width_steps` | int | All | Width of the pulse in number of time steps | – | – |
| `component` | str | All | Direction of the electric field ("x", "y", or "z") | – | – |
| `additional_parameters.wavelength` | float | `pulse` | Central wavelength of the pulse | – | μm |
| `additional_parameters.frequency` | float | `pulse` | Central frequency of the pulse (alternative to wavelength) | – | 1/a.u. |

For absorption spectra use `"type": "kick"` together with the `"fourier"` section under `additional_parameters` top-level key.

### 3.4 Broadening (Lopata-style)

Optional energy-dependent imaginary potential added to the Fock matrix for lifetime effects and smoother spectra. See the [Lopata paper](https://pubs.acs.org/doi/abs/10.1021/ct400569s) for more details.

```json
{
  "broadening": {
    "type": "static",
    "gam0": 1.0,
    "xi": 0.5,
    "eps0": 0.0477,
    "clamp": 100
  }
}
```

| Key | Type | Description | Default | Units |
|-----|------|---------|-------------|-------|
| `type` | str | `"static"` or `"dynamic"` broadening | `"static"` | – |
| `gam0` | float | Base broadening strength | 1.0 | a.u. |
| `xi` | float | Exponent controlling energy dependence of broadening | 0.5 | – |
| `eps0` | float or `"tune"` | Reference energy (vacuum level); use `"tune"` for automatic estimation | 0.05 | a.u. |
| `clamp` | float | Maximum allowed broadening value | 100 | a.u. |

## 4. "files"

Controls output file names and checkpointing behavior.

```json
{
  "files": {
    "checkpoint": {
      "frequency_steps": 100,
      "frequency_time": null,
      "filepath": "checkpoint.npz"
    },
    "field_e_filepath": "field_e.csv",
    "field_p_filepath": "field_p.csv",
    "spectra_e_vs_p_filepath": "output.png"
  }
}
```

| Key | Type | Description | Default | Units |
|-----|------|---------|-------------|-------|
| `checkpoint.frequency_steps` | int | Number of time steps between checkpoint saves | – | – |
| `checkpoint.frequency_time` | float | Amount of simulation time between checkpoint saves (alternative to frequency_steps) | – | a.u. |
| `checkpoint.filepath` | str | Path to the `.npz` checkpoint file | – | – |
| `field_e_filepath` | str | CSV file for the electric field felt by the molecule | `"field_e.csv"` | – |
| `field_p_filepath` | str | CSV file for the induced dipole (polarization) of the molecule | `"field_p.csv"` | – |
| `spectra_e_vs_p_filepath` | str | PNG file showing incident field vs. molecular response | auto-timestamped | – |

**Note**: Checkpointing is only supported for pure quantum simulations. Use either `frequency_steps` **or** `frequency_time`, not both.

## 5. "additional_parameters"

This top-level section holds advanced or workflow-specific options.

### 5.1 "fourier" (Absorption spectrum workflow)

When present (and the molecule source is a delta kick), PlasMol automatically runs three directional simulations and performs a Fourier transform to produce an absorption spectrum.

```json
{
  "fourier": {
    "gamma": 0.01,
    "min_ev": 1.5,
    "max_ev": 5.0,
    "spectrum_filepath": "spectrum.png",
    "npz_filepath": null,
    "field_p_damping_gamma": null
  }
}
```

| Key | Type | Description | Default | Units |
|-----|------|---------|-------------|-------|
| `gamma` | float | Broadening (damping) factor applied before FFT | – | a.u. |
| `min_ev` | float | Lower energy limit of the plotted spectrum | 1.5 | eV |
| `max_ev` | float | Upper energy limit of the plotted spectrum | 5.0 | eV |
| `spectrum_filepath` | str | Output PNG file for the absorption spectrum | – | – |
| `npz_filepath` | str | Optional `.npz` file containing raw Fourier data | – | – |
| `field_p_damping_gamma` | float | Extra artificial damping applied to the time-domain polarization signal | – | a.u. |

### 5.2 "comparison" (MO energy diagrams)

Runs a series of ground-state SCF calculations for different basis sets / XC functionals and produces publication-quality MO energy plots.

```json
{
  "comparison": {
    "bases": ["6-31g", "def2-tzvpp"],
    "xcs": ["pbe0", "b3lyp", "cam-b3lyp"],
    "lrc_parameters": {"cam-b3lyp": 0.33},
    "num_occupied": 5,
    "num_virtual": 10,
    "y_min": -1.0,
    "y_max": 0.6,
    "index_min": null,
    "index_max": null,
    "dir_name": "mo_comparison"
  }
}
```

| Key | Type | Description | Default | Units |
|-----|------|---------|-------------|-------|
| `bases` | list of str | List of basis sets to compare | – | – |
| `xcs` | list of str | List of exchange-correlation functionals to compare | – | – |
| `lrc_parameters` | dict | Mapping of XC name → μ value for range-separated hybrids | – | – |
| `num_occupied` | int | Number of occupied orbitals to display | – | – |
| `num_virtual` | int | Number of virtual orbitals to display | – | – |
| `y_min` / `y_max` | float | Energy axis limits (Hartree) | – | Ha |
| `index_min` / `index_max` | int | MO index range to display (1-based) | – | – |
| `dir_name` | str | Directory in which comparison plots are saved | auto-timestamped | – |

### 5.3 "probe_points"

List of spatial locations (in μm) at which the electric field time series will be recorded. Primarily used by custom classical drivers such as `chen2010_fig1`.

```json
{
  "probe_points": [
    [0.011, 0, 0],
    [0.012, 0, 0]
  ]
}
```

Each entry is a list of three floats `[x, y, z]`.

### 5.4 Using a Custom Driver

To run a completely custom workflow, set the driver name in `settings` and (optionally) supply extra parameters under `additional_parameters`.

```json
{
  "settings": {
    "dt": 0.1,
    "t_end": 100,
    "driver": "my_awesome_driver"
  },
  "additional_parameters": {
    "my_custom_setting": 42
  }
}
```

See the dedicated guide [Adding Custom Drivers](adding_custom_drivers.md) for the full procedure (creating the driver file, registering it in `get_driver()`, and adding parameter definitions).

---

*This document reflects PlasMol v1.1.0 JSON input format.*