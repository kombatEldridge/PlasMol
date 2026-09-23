# Usage

PlasMol is run from the command line and is controlled almost entirely by a single **JSON input file**.

## Command-Line Interface (CLI)

```bash
python -m plasmol.main input.json [options]
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
  "settings": {
    "dt": 0.1,
    "t_end": 400,
    "driver": "plasmol"
  },
  "plasmon": {
    "simulation": {
      "cell_length": 0.1,
      "pml_thickness": 0.01,
      "surrounding_material_index": 1.33,
      "symmetries": ["Y", 1, "Z", -1],
      "courant": 0.5
    },
    "source": {
      "type": "continuous",
      "center": [-0.04, 0.0, 0.0],
      "size": [0.0, 0.1, 0.1],
      "component": "z",
      "amplitude": 1.0,
      "is_integrated": true,
      "additional_parameters": {
        "frequency": 5.0
      }
    },
    "nanoparticle": {
      "material": "Au_JC_visible",
      "radius": 0.03,
      "center": [0.0, 0.0, 0.0]
    },
    "molecule": {
      "position": [0.035, 0.0, 0.0],
      "tolerance_field_e": 1e-12,
      "back_propagation": true
    }
  },
  "molecule": {
    "geometry": [{"atom": "O", "coord": [0.0, 0.0, -0.1302]}, {"atom": "H", "coord": [1.4891, 0.0, 1.0332]}, {"atom": "H", "coord": [-1.4891, 0.0, 1.0332]}],
    "geometry_units": "bohr",
    "charge": 0,
    "spin": 0,
    "basis": "6-31g",
    "xc": "pbe0",
    "propagator": {
      "type": "magnus2",
      "pc_convergence": 1e-12,
      "max_iterations": 200
    }
  },
  "files": {
    "field_e_filepath": "field_e.csv",
    "field_p_filepath": "field_p.csv",
    "spectra_e_vs_p_filepath": "output.png"
  },
  "additional_parameters": {}
}
```

The top-level keys are `settings`, `plasmon`, `molecule`, `files`, and `additional_parameters`. Only `settings` is always required; which other sections appear depends on the driver (see [Simulations](simulations/index.md)).

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
| ----- | ------ | --------- | ------------- | ------- |
| `dt` | float | – | Time step | a.u. |
| `t_end` | float | – | Simulation end time | a.u. |
| `driver` | str, dict, or null | null | Driver name, or a dict with `"name"` plus that driver's keys | – |

`driver` may be a **string** (name only, no extra keys) or a **dict**:

```json
{ "driver": "quantum" }
```

```json
{
  "driver": {
    "name": "absorption",
    "polarization": "perpendicular",
    "spectrum_filepath": "spectrum_perp.png",
    "npz_filepath": "fourier_perp.npz",
    "min_ev": 1.5,
    "max_ev": 5.0,
    "field_e_ref_filepath": "field_e_ref_perp.csv"
  }
}
```

If `driver` is omitted it is inferred from which top-level sections are present:

- Only `"molecule"` → `quantum`
- Only `"plasmon"` → `classical`
- Both `"molecule"` and `"plasmon"` → `plasmol`

Driver-specific keys (absorption, comparison, core-hole, NP cross-section, scatter probes) live **on this dict**, next to `"name"`. See [§5](#5-additional_parameters) and [Simulations](simulations/index.md).

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
    "courant": 0.5
  }
}
```

| Key | Type | Description | Default | Units |
| ----- | ------ | --------- | ------------- | ------- |
| `cell_length` | int or float | Simulation cubic box length | 0.1 | μm |
| `cell_volume` | list of int or float | Simulation box size; `cell_volume` overrides `cell_length` if both are given | – | μm |
| `pml_thickness` | int or float | Perfectly matched layer thickness; recommended ≈ λ_max / 2 | 0.01 | μm |
| `symmetries` | list of str, int pairs | Pairs of `axis, phase`, e.g. `["Y", 1, "Z", -1]`; Axes = X/Y/Z, phase = ±1 | – | – |
| `surrounding_material_index` | int or float | Refractive index of background medium | 1.33 | – |
| `courant` | int or float | Courant number for stability | 0.5 | – |

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
      "end_time": 1e+20,
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
| ----- | ------ | --------- | ---- | ------------- | ------- |
| `type` | str | All | Type of preset electric field to add ("continuous", "gaussian", or <custom\>) | – | – |
| `center` | list of 3 floats | All | Center coordinates of the source | – | μm |
| `size` | list of 3 floats | All | Size of the source volume; for 2D/3D sources, set the propagation dimension size to 0 | – | μm |
| `component` | str | All | Electric field component ("x", "y", "z"). Optional when the driver sets polarization (hybrid `absorption` `full`/`parallel`/`perpendicular`, `scatter_response_fxn`). Required for absorption `polarization: single`. | – | – |
| `amplitude` | int or float | All | Overall amplitude multiplying the source | 1 | arb. |
| `is_integrated` | bool | All | Whether the source is integrated over time (dipole moment) | True | – |
| `additional_parameters.frequency` | int or float | All | Frequency of the source | – | 1/μm |
| `additional_parameters.wavelength` | int or float | `continuous` + `gaussian` | Frequency of the source; gets converted to frequency if given instead | – | μm |
| `additional_parameters.start_time` | int or float | All | The starting time for the source | 0 | t_meep |
| `additional_parameters.end_time` | int or float | All | The end time for the source | 1e20 | t_meep |
| `additional_parameters.width` | int or float | `continuous` + `gaussian` | Roughly, the temporal width of the smoothing | 0 | – |
| `additional_parameters.fwidth` | int or float | All | frequency width is proportional to the inverse of the temporal width; equal to 1/width | inf | – |
| `additional_parameters.slowness` | int or float | `continuous` | Controls how far into the exponential tail of the tanh function the source turns on | 3.0 | – |
| `additional_parameters.cutoff` | int or float | `gaussian` | How many widths the current decays for before it is cut off and set to zero | 5.0 | – |

If you want to provide a custom source, you'll need to go into `classical/sources.py` and inject the source function (`src_func`). Additionally the `"type"` must be the name of the function.

Note: At v1.2.0, all other variables for your custom source function not stated above as supported (such as wavelength and frequency) must be hard coded into the `src_func` within `classical/sources.py`.

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
| ----- | ------ | --------- | ------------- | ------- |
| `material` | str | Name from `meep.materials` (e.g. `Au_JC_visible`, `Ag_JC_visible`) | – | – |
| `radius` | int or float | Radius of Spherical NP | – | μm |
| `center` | list of int or float | Center position of Spherical NP | [0, 0, 0] | μm |

Note: At v1.2.0, only **spherical** NPs are supported.

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
| ----- | ------ | --------- | ------------- | ------- |
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
| ----- | ------ | --------- | ------------- | ------- |
| `position` | list of int or float | Location of the quantum molecule inside the Meep cell | – | μm |
| `tolerance_field_e` | int or float | Minimum E at molecule position before triggering quantum propagation (hybrid only) | 1e-20 | a.u. |
| `back_propagation` | bool | Whether to inject the molecular induced dipole back into Meep as a CustomSource | True | – |

## 3. "molecule"

Contains all parameters for the RT-TDDFT quantum simulation of the molecule. This section is used both for pure quantum runs and for the quantum part of hybrid `plasmol` runs.

### 3.1 Geometry & Electronic Structure

```json
{
  "geometry": [{"atom": "O", "coord": [0.0, 0.0, -0.1302]}, {"atom": "H", "coord": [1.4891, 0.0, 1.0332]}, {"atom": "H", "coord": [-1.4891, 0.0, 1.0332]}],
  "geometry_units": "bohr",
  "charge": 0,
  "spin": 0,
  "basis": "6-31g",
  "basis_coords": "cartesian",
  "xc": "pbe0",
  "lrc_parameter": null
}
```

| Key | Type | Description | Default | Units |
| ----- | ------ | --------- | ------------- | ------- |
| `geometry` | list of dicts or str | List of `{"atom": "...", "coord": [x,y,z]}` or path to a `.xyz` file | – | – |
| `geometry_units` | str | Units of the geometry coordinates ("bohr" or "angstrom") | – | – |
| `charge` | int | Total molecular charge | 0 | – |
| `spin` | int | Spin multiplicity minus one (0 = closed shell) | 0 | – |
| `basis` | str | Basis set name (e.g. `"6-31g"`, `"def2-tzvpp"`) | – | – |
| `basis_coords` | str | `"cartesian"` (6 d functions; **PlasMol default**, stock NWChem) or `"spherical"` (5 d; PySCF default). Aliases: `cart`, `sph` | `"cartesian"` | – |
| `cartesian` | bool | Same as `basis_coords`. Default `true`. Do not set both unless they agree | true | – |
| `grid_level` | int | PySCF DFT grid level (0–9). Omit for the PySCF default | – | – |
| `xc` | str | Exchange-correlation functional (PySCF/Libxc name or a compound mix) | – | – |
| `lrc_parameter` | float or `"tune"` | Range-separation parameter μ (ω) for RSH functionals; use `"tune"` for automatic IP-tuning | – | a.u. |

**Basis angular type.** PlasMol defaults to **Cartesian** Gaussians (`basis_coords: "cartesian"`, 6 \(d\) functions), matching stock NWChem. PySCF’s own default is spherical (5 \(d\)). Set `"basis_coords": "spherical"` (or `"cartesian": false`) to match PySCF.

A `geometry` string is a path to a `.xyz` file (relative paths are resolved from the input JSON’s directory). Use the usual two-line XYZ header, then one atom per line:

```xyz
3
water, optional comment — this entire line is ignored
O   0.000000   0.000000  -0.065588
H   0.000000   0.757000   0.520588
H   0.000000  -0.757000   0.520588
```

| Line (after blank lines are dropped) | Role |
| -------------------------------------- | ------ |
| 1 | Atom count (a lone integer). Optional as a *value*, but this slot must still exist |
| 2 | Title / comment. **Any text** is allowed and is never parsed as an atom |
| 3 … | `Symbol x y z` (or atomic number + coordinates), whitespace-separated |

`geometry_units` still applies (`"bohr"` or `"angstrom"`). Extra columns after *z* are ignored.

```{note}
The reader always takes coordinates starting at the **third non-empty line**. The first line does **not** have to be the atom count (any lone integer in the file can supply the count), but you still need two non-empty lines before the atoms you want parsed. A file that is only coordinate lines will skip the first two atoms.
```

```{warning}
JSON comment markers (`#`, `//`, `--`, `%`) are **not** stripped from `.xyz` files. Do not put `#` comments on atom lines or extra comment lines between atoms — those shift the “start at line 3” window and will be read as atoms. Put remarks on line 2 only. Comma-separated coordinates (`O,0,0,0`) are not accepted.
```

#### Rotation

Optional. Applied to the parsed geometry **before SCF**, about the nuclear-charge center, so the molecule does not translate. The input `.xyz` is left unchanged. The rotated Bohr geometry is written to `rotated_geometry.xyz` in the working directory.

A step is either an axis-angle or a bond alignment. A list of steps is applied in order. Axes are fixed lab axes (a later step does not follow the molecule). `align` maps the vector from `from_atom` to `to_atom` (0-based) onto `axis`, then applies `twist_deg` about that axis (default 0).

```json
{"rotation": {"axis": "z", "angle_deg": 90}}
```

```json
{"rotation": [{"align": {"from_atom": 2, "to_atom": 3, "axis": "x"}, "twist_deg": 90}]}
```

| Key | Type | Description | Default | Units |
| ----- | ------ | ------------- | ------- | ----- |
| `axis` | str or list | `"x"`, `"y"`, `"z"`, or a 3-vector. Not used together with `align` | – | – |
| `angle_deg` | float | Right-handed rotation about `axis` | – | degrees |
| `align.from_atom`, `align.to_atom` | int | Atom indices. The bond is `coord[to] - coord[from]` | – | – |
| `align.axis` | str or list | Lab direction that bond is mapped onto | – | – |
| `twist_deg` | float | Extra right-handed rotation about `align.axis` after the bond is aligned | 0 | degrees |

Omit `rotation` to keep the input frame. The same block is honored on `quantum`, `absorption`, `plasmol`, the core-hole survey, `comparison`, and `tune`, because each of those builds the molecule from `molecule_coords`.

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
| ----- | ------ | --------- | ------------- | ------- |
| `propagator.type` | str | Time-propagation algorithm ("magnus2", "rk4", or "step") | `"magnus2"` | – |
| `propagator.pc_convergence` | float | Predictor-corrector convergence threshold (Magnus2 only) | 1e-12 | a.u. |
| `propagator.max_iterations` | int | Maximum predictor-corrector iterations (Magnus2 only) | 200 | – |
| `hermiticity_tolerance` | float | Tolerance used when checking that matrices are Hermitian | 1e-12 | a.u. |

### 3.3 Quantum Source (pure quantum runs only)

When running a standalone RT-TDDFT simulation (no `"plasmon"` section), provide an incident electric field via this block. Field-free sudden core-hole runs may omit it (the drive is then a zero field).

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
| ----- | ------ | --------- | ---- | ------------- | ------- |
| `type` | str | All | Shape of the external field ("pulse" or "kick") | – | – |
| `intensity` | float | All | Peak electric-field strength | – | a.u. |
| `peak_time` | float | All | Time at which the pulse/kick reaches maximum | – | a.u. |
| `width_steps` | int | All | Width of the pulse in number of time steps | – | – |
| `component` | str | All | Direction of the electric field ("x", "y", or "z") | – | – |
| `additional_parameters.wavelength` | float | `pulse` | Central wavelength of the pulse | – | μm |
| `additional_parameters.frequency` | float | `pulse` | Central frequency of the pulse (alternative to wavelength) | – | 1/a.u. |

For absorption spectra use `"type": "kick"` together with `"driver": {"name": "absorption", ...}` (or the string `"absorption"` plus defaults).

### 3.4 CAP (Lopata-style)

Optional energy-dependent imaginary potential added to the Fock matrix for lifetime effects and smoother spectra. See the [Lopata paper](https://pubs.acs.org/doi/abs/10.1021/ct400569s) for more details.

```json
{
  "cap": {
    "type": "static",
    "gam0": 1.0,
    "xi": 0.5,
    "eps0": 0.0477,
    "clamp": 100
  }
}
```

| Key | Type | Description | Default | Units |
| ----- | ------ | --------- | ------------- | ------- |
| `type` | str | `"static"` or `"dynamic"` CAP | `"static"` | – |
| `gam0` | float | Base CAP strength | 1.0 | a.u. |
| `xi` | float | Exponent controlling energy dependence of CAP | 0.5 | – |
| `eps0` | float or `"tune"` | Reference energy (vacuum level); use `"tune"` for automatic estimation | 0.05 | a.u. |
| `clamp` | float | Maximum allowed CAP value | 100 | a.u. |

### 3.5 Core-hole (`molecule.core_hole`)

Optional sudden SCH/DCH on a **quantum**, **absorption**, or **plasmol** run. The dedicated `"driver": "core_hole"` workflow does **not** use this for propagation — it only surveys which atoms contribute to listed MOs. See [Core-Hole Dynamics](core_hole.md).

```json
{
  "core_hole": {
    "mo_removal_index_dict": {"0": 2},
    "mo_occ_filepath": "mo_occ.csv",
    "watch_indices": [0, 1, 2, 3],
    "filter_by_amplitude": false,
    "amplitude_threshold": 0.2
  }
}
```

| Key | Type | Description | Default |
| ----- | ------ | ------------- | --------- |
| `mo_removal_index_dict` | dict | 0-based MO index → electrons to remove (1 or 2). `{i:1}` SCH; `{i:2}` DCH; `{i:1,j:1}` two SCH | required |
| `mo_occ_filepath` | str | CSV path for time-dependent hole occupations | required (production) |
| `watch_indices` | list | MO indices to plot (logging is 0…LUMO+1) | all logged |
| `filter_by_amplitude` | bool | Filter plot by peak-to-peak amplitude | false |
| `amplitude_threshold` | float | Amplitude cutoff when filtering | 0.2 |

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
| ----- | ------ | --------- | ------------- | ------- |
| `checkpoint.frequency_steps` | int | Number of time steps between checkpoint saves | – | – |
| `checkpoint.frequency_time` | float | Amount of simulation time between checkpoint saves (alternative to frequency_steps) | – | a.u. |
| `checkpoint.filepath` | str | Path to the `.npz` checkpoint file | – | – |
| `field_e_filepath` | str | CSV file for the electric field felt by the molecule | `"field_e.csv"` | – |
| `field_p_filepath` | str | CSV file for the induced dipole (polarization) of the molecule | `"field_p.csv"` | – |
| `spectra_e_vs_p_filepath` | str | PNG file showing incident field vs. molecular response | auto-timestamped | – |

**Note**: Checkpointing is only supported for pure quantum (molecule-only) simulations. Use either `frequency_steps` **or** `frequency_time`, not both. Resume workflow, what is stored, and what is **not** supported: [Checkpointing](checkpointing.md).

## 5. Driver keys and `"additional_parameters"`

Workflow-specific options belong on `settings.driver` when that field is a dict (`"name"` plus keys below). A string `"driver": "absorption"` is the same as `{"name": "absorption"}` with defaults.

The top-level `"additional_parameters"` object is only for **plasmon-wide** flags (`decay_stop`, `decay_threshold`) and the checkpoint-resume injection `checkpoint_filename_used`. Older nested blocks (`additional_parameters.absorption`, …) are still accepted and merged onto `settings.driver` with a warning. Flat core-hole keys (`mo_removal_index_dict`, `core_hole_mo_occ_filepath`, …) are moved onto `molecule.core_hole`.

### 5.1 Absorption (`"name": "absorption"`)

Runs directional trajectories and Fourier-transforms the induced dipole. Hybrid spectra \(\sigma_m\), \(A_{\mathrm{diss}}\), and \(A_{\mathrm{raw}}\) are derived in [Hybrid absorption observables](observables.md).

```json
{
  "name": "absorption",
  "gamma": 0.01,
  "min_ev": 1.5,
  "max_ev": 5.0,
  "spectrum_filepath": "spectrum.png",
  "npz_filepath": null,
  "tau": null,
  "observables": ["cross_section"]
}
```

| Key | Type | Description | Default | Units |
| ----- | ------ | --------- | ------------- | ------- |
| `gamma` | float | Broadening (damping) factor applied before FFT | – | a.u. |
| `min_ev` | float | Lower energy limit of the plotted spectrum | 1.5 | eV |
| `max_ev` | float | Upper energy limit of the plotted spectrum | 5.0 | eV |
| `spectrum_filepath` | str | Output PNG (and `.csv`) for the spectrum. With several `observables`, files are `{stem}_{name}.png` | – | – |
| `observables` | list of str | Spectra to write: `cross_section` (σ_m), `dissipative_power` (A_diss), `A_raw` (μ/E_inc) | `["cross_section"]` | – |
| `npz_filepath` | str | Optional `.npz` file containing raw Fourier data | – | – |
| `tau` | float | Extra artificial damping time constant tau (signal *= exp(-t/tau)) applied to time-domain polarization before FFT | – | a.u. |
| `polarization` | str | `full` (x+y+z, **no NP**), `parallel` (E along NP–mol axis), `perpendicular`, or `single` (JSON source as given). `full` is rejected when a nanoparticle is present. | `full` (molecule-only) | – |
| `perp_component` | str | Optional `x`/`y`/`z` for perpendicular mode | auto | – |
| `field_e_ref_filepath` | str | Vacuum \(E_{inc}\) CSV (`time,xx,yy,zz`) | `field_e_ref.csv` | – |
| `use_existing_e_field_ref` | bool | Skip vacuum Meep runs when reference file exists | auto | – |
| `reference_only` | bool | Only build vacuum references and exit (`full` only; no nanoparticle) | false | – |

### 5.2 Comparison (`"name": "comparison"`)

Ground-state SCF across basis sets / XC functionals; MO energy plots.

```json
{
  "name": "comparison",
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
```

| Key | Type | Description | Default | Units |
| ----- | ------ | --------- | ------------- | ------- |
| `bases` | list of str | List of basis sets to compare | – | – |
| `xcs` | list of str | List of exchange-correlation functionals to compare | – | – |
| `lrc_parameters` | dict | Mapping of XC name → μ value for range-separated hybrids | – | – |
| `num_occupied` | int | Number of occupied orbitals to display | – | – |
| `num_virtual` | int | Number of virtual orbitals to display | – | – |
| `y_min` / `y_max` | float | Energy axis limits (Hartree) | – | Ha |
| `index_min` / `index_max` | int | MO index range to display (1-based) | – | – |
| `dir_name` | str | Directory in which comparison plots are saved | auto-timestamped | – |

### 5.3 Scatter response (`"name": "scatter_response_fxn"`)

`probe_points` is a list of `[x, y, z]` sample locations in μm:

```json
{
  "name": "scatter_response_fxn",
  "probe_points": [[0.011, 0, 0], [0.012, 0, 0]]
}
```

### 5.4 NP absorption cross-section (`"name": "np_abs_cross_sec"`)

| Key | Type | Description | Default | Units |
| ----- | ------ | --------- | ------------- | ------- |
| `n_flux_freqs` | int | Frequency samples on the flux monitors | 50 | – |
| `flux_padding` | float | Extra radius around the NP for the flux box | 0.005 | μm |
| `line_fit` | bool | Lorentzian fit of the efficiency spectrum | false | – |

`decay_stop` / `decay_threshold` may be set on the driver dict or under `additional_parameters` (plasmon-wide).

### 5.5 Core-hole survey (`"name": "core_hole"`)

This driver has **no extra keys**. It always surveys per-atom contributions for MOs in `molecule.core_hole.mo_removal_index_dict` and exits (no ionization, no time loop). Sudden SCH/DCH belongs under `molecule.core_hole` (section 3.5) with driver `quantum` or `absorption`.

See [Core-Hole Dynamics](core_hole.md).

### 5.6 Drivers with no extra keys

`classical`, `quantum`, `plasmol`, `tune`, `verify_source`, and `core_hole` take a string (or `{"name": "..."}` with no other keys):

```json
{
  "settings": {
    "dt": 0.1,
    "t_end": 100,
    "driver": "verify_source"
  },
  "plasmon": {
    "simulation": {
      "cell_length": 0.2,
      "pml_thickness": 0.05,
      "surrounding_material_index": 1.0
    },
    "source": {
      "type": "gaussian",
      "center": [0.0, 0.0, 0.0],
      "size": [0.0, 0.2, 0.2],
      "component": "z",
      "amplitude": 1.0,
      "is_integrated": true,
      "additional_parameters": {
        "frequency": 2.29,
        "fwidth": 2.08
      }
    }
  }
}
```

---

*This document reflects PlasMol v1.2.0 JSON input format.*
