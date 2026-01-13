# Usage

PlasMol is designed to be run from the command line, with most parameters specified in a single input file. This page explains how to create and structure an input file, as well as details on each supported block and parameter. For step-by-step examples, see the [Tutorials](tutorials.md) page.

## General Process for Creating an Input File

1. **Determine Simulation Type**: PlasMol supports three modes based on your input file:
    - **Classical Only**: Simulate a nanoparticle (NP) using FDTD (via MEEP). Include a `classical` block but no `quantum` block.
    - **Quantum Only**: Simulate a molecule using RT-TDDFT (via PySCF). Include a `quantum` block but no `classical` block.
    - **PlasMol (Full)**: Simulate NP-molecule interactions. Include both `classical` and `quantum` blocks.

2. **Structure the Input File**: The file uses a block-based format with `start` and `end` keywords. Comments start with `#`, `--`, or `%`. Sections can be nested.
    - Always start with a `general` or `settings` block (these two keywords are interchangeable) for shared parameters (e.g., time step, end time).
    - Add `classical` and/or `quantum` blocks as needed.
    - Use templates from the `templates/` directory as starting points:
        - [template-classical.in](https://github.com/kombatEldridge/PlasMol/blob/cf3ee73f07f7ae5be5b0de6c1b8a8f9e7913c45b/templates/template-classical.in)
        - [template-quantum.in](https://github.com/kombatEldridge/PlasMol/blob/cf3ee73f07f7ae5be5b0de6c1b8a8f9e7913c45b/templates/template-quantum.in)
        - [template-plasmol.in](https://github.com/kombatEldridge/PlasMol/blob/cf3ee73f07f7ae5be5b0de6c1b8a8f9e7913c45b/templates/template-plasmol.in)

3. **Run PlasMol**: From the command line:

    ```bash
    python plasmol/main.py -f /path/to/plasmol.in -vv -l plasmol.log -r
    ```

    or

    ```bash
    python -m plasmol.main -f /path/to/plasmol.in -vv -l plasmol.log -r
    ```

    - Common options: `-v` (verbose), `-vv` (debug), `-l log.txt` (log file), `-r` (restart, deletes old output files).
    - See `plasmol/utils/input/cli.py` or run `python plasmol/main.py --help` for full options.

4. **Output Files**: PlasMol can generate CSVs (e.g., `eField.csv`, `pField.csv`), images (e.g., via HDF5 or plots), and checkpoint files (`.npz`). For details, see the `files` block below.

5. **Units and Conventions**: Times are in atomic units (au) unless specified. Coordinates are in Bohr or Angstrom (specify in `units`). Electric fields are in au.

6. **Customization**: For advanced tweaks (e.g., custom sources or propagators), modify the code as noted in [API Reference](api-reference.md).

## Input Blocks and Parameters

Below are details on each block, drawn from the code and templates. Parameters are listed with defaults (if any), types, and descriptions. Required parameters are marked with *.

### General/Settings Block

Shared simulation parameters. Use `start general` or `start settings` (interchangeable). This block is required.

- `dt`*: Float. Time step in au (e.g., `0.001`).
- `t_end`*: Float. End time in au (e.g., `40`).
- `eField_path`: String. Path to output CSV for electric field (e.g., `eField.csv`).

Example:

```lua
start general
    dt 0.001
    t_end 40
    eField_path eField.csv
end general
```

### Classical Block

For NP simulations (FDTD via MEEP). Required for classical or full PlasMol modes.

#### Source Sub-Block

Defines the incident electric field. Required for most simulations. For more information about these sources, visit [MEEP's documentation](https://meep.readthedocs.io/en/master/Python_User_Interface/#source).

- `sourceType`*: String. One of: `continuous`, `gaussian`, `chirped`, `pulse`.
- `sourceCenter`*: Float or list (microns) (e.g., `-0.04` or `-0.04 0 0`). Center position. If one component given, assumed in the 'x' direction.
- `sourceSize`*: List of 3 floats (microns) (e.g., `0 0.1 0.1`). Size dimensions.
- `frequency`: Float. Frequency (au); mutually exclusive with `wavelength`.
- `wavelength`: Float. Wavelength (microns); mutually exclusive with `frequency`.
- `width`: Float (default 0). Gaussian width.
- `fwidth`: Float (default inf). Frequency width (alternative to `width`).
- `start_time`: Float (default 0 or -inf). Start time.
- `end_time`: Float (default inf). End time.
- `cutoff`: Float (default 5.0, for Gaussian). Truncation cutoff.
- `slowness`: Float (default 3.0, for Continuous). Ramp-up slowness.
- `peakTime`: Float (for Chirped/Pulse). Peak time.
- `chirpRate`: Float (for Chirped). Chirp rate.
- `is_integrated`: Boolean (default True). Integrate source over time.
- `component`: String (default 'z'). Field component ('x', 'y', 'z').

Example (Continuous):

```lua
start source
    sourceType continuous
    sourceCenter -0.04
    sourceSize 0 0.1 0.1
    frequency 5
    isIntegrated True
end source
```

#### Simulation Sub-Block

Core FDTD parameters. For more information about these parameters, visit [MEEP's documentation](https://meep.readthedocs.io/en/master/).

- `cellLength`*: Float (e.g., `0.1`). Simulation cell size.
- `pmlThickness`*: Float (e.g., `0.01`). PML boundary thickness.
- `eFieldCutOff`: Float (e.g., `1e-12`). Threshold to trigger quantum propagation. Ignored if no quantum blocked defined.
- `symmetries`: List (e.g., `Y 1 Z -1`). Mirror symmetries (X/Y/Z with phase ±1).
- `surroundingMaterialIndex`: Float (default 1.0). Refractive index of medium.
- `resolution`: Float (optional). Spatial resolution; auto-calculated from `dt` if omitted.

Example:

```lua
start simulation
    cellLength 0.1
    pmlThickness 0.01
    eFieldCutOff 1e-12
    symmetries Y 1 Z -1
    surroundingMaterialIndex 1.33
end simulation
```

#### Object Sub-Block

Defines the NP (currently supports spheres only).

- `material`*: String. Either `Au` or `Ag`. Material type.
- `radius`*: Float (microns) (e.g., `0.03`). Sphere radius.
- `center`*: List of 3 floats (e.g., `0 0 0`). Center position.

Example:

```lua
start object
    material Au
    radius 0.03
    center 0 0 0
end object
```

#### HDF5 Sub-Block

For generating 2D cross-section images of the simulation.

- `timestepsBetween`*: Integer (e.g., `1`). Interval for image output.
- `intensityMin`*: Float (e.g., `3`). Min intensity for color scale.
- `intensityMax`*: Float (e.g., `10`). Max intensity for color scale.
- `imageDirName`: String (optional, auto-generated if omitted). Output directory.

Example:

```lua
start hdf5
    timestepsBetween 1
    intensityMin 3
    intensityMax 10
    imageDirName images
end hdf5
```

#### Molecule Sub-Block

Places a molecule in the simulation (required for full PlasMol).

- `center`*: List of 3 floats (microns) (e.g., `0 0 0`). Molecule position.

Example:

```lua
start molecule
    center 0 0 0
end molecule
```

### Quantum Block

For molecule simulations (RT-TDDFT). Required for quantum or full PlasMol modes.

#### RTTDDFT Sub-Block

Core quantum parameters.

- Geometry sub-sub-block*:
  - Atom coordinates. Inline after `start geometry`.
- `units`*: String (`bohr` or `angstrom`).
- `check_tolerance`: Float (default 1e-12). Tolerance for matrix checks for hermiticity.
- `charge`*: Integer (default 0).
- `spin`*: Integer (default 0).
- `basis`*: String (e.g., `6-31g`).
- `xc`*: String (e.g., `pbe0`). Exchange-correlation functional.
- `propagator`*: String, one of `step`, `rk4`, or `magnus2`.
- `pc_convergence`: Float (only for magnus2, e.g., `1e-12`).
- `maxiter`: Integer (only for magnus2, e.g., `200`).
- `transform`: (No value needed). Enables absorption spectrum calculation. See the [API Reference](api-reference.md#quantumpy) and [Tutorial #3](tutorials.md#tutorial-3-molecular-absorption-spectrum-rt-tddft-with-transform-flag) for more details.

Example:

```lua
start rttddft
    start geometry
        O 0.0 0.0 -0.13
        H 1.49 0.0 1.03
        H -1.49 0.0 1.03
    end geometry
    units bohr
    basis 6-31g
    xc pbe0
    propagator magnus2
    transform
end rttddft
```

#### Files Sub-Block

Output paths.

- `checkpoint` sub-sub-block:
  - `frequency` int (e.g., `100`). Number of time steps between checkpoints.
  - `path` string (e.g., `checkpoint.npz`). Path to checkpoint file.
- `pField_path`: String (e.g., `pField.csv`). Polarization field CSV.
- `eField_vs_pField_path`: String (e.g., `output.png`). Plot of fields.
- `pField_Transform_path`: String (e.g., `pField-transformed.npz`). Transformed data (for spectrum).
- `eV_spectrum_path`: String (e.g., `spectrum.png`). Absorption spectrum plot.

Example:

```lua
start files
    start checkpoint
        frequency 100
        path checkpoint.npz
    end checkpoint
    pField_path pField.csv
end files
```

#### Source Sub-Block (Quantum-Only)

Electric field for standalone quantum simulations.

- `shape`*: String (`pulse` or `kick`).
- `wavelength_nm`: Float (for pulse).
- `peak_time_au`: Float.
- `width_steps`: Integer.
- `intensity_au`: Float.
- `dir`: String (`x`, `y`, `z`).

Example:

```lua
start source
    shape pulse
    wavelength_nm 500
    peak_time_au 0.1
    width_steps 5
    intensity_au 5e-5
    dir z
end source
```

For more details on code internals, see [API Reference](api-reference.md).
