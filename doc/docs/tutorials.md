# Tutorials

!!! danger "this page is a WIP"
    I still need to go through these tutorials and add data and check the template files. 



## Tutorial 1: Classical Nanoparticle Simulation (FDTD Only)

Simulate a gold sphere in water interacting with a continuous-wave source. Produces field CSVs and optional PNG/GIF frames.

**JSON input** (`classical.json`):

```json
{
  "settings": {
    "dt": 0.1,
    "t_end": 50
  },
  "plasmon": {
    "simulation": {
      "cell_length": 0.1,
      "pml_thickness": 0.01,
      "surrounding_material_index": 1.33,
      "symmetries": ["Y", 1, "Z", -1]
    },
    "source": {
      "type": "continuous",
      "center": [-0.04, 0, 0],
      "size": [0, 0.1, 0.1],
      "component": "z",
      "additional_parameters": {
        "frequency": 5.0
      }
    },
    "nanoparticle": {
      "material": "Au_JC_visible",
      "radius": 0.03,
      "center": [0, 0, 0]
    },
    "images": {
      "timesteps_between": 2,
      "dir_name": "classical_frames",
      "make_gif": true
    }
  },
  "files": {
    "field_e_filepath": "field_e.csv"
  }
}
```

**Run**:
```bash
python -m plasmol.main -f classical.json -vv -l classical.log
```

**Outputs**:

- `field_e.csv` — Electric field time series at origin (or probe points if added).
- `classical_frames/` + `classical_frames.gif` — 2D slices of |Ez|.

You can add `probe_points` under `additional_parameters` and use the `scatter_response_fxn` driver for more advanced post-processing (see custom drivers).

---

## Tutorial 2: Quantum RT-TDDFT — Induced Dipole of a Molecule

Compute the time-dependent induced dipole of a water molecule under a pulsed electric field.

**JSON input** (`quantum_pulse.json`):

```json
{
  "settings": {
    "dt": 0.05,
    "t_end": 1000
  },
  "molecule": {
    "geometry": [
      {"atom": "O", "coord": [0.0, 0.0, -0.1302]},
      {"atom": "H", "coord": [1.4891, 0.0, 1.0332]},
      {"atom": "H", "coord": [-1.4891, 0.0, 1.0332]}
    ],
    "geometry_units": "bohr",
    "basis": "6-31g",
    "xc": "pbe0",
    "charge": 0,
    "spin": 0,
    "propagator": {
      "type": "magnus2",
      "pc_convergence": 1e-12,
      "max_iterations": 200
    },
    "source": {
      "type": "pulse",
      "intensity": 5e-5,
      "peak_time": 200,
      "width_steps": 1000,
      "component": "z",
      "additional_parameters": {
        "wavelength": 0.4
      }
    }
  },
  "files": {
    "field_e_filepath": "field_e.csv",
    "field_p_filepath": "field_p.csv",
    "spectra_e_vs_p_filepath": "field_vs_polarization.png"
  }
}
```

**Run**:
```bash
python -m plasmol.main -f quantum_pulse.json -vv -l quantum.log
```

**Outputs**:

- `field_e.csv` + `field_p.csv` — Incident field and induced dipole (polarization) vs time.
- `field_vs_polarization.png` — Side-by-side plot (generated automatically).

---

## Tutorial 3: Molecular Absorption Spectrum (Fourier Workflow)

Compute the absorption spectrum of water using three directional delta-kick simulations + Fourier transform. This is the recommended way to obtain spectra.

**JSON input** (`absorption_spectrum.json`):

```json
{
  "settings": {
    "dt": 0.1,
    "t_end": 4000
  },
  "molecule": {
    "geometry": [ ... same water geometry ... ],
    "geometry_units": "bohr",
    "basis": "6-31g",
    "xc": "pbe0",
    "propagator": {"type": "magnus2"},
    "source": {
      "type": "kick",
      "intensity": 5e-5,
      "peak_time": 0.1,
      "width_steps": 5,
      "component": "z"
    }
  },
  "files": {
    "field_e_filepath": "field_e.csv",
    "field_p_filepath": "field_p.csv",
    "spectra_e_vs_p_filepath": "raw_response.png"
  },
  "additional_parameters": {
    "fourier": {
      "gamma": 0.005,
      "min_ev": 1.5,
      "max_ev": 12.0,
      "spectrum_filepath": "water_absorption_spectrum.png",
      "tau": 0.01
    }
  }
}
```

**Run**:
```bash
python -m plasmol.main -f absorption_spectrum.json -vv -l spectrum.log
```

PlasMol automatically runs **three parallel simulations** (x/y/z kicks), applies damping, performs the FFT, and produces a normalized absorption spectrum.

**Outputs**:

- `x_dir/`, `y_dir/`, `z_dir/` subdirectories with per-direction CSVs.
- `water_absorption_spectrum.png` — Final absorption spectrum (eV vs. intensity).
- Optional `.npz` file with raw Fourier data.

---

## Tutorial 4: Full Hybrid PlasMol Simulation (NP + Molecule)

Gold nanoparticle + water molecule inside the FDTD grid. The molecule feels the local field; its induced dipole is fed back into Meep.

**JSON input** (`hybrid.json`):

```json
{
  "settings": {
    "dt": 0.1,
    "t_end": 2000
  },
  "plasmon": {
    "simulation": {
      "cell_length": 0.12,
      "pml_thickness": 0.015,
      "tolerance_field_e": 1e-12,
      "surrounding_material_index": 1.33
    },
    "source": {
      "type": "gaussian",
      "center": [-0.05, 0, 0],
      "size": [0, 0.08, 0.08],
      "component": "z",
      "additional_parameters": {
        "frequency": 2.5,
        "width": 0.8
      }
    },
    "nanoparticle": {
      "material": "Au_JC_visible",
      "radius": 0.025,
      "center": [0, 0, 0]
    },
    "molecule": {
      "position": [0.035, 0, 0],
      "back_propagation": true
    },
    "images": {
      "timesteps_between": 5,
      "dir_name": "hybrid_frames",
      "make_gif": true
    }
  },
  "molecule": {
    "geometry": [ ... water ... ],
    "geometry_units": "bohr",
    "basis": "6-31g",
    "xc": "pbe0",
    "propagator": {"type": "magnus2"}
  },
  "files": {
    "field_e_filepath": "field_e.csv",
    "field_p_filepath": "field_p.csv",
    "spectra_e_vs_p_filepath": "hybrid_response.png"
  }
}
```

**Run**:
```bash
python -m plasmol.main -f hybrid.json -vv -l hybrid.log
```

**What happens internally**:

1. Meep starts with the Au sphere and incident source.
2. Every time step, if |E| at molecule position > tolerance, the quantum propagator is called.
3. Induced dipole is stored and injected back into Meep as a CustomSource (point dipole).
4. Both `field_e.csv` (local field felt by molecule) and `field_p.csv` (molecular response) are written.

This is the core capability of PlasMol for studying plasmon-enhanced phenomena (SERS, energy transfer, etc.).

---

## Tutorial 5: Molecular Orbital Energy Comparison

Quickly compare HOMO/LUMO and orbital energies across multiple basis sets and functionals (very useful for method benchmarking).

**JSON input** (`mo_comparison.json`):

```json
{
  "settings": {
    "dt": 0.1,
    "t_end": 10
  },
  "molecule": {
    "geometry": [ ... water or any molecule ... ],
    "geometry_units": "bohr",
    "basis": "6-31g",   // will be overridden by comparison
    "xc": "pbe0",
    "propagator": {"type": "magnus2"}
  },
  "additional_parameters": {
    "comparison": {
      "bases": ["6-31g", "6-31g*", "def2-svp", "aug-cc-pvdz"],
      "xcs": ["pbe0", "b3lyp", "cam-b3lyp"],
      "lrc_parameters": {"cam-b3lyp": 0.33},
      "num_occupied": 5,
      "num_virtual": 8,
      "y_min": -0.8,
      "y_max": 0.6,
      "dir_name": "mo_comparison"
    }
  }
}
```

**Run**:
```bash
python -m plasmol.main -f mo_comparison.json -vv
```

**Outputs**:

- `mo_comparison/individuals/` — One PNG per (basis, xc) pair.
- `mo_comparison/all_mo_energies.png` — Beautiful grid plot with HOMO/LUMO annotations and color-coded background (red = negative virtual orbitals, yellow = near-zero LUMO, green = healthy).

The comparison driver only performs ground-state SCF calculations — no time propagation is needed.

---

## Tutorial 6: Nanoparticle Absorption & Scattering Cross-Sections

Use the dedicated `np_abs_cross_sec` driver to compute absorption, scattering, and extinction efficiencies of a nanoparticle (Mie-type calculation with flux boxes).

Add to your JSON:

```json
{
  "settings": {
    "driver": "np_abs_cross_sec",
    ...
  },
  "plasmon": {
    "nanoparticle": { ... },
    "source": {
      "type": "gaussian",
      "additional_parameters": {
        "frequency": 2.0,
        "fwidth": 1.5
      }
    },
    ...
  }
}
```

Then run with that driver. The script produces `output_arrays.txt`, efficiency plots, and a multi-peak Lorentzian fit of the plasmon resonance.

Similar workflow exists for the full `plasmol_abs_cross_sec` driver (hybrid NP + molecule cross-sections).

---

## Advanced / Custom Workflows

- **Chen 2010 replication** (`driver: "scatter_response_fxn"`): Runs four parallel FDTD simulations (X/Y pol × NP/VAC) and saves probe-point field data for post-processing of λ-dependent response.
- **Adding custom observables**: In `quantum/molecule.py` add a new method (e.g. `calculate_sers_enhancement()`), then call it inside `quantum/propagation.py` after each step. The result can be written to CSV.
- **New electric field shapes**: Add to `classical/sources.py` (MEEPSOURCE) or `quantum/sources.py` (QUANTUMSOURCE) and register in the JSON schema via `params.py`.
- **New propagators**: Implement in `quantum/propagators/`, add to the map in `params.py`, and update validation.

All extension points are documented with comments in the source code.

These tutorials cover the vast majority of use cases. For the complete parameter reference, see [Usage](usage.md) or run `--describe`. Happy simulating!