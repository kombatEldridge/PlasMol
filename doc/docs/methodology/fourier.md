# Fourier Absorption Spectra

The **Fourier driver** (`settings.driver: "fourier"` or a `fourier` block under `additional_parameters`) is the recommended workflow for linear absorption spectra in PlasMol. A broadband excitation drives the molecule; the induced dipole \(\boldsymbol{\mu}(t)\) is Fourier-transformed and assembled into a peak-normalized spectrum.

This page covers:

1. Shared post-processing (fold, damp, FFT, absorption)
2. Quantum-only δ-kick path
3. Hybrid Meep path with vacuum \(E_{\mathrm{inc}}\) deconvolution (**full** polarization)
4. **Parallel and perpendicular modes for full hybrid PlasMol runs** (orientation-resolved)

## Physical overview

### Induced dipole

\[
\mu_i(t)
=

\operatorname{Tr}\!\big[
  \hat{\mu}_i \big(\mathbf{D}(t) - \mathbf{D}_0\big)
\big],
\qquad i \in \{x,y,z\}.
\]

Open-shell (UKS) runs sum α and β density blocks before the trace.

### Absorption-like spectrum

\[
A(\omega)
=

-\frac{4\pi\omega}{3c}
\sum_{i\in\{x,y,z\}}
\operatorname{Im}\!\big[\mathcal{R}_i(\omega)\big]
\quad\text{(isotropic / full)},
\]

with \(c = \texttt{constants.C_AU}\). For a single polarization \(i \in \{x,y,z\}\),

\[
A_i(\omega)
=

-\frac{4\pi\omega}{c}\,
\operatorname{Im}\!\big[\mathcal{R}_i(\omega)\big].
\]

Because a delta-like kick is difficult to do in MEEP, \(\mathcal{R}_i\) varies. It is either \(\mu_i(\omega)\) (kick within RT-TDDFT only sims) or \(\mu_i(\omega)/E_{\mathrm{inc},i}(\omega)\) (gaussian broadband pulse within full PlasMol sims). Curves are peak-normalized before plotting.

Frequencies come from the real FFT grid, converted to eV via \(E[\mathrm{eV}] = \omega[\mathrm{a.u.}]\times 27.211386\), and clipped to `min_ev`–`max_ev`.

## Common post-processing

1. **Damping** — optional \(\gamma\) window \(e^{-\gamma t}\) inside the FFT path; optional \(\tau\) factor \(e^{-t/\tau}\) on polarization (and matching \(E_{\mathrm{inc}}\) in memory when deconvolving).
2. **FFT** — \(S_{\mu,i}(\omega) = \Delta t\sum_n \mu_i(t_n)\,w(t_n)\,e^{-i\omega t_n}\).
3. **Outputs** — `spectrum_filepath` PNG + sibling CSV; optional `npz_filepath`; working dirs `x_dir/`, `y_dir/`, `z_dir/`.

## Quantum-only path (δ-kicks)

With a `molecule` section and **no** `plasmon` section, the driver launches three RT-TDDFT jobs (x, y, z kicks). A true δ-kick has a flat spectrum, so

\[
\mathcal{R}_i(\omega) = \mu_i(\omega).
\]

```json
{
  "settings": { "dt": 0.05, "t_end": 400, "driver": "fourier" },
  "molecule": {
    "geometry": [ /* ... */ ],
    "source": {
      "type": "kick",
      "intensity": 1e-4,
      "peak_time": 0.0,
      "width_steps": 1
    }
  },
  "additional_parameters": {
    "fourier": {
      "gamma": 0.01,
      "min_ev": 1.5,
      "max_ev": 10.0,
      "spectrum_filepath": "spectrum.png"
    }
  }
}
```

## Hybrid Meep path — full polarization (default)

When both `plasmon` and `molecule` are present (and `polarization` is `"full"` or omitted):

1. **Production runs** (x, y, z): Meep cell with NP (optional) and molecule; record \(\mu(t)\) (and local \(\mathbf{E}\)).
2. **Vacuum reference runs**: same source and cell geometry, **no NP, no molecule**, `record_field_only` → incident field \(E_{\mathrm{inc}}(t)\) at the molecule site.
3. **Deconvolution**

\[
\mathcal{R}_i(\omega)
=

\frac{\mu_i(\omega)}{E_{\mathrm{inc},i}(\omega)}
\]

with a spectral floor on \(|E|\) to avoid division noise. Dividing by bare \(E_{\mathrm{inc}}\) removes the source envelope; **local-field and NP enhancement remain in \(\mu\)**, so \(\mathcal{R}\) is an _effective_ polarizability-like response—not the bare \(\alpha_m\). See [Effective Polarizability](effective_polarizability.md).

### Reference file options

| Key | Role |
| ----- | ------ |
| `field_e_ref_filepath` | Merged vacuum \(E_{\mathrm{inc}}\) CSV (`time,xx,yy,zz`); default `field_e_ref.csv` |
| `use_existing_e_field_ref` | Skip vacuum Meep runs if the reference file already exists |
| `reference_only` | Only build the three-direction vacuum reference and exit (incompatible with ∥/⊥) |

---

## Parallel and perpendicular methodology for full PlasMol runs

Near a spherical nanoparticle the hybrid response is **anisotropic**: radial (∥) and tangential (⊥) polarizations couple differently to the plasmon. Averaging x+y+z mixes inequivalent axes once an NP–molecule geometry is fixed. PlasMol therefore supports orientation-resolved hybrid Fourier spectra:

```json
"fourier": {
  "polarization": "parallel",
  "spectrum_filepath": "spectrum_par.png"
}
```

or

```json
"fourier": {
  "polarization": "perpendicular",
  "perp_component": "z",
  "spectrum_filepath": "spectrum_perp.png"
}
```

Allowed values: `"full"` (default), `"parallel"`, `"perpendicular"`.

### Motivation

| Mode | Physics | Cost |
| ------ | --------- | ------ |
| `full` | Isotropic average over three Cartesian drives | ~3 production + up to 3 vacuum refs |
| `parallel` | \(\mathbf{E}\) along NP→molecule axis | 1 production + 1 ref |
| `perpendicular` | \(\mathbf{E}\) ⊥ NP→molecule axis | 1 production + 1 ref |

Parallel/perpendicular maps naturally onto Gersten–Nitzan \(G_\parallel,S_\parallel\) vs \(G_\perp,S_\perp\) (see [Effective Polarizability](effective_polarizability.md)).

### Geometry and axis definition

- Nanoparticle center: `plasmon.nanoparticle.center` (origin if no NP)
- Molecule sample point: `plasmon.molecule.position` (**required** for ∥/⊥)
- Axis \(\mathbf{r} = \mathbf{r}_{\mathrm{mol}} - \mathbf{r}_{\mathrm{NP}}\)

**Parallel component** — Cartesian direction nearest to \(\mathbf{r}\) (`resolve_parallel_component`). A warning is logged if \(|\hat{\mathbf{u}}\cdot\hat{\mathbf{e}}| < 0.9\) (molecule not well aligned with a lab axis).

**Perpendicular component** — either user `perp_component` (`x`/`y`/`z`) or the Cartesian axis most orthogonal to \(\mathbf{r}\). A user choice nearly parallel to \(\mathbf{r}\) is rejected.

**Validation:** `polarization` ∈ {`parallel`,`perpendicular`} requires a `plasmon` section and `plasmon.molecule.position`. It is incompatible with `reference_only`.

### Source geometry (plane-wave face)

Hybrid Fourier sources are planar faces meant to represent a plane wave with **propagation \(\mathbf{k}\) perpendicular to \(\mathbf{E}\)**. Before each run PlasMol calls `ensure_transverse_plane_wave_source`:

- Inspects source `center`, `size`, and `component`
- If the face normal is parallel to \(\mathbf{E}\) (longitudinal), the size/component are rearranged so \(\mathbf{k}\perp\mathbf{E}\)
- Optional preference to put \(\mathbf{k}\) along \(z\) when rearranging

This keeps Meep’s drive consistent with a transverse plane wave for the chosen polarization.

### Run construction

**Parallel / perpendicular** (implemented in `build_parallel_abs_spec_runs` / `build_perpendicular_abs_spec_runs`):

1. Resolve Cartesian component \(c\).
2. Build one **production** params copy: hybrid plasmol in `{c}_dir/`, source component \(c\), transverse face enforced.
3. Unless `use_existing_e_field_ref`, build one **vacuum reference** copy: no NP, no molecule, `record_field_only=True`, same source geometry, \(E\) sampled at the molecule position.
4. Workers run production (and ref); post-process with `fourier_post_process_single`.

**Contrast with `full`:** three production directories `x_dir`, `y_dir`, `z_dir` and up to three vacuum references, isotropic absorption.

### Hybrid field path during production

Each Meep step (when \(|\mathbf{E}|\) exceeds `tolerance_field_e`):

1. Sample \(\mathbf{E}\) at `plasmon.molecule.position`
2. Advance RT-TDDFT one step
3. Optionally inject \(\boldsymbol{\mu}\) back into Meep (`back_propagation`)

For the spectrum, only the **recorded** \(\mu_c(t)\) and vacuum \(E_{\mathrm{inc},c}(t)\) enter deconvolution. Local-field physics is therefore encoded in \(\mu\), not in the denominator.

### Single-polarization post-processing

\[
\mathcal{R}_c(\omega)
=

\frac{\mu_c(\omega)}{E_{\mathrm{inc},c}(\omega)},
\qquad
A_c(\omega)
=

-\frac{4\pi\omega}{c}\,\mathrm{Im}\,\mathcal{R}_c(\omega).
\]

`fourier_post_process_single` can rebuild a spectrum offline from existing `field_p.csv` and a raw or merged reference CSV (useful for reprocessing without re-running Meep).

### Relation to the classical model and fits

PlasMol `parallel` / `perpendicular` hybrid spectra are the data behind the joint Gersten–Nitzan fit in [Effective Polarizability](effective_polarizability.md):

- Parallel spectrum ↔ \(G_\parallel\), \(S_\parallel\)
- Perpendicular spectrum ↔ \(G_\perp\), \(S_\perp\)
- Fit geometry example: \(a=25\,\mathrm{nm}\), \(d=26.451\,\mathrm{nm}\), \(n_{\mathrm{host}}=1.33\), \(\alpha_0\sim 0.43\,\mathrm{nm}^3\)

### JSON examples

**Parallel hybrid**

```json
{
  "settings": { "dt": 0.05, "t_end": 400, "driver": "fourier" },
  "plasmon": {
    "simulation": {
      "cell_length": 0.2,
      "pml_thickness": 0.05,
      "surrounding_material_index": 1.33
    },
    "source": {
      "type": "gaussian",
      "center": [-0.04, 0, 0],
      "size": [0, 0.2, 0.2],
      "component": "z",
      "additional_parameters": { "frequency": 2.29, "fwidth": 2.08 }
    },
    "nanoparticle": {
      "material": "Au_JC_visible",
      "radius": 0.025,
      "center": [0, 0, 0]
    },
    "molecule": { "position": [0.02645, 0, 0] }
  },
  "molecule": {
    "geometry": "Na.xyz",
    "geometry_units": "angstrom",
    "basis": "6-31g*",
    "xc": "pbe0",
    "spin": 1
  },
  "additional_parameters": {
    "fourier": {
      "polarization": "parallel",
      "gamma": 0.0,
      "tau": 200,
      "min_ev": 1.5,
      "max_ev": 4.0,
      "spectrum_filepath": "spectrum_parallel.png",
      "field_e_ref_filepath": "field_e_ref_parallel.csv"
    }
  }
}
```

**Perpendicular hybrid**

```json
"fourier": {
  "polarization": "perpendicular",
  "perp_component": "y",
  "spectrum_filepath": "spectrum_perp.png",
  "field_e_ref_filepath": "field_e_ref_perp.csv"
}
```

### Practical guidance

- Prefer placing the molecule on a **Cartesian axis** relative to the NP for a clean ∥ component.
- Mirror symmetries must be compatible with the source and NP; back-propagation has additional constraints (see Usage).
- Cost per orientation is roughly **one third** of a full three-direction hybrid Fourier campaign.
- Checkpointing multi-direction Fourier is specialized; single-pol runs are simpler to manage as one job each.
- Reuse vacuum references with `field_e_ref_filepath` when only the molecule/NP changes and the source/cell are identical.

## Parameter reference (`additional_parameters.fourier`)

| Key | Type | Default | Description |
| ----- | ------ | --------- | ------------- |
| `gamma` | float | 0 | Broadening window \(e^{-\gamma t}\) (a.u.) |
| `tau` | float | — | Extra damping \(e^{-t/\tau}\) (a.u.) |
| `min_ev` / `max_ev` | float | 1.5 / 5.0 | Plot window (eV) |
| `spectrum_filepath` | str | required* | Output spectrum PNG |
| `npz_filepath` | str | null | Optional raw FFT storage |
| `polarization` | str | `full` | `full` \| `parallel` \| `perpendicular` |
| `perp_component` | str | auto | `x`/`y`/`z` for perpendicular mode |
| `field_e_ref_filepath` | str | `field_e_ref.csv` | Vacuum \(E_{\mathrm{inc}}\) CSV |
| `use_existing_e_field_ref` | bool | auto if file exists | Skip vacuum Meep runs |
| `reference_only` | bool | false | Vacuum refs only (`full` only) |

\*Or fallback to `files.spectra_e_vs_p_filepath` when building the Fourier block.

## Implementation map

| Piece | Module |
| ------- | -------- |
| Orchestration | `fourier/driver.py` |
| ∥ / ⊥ builders | `fourier/polarization.py` |
| Transverse source | `fourier/source_face.py` |
| Direction / ref copies | `fourier/setup.py` |
| Fold / damp / refs | `fourier/io_fields.py` |
| FFT / absorption | `fourier/spectrum.py` |
| Post-process | `fourier/postprocess.py` |
| Vacuum field record | `SIMULATION._record_field_only` |

## See also

- [Effective Polarizability](effective_polarizability.md) — Gersten–Nitzan \(G,S\) and joint fits
- [Usage](../usage.md) — full JSON schema
- [Theory & Methodology](../methodology.md) — hybrid time loop
