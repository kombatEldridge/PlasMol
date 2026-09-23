# Plan: nanoparticle and CAP effects on trans-thioindigo DCH

This is the working plan for the double-core-hole campaign. The run list is `jobs/final`. Read those steps in order. This file states the goal, what was already checked on other systems, and how each remaining run is set up.

The molecule is trans-thioindigo, `jobs/final/Step_1/transthioindigo.xyz`. The nanoparticle is a 35 nm silver sphere, Meep material `Ag` (the Rakic fit, not `Ag_visible`), in water (`surrounding_material_index` 1.33). Step 1 records why that pair was chosen. Later jobs use it.

DCH means two electrons are removed from one core MO, so the propagation stays closed-shell. The index is not known until the survey in Step 2. Thioindigo has two sulfur 1s orbitals below the two oxygen 1s, so MO 0 is not an oxygen core. The two oxygen 1s are equivalent: a canonical orbital is a combination of them, not a hole on one carbonyl.

## 1. Goals

Investigate how two things change DCH dynamics and spectra of this dye next to its matched plasmon:

1. The silver sphere, through the local field and, when back-propagation is on, through the field the molecule itself radiates.
2. A complex absorbing potential (CAP; the Lopata non-Hermitian term on the Fock matrix), through lifetimes and peak widths.

The comparison that matters is the change produced by turning one of those pieces on, with everything else held fixed. A single “DCH + NP + CAP + pulse” job mixes the causes.

Two observables:

- **Dynamics.** Time-dependent hole occupations on the neutral MO basis (`mo_occ.csv` and its plot). Sudden DCH is already a non-stationary initial condition, so the hole moves even when the external electric field is zero.
- **Spectra.** Real-time absorption of the isolated molecule (δ-kick) and the hybrid absorption cross section under a Gaussian pulse centered on the dye (Meep field divided by the vacuum incident field).

Field-free hybrid jobs must not be analyzed as a cross section. $\sigma_m$ divides by $|E_{\mathrm{inc}}|^2$. With no incident source that denominator is ~0. Those runs are dynamics: occupations, the local field, and the induced dipole.

## 2. Checks already done

These runs are not part of `jobs/final`. They are why the methods in this plan are trusted. Do not rerun them as a substitute for Steps 2–6, and do not overlay their outputs on the thioindigo traces. The old inputs use retired driver names.

### 2.1 Sudden DCH occupations

`jobs/DCH/nascimento_fig8` compares PlasMol hole occupations for 3-pentanone with the digitized curves from Nascimento and co-workers (`jobs/DCH/Nascimento2020.pdf`, Fig. 8a files under `Gold Data`) and with an NWChem real-time run of the same sudden double hole (`nwchem_dch/3p_dch.out`). The valence holes that were followed are MOs 21, 23, and 24. `parse_compare.py` measures the oscillation periods of those holes.

That comparison is what established the sudden-DCH path: neutral SCF, remove two electrons from one core MO without a second SCF, propagate, and project the density back onto the neutral orbitals. The same path is used here. The orbital indices are not. 21, 23, and 24 were the 3-pentanone valence set in that figure. They are not the thioindigo frontier.

Older traces in `jobs/DCH/mo_tracking_*` used `"driver": "dch"` and `dch_*` keys. Those names are not migrated. They are not valid production inputs.

### 2.2 Lopata CAP broadening

The CAP is the non-Hermitian term of Lopata and Govind, *J. Chem. Theory Comput.* **9**, 4939 (2013). Orbitals above a vacuum level $\varepsilon_0$ are damped; orbitals below it are not. PlasMol adds that term inside the Fock build.

Two checks sit on disk:

- Water, with and without the CAP, in `jobs/H2O` (`run_631g*`, `run_631g*_noCAP`, and the aug-cc-pVTZ pair). The methodology notes compare that kind of spectrum with the electron-energy-loss data Lopata and Govind used. The width depends on the functional and the basis, because those move the orbital energies relative to $\varepsilon_0$.
- The numbers locked for this series come from the tune in `jobs/DCH/broadening`: $\mu = 0.342720$ and $\varepsilon_0 = 0.003028$ Ha. That tune was on 3-pentanone, not on thioindigo. $\mu$ is kept on every thioindigo run, including the runs with no CAP, because it is part of the functional (Libxc’s default for LC-ωPBE is 0.4). $\varepsilon_0$ is kept inside `molecule.cap` only. It has not been retuned on thioindigo.

### 2.3 Hybrid FDTD against the point-dipole model

`jobs/Au_Na` is a sodium atom next to a gold sphere, in water. PlasMol’s hybrid Fourier spectra were computed in the parallel and perpendicular channels and compared with the quasistatic Gersten–Nitzan model (a point-polarizable molecule plus the sphere’s Mie multipoles). The write-up is `doc/docs/methodology.md` (validation section) and `jobs/Au_Na/PlasMol_Parallel/TeX/hybrid_spectra.tex`. The model notes are `jobs/Au_Na/Models/Gersten-Nitzan` and the two-oscillator reduction in `jobs/Au_Na/Models/COM`.

With `Au_JC_visible`, host index 1.33, and multipoles through $\ell_{\max}=25$, the joint fit is $\omega_m \approx 2.055$ eV, $\gamma_m \approx 0.076$ eV, $\alpha_0 \approx 0.43$ nm$^3$. Parallel and perpendicular peak positions agree to within a few tens of meV. The mean-squared error on the peak-normalized spectra is about $10^{-3}$. The same protocol with the Rakić gold dielectric reproduces both channels, so the match is not an artifact of one table. That is the alignment of the Meep hybrid loop (drive, local field, induced dipole, vacuum deconvolution, and the two orientations) against a closed classical dipole model. No separate Draine discrete-dipole-approximation program is in the tree.

This campaign does not reuse that sodium resonance or that gold sphere. The classical cross section of the silver sphere used here was computed with Mie theory and Meep’s `Ag` dielectric (`jobs/final/Step_1/ag_mie.py`). The matching Meep flux job, `Step_1/d35/d35.json`, has not been run.

## 3. Why the new runs are factored

| Factor | Off | On | What it isolates |
| --- | --- | --- | --- |
| DCH | Neutral thioindigo | Sudden double hole on the surveyed MO | The hole itself, rather than the response of the closed-shell dye |
| CAP | No `molecule.cap` | Static CAP, $\varepsilon_0 = 0.003028$ Ha | Lifetime and broadening, rather than a coherent oscillation that does not decay |
| NP | No nanoparticle | 35 nm `Ag` sphere, gap 0.015 μm | The plasmonic scatterer |
| Back-propagation | Molecule feels $\mathbf{E}$ but does not source it | Induced dipole is injected into Meep | Whether the sphere only reshapes the incident field, or also responds to the field the molecule radiates |
| Drive | None | δ-kick, or Gaussian at 0.417 μm | External field versus free evolution of the sudden hole |

Back-propagation exists only inside a Meep cell.

### Shared electronic structure

If any of these change between two runs, the difference is no longer one factor.

| Item | Choice | Note |
| --- | --- | --- |
| Geometry | `jobs/final/Step_1/transthioindigo.xyz` | Do not edit it. Survey, D1, D2, and K stay on this frame. D3–D6 gain a rotation only after the D1 check |
| Basis | `6-311G*`, Cartesian (6 $d$) | PlasMol default |
| Charge / spin | 0 / 0 | Singlet DCH stays restricted. Do not force UKS |
| XC | LC-ωPBE, $\mu = 0.34272$ | On every run, CAP on or off. Do not omit `lrc_parameter` |
| Core MO | Survey in Step 2, then one index | Production files still say MO 0. That is a placeholder |
| Plot MOs | 74, 75, 76 | HOMO−1, HOMO, LUMO of the 76-orbital closed shell (152 electrons). Plot only. Logging still covers MO 0 through the neutral LUMO+1 |
| Molecule-only time | `dt` 0.05 au. Dynamics end at 400 au. Kicks end at 4000 au | |
| Hybrid time | `dt` 0.1 au, `t_end` 10000 au | PlasMol snaps `dt` to the Meep step (~0.10007 au) at Courant 0.5. Every hybrid job uses this pair. Do not overlay a 0.05 au trace on a 0.1 au trace as if they were one grid; compare only the overlapping window |
| Medium | 1.33 | Water. Every Meep cell. Molecule-only runs have no classical medium |
| Sphere | `Ag`, radius 0.0175 μm, center at the origin | Molecule at `[0.0325, 0, 0]` |
| Gaussian | wavelength 0.417 μm, `fwidth` 2.0 | Same source in every G file |

$\mu$ changes the real Fock matrix on every run. A CAP-off job that dropped it would not be the same Hamiltonian as its CAP-on partner.

### What each run writes

| Run type | Primary output | Do not treat as a result |
| --- | --- | --- |
| Dynamics, molecule only | `mo_occ.csv`, occupation PNG, `field_p.csv` | A spectrum. There is no kick |
| Dynamics, hybrid, no source | `mo_occ.csv`, `fields/field_e.csv`, `fields/field_p.csv` | $\sigma_m$. $E_{\mathrm{inc}}\approx 0$ |
| δ-kick absorption | `cross_section` spectrum, plus `mo_occ.csv` when DCH is on | Hybrid deconvolution. There is no Meep cell |
| Gaussian hybrid | `cross_section` spectrum, vacuum `field_e_ref`, production fields, `mo_occ.csv` when DCH is on | A three-kick `full` spectrum. `full` is rejected when a nanoparticle is present |

The published hybrid comparison is `cross_section`. `dissipative_power` and `A_raw` stay available as diagnostics.

## 4. Run list

These are the directories in `jobs/final`.

| Step | Status | Question |
| --- | --- | --- |
| 1 | Linear response and Mie already run | Which dye and which sphere share a resonance |
| 2 | Not run | Which MO is the oxygen core |
| 3 | Not run | Field-free DCH, one factor at a time (D1–D6) |
| 4 | Not run | δ-kick absorption, ± DCH, ± CAP (K1–K4) |
| 5 | Not run | Gaussian hybrid cross section, ± DCH, ± NP, ± CAP (G1–G8) |
| 6 | Not run | Two controls on G4, after G4 itself moves |

### Step 1 — the pair

Notes: `jobs/final/Step_1/README.md`.

Ten small dyes were considered for a bright band near 540 nm, where a 50 nm gold sphere resonates. Only trans-thioindigo was calculated. Gas-phase LC-ωPBE puts the bright root at 2.976 eV (417 nm, $f = 0.29$). The benzene maximum is 543 nm; the vapor maximum is 508 nm. Most of the blue shift is the functional, not the missing solvent. The resonance this series matches is the calculated root.

Mie theory with Meep’s `Ag` dielectric and $n = 1.33$ peaks at 416.5 nm for a 32.6 nm sphere and at 418.5 nm for a 35 nm sphere. At 417 nm the 35 nm cross section is 99% of its peak. The jobs use 35 nm. `Ag_visible` peaks at the same wavelength only for a different diameter, and Meep marks that fit unstable. It is not used.

`d35/d35.json` is the bare 35 nm sphere as a Meep flux job (300–550 nm). It has not been run. The Mie spectrum is the classical result already in hand.

### Step 2 — core survey

`jobs/final/Step_2/survey.json`. Driver `core_hole`. Neutral SCF, then the atoms that contribute to MOs 0 through 5, then exit. The table is written to `mo_survey.txt` in that directory. Launch with `--log log.out` so the run record is saved there too. No ionization and no propagation. `t_end` is unused.

Those six orbitals are where the two sulfur 1s and the two oxygen 1s should lie. Read the log and replace `"0"` in every later `mo_removal_index_dict` with the one oxygen-core index this series will ionize. If that character is outside `{0, 1, 2, 3, 4, 5}`, add the index and run the survey again. Do not launch Step 3 until that replacement is made.

### Step 3 — field-free dynamics

No external electric field. The drive is the sudden hole, plus whatever field that hole’s dipole induces on the sphere.

| ID | CAP | NP | Back-propagation | Driver |
| --- | --- | --- | --- | --- |
| D1 | off | no | — | `quantum` |
| D2 | on | no | — | `quantum` |
| D3 | off | yes | on | `plasmol` |
| D4 | off | yes | off | `plasmol` |
| D5 | on | yes | on | `plasmol` |
| D6 | on | yes | off | `plasmol` |

- D2 − D1: CAP on the free hole.
- D3 − D1: sphere, molecule allowed to radiate, no CAP. Compare only the first 400 au. The time steps differ (0.05 au versus the Meep step).
- D4 − D1: sphere present, molecule does not source the cell. If D4 matches D1 and D3 does not, the nanoparticle effect is the radiated field coming back.
- D5 − D3 and D6 − D4: CAP on top of each nanoparticle case.
- D3 − D4 and D5 − D6: back-propagation at fixed CAP.

D1 and D2 stay on the stock geometry and can checkpoint. D3–D6 have no Meep source and cannot checkpoint. After D1, `python -m plasmol.quantum.orientation_check` reads `jobs/final/Step_3/D1/field_p_D1.csv` and prints the rotation that puts the moving core-hole dipole on +x. Paste that into D3–D6 only. There is no separate orientation search. K1’s three dipole files are the valence axis if a Gaussian job needs one; they are not this rotation.

Do not treat D3–D6 as cross sections.

Neutral field-free dynamics are not in this matrix. With no hole and no field the density does not move.

### Step 4 — δ-kick spectra

Molecule only. Absorption driver, `polarization: "full"`. Kick strength 0.001 au. `t_end` 4000 au. Fourier window `gamma` 0.01, plotted from 1.5 to 15 eV. These jobs can checkpoint.

| ID | DCH | CAP |
| --- | --- | --- |
| K1 | off | off |
| K2 | off | on |
| K3 | on | off |
| K4 | on | on |

K1 is the ordinary thioindigo spectrum. The plotted curve averages x, y, and z, so a rotation would not change it. K3 − K1 is the sudden hole with no CAP. K2 − K1 and K4 − K3 are the CAP on a given initial state. `gamma` is only the Fourier window (~0.27 eV half-width). It is not a substitute for the CAP. The dipole is already small by a few hundred au, so 4000 au does not sharpen the spectrum unless `gamma` is lowered.

DCH kicks write `mo_occ.csv` under `x_dir/`, `y_dir/`, and `z_dir/`.

Step 1 is the linear-response check of the same neutral molecule. It is not a substitute for K1.

### Step 5 — Gaussian hybrid cross sections

Same cell and the same Gaussian in every file. Sphere `Ag`, radius 0.0175 μm, when a nanoparticle is present. Source wavelength 0.417 μm, `fwidth` 2.0. Plotted window 1.5 to 5 eV. `gamma` is 0. The published curve is `cross_section`. Polarization `parallel` puts $\mathbf{E}$ along the nanoparticle–molecule axis and rearranges the source so $\mathbf{k}\perp\mathbf{E}$. Polarization `single` is the empty cell: no nanoparticle, source left as written. Back-propagation is on. No checkpoints.

| ID | DCH | NP | CAP | Polarization |
| --- | --- | --- | --- | --- |
| G1 | off | no | off | `single` |
| G2 | on | no | off | `single` |
| G3 | off | yes | off | `parallel` |
| G4 | on | yes | off | `parallel` |
| G5 | off | no | on | `single` |
| G6 | on | no | on | `single` |
| G7 | off | yes | on | `parallel` |
| G8 | on | yes | on | `parallel` |

G3 − G1 and G4 − G2 are the nanoparticle at fixed initial state. G2 − G1 and G4 − G3 are the hole at fixed cell. G8 − G4 is the CAP on the production hybrid. Run G1–G4 first if the full set is too many at once.

The empty-cell Gaussian is not a substitute for Step 4. K1–K4 stay the real-time molecular reference. The empty cell checks that Meep plus deconvolution recovers that molecule before the sphere is introduced. Do not reuse a vacuum reference computed with a different source or a different `dt`.

### Step 6 — two controls on G4

Not part of the grid. Run them only after G4 itself shows a nanoparticle effect.

| File | Change from G4 |
| --- | --- |
| `G4_perpendicular` | Polarization `perpendicular` |
| `G4_no_backprop` | `back_propagation` false. The molecule feels the cell and does not source it |

One of each is enough. Do not repeat them across CAP and DCH until G4 has moved.

Do not add a gap scan yet. The gap stays 0.015 μm. A second gap is a follow-up if G4 − G2 is nonzero. Single core holes are out of this plan: removing one electron forces an open shell and is a different initial state.

## 5. Setup

Snippets match the JSON on disk. D3–D6 are the only jobs that gain `molecule.rotation`, and only after Step 3’s D1 check.

### 5.1 Survey

```json
{
  "settings": {"dt": 0.05, "t_end": 1.0, "driver": "core_hole"},
  "molecule": {
    "geometry": "../Step_1/transthioindigo.xyz",
    "geometry_units": "angstrom",
    "charge": 0,
    "spin": 0,
    "basis": "6-311G*",
    "xc": "HYB_GGA_XC_LC_WPBE",
    "lrc_parameter": 0.34272,
    "core_hole": {"mo_removal_index_dict": {"0": 2, "1": 2, "2": 2, "3": 2, "4": 2, "5": 2}}
  }
}
```

Electron counts in that dictionary are ignored. The driver prints atom contributions and exits.

### 5.2 Dynamics without a sphere (D1, D2)

`"driver": "quantum"`. No molecule source. No `plasmon` section. D2 adds `molecule.cap`. Both set:

```json
"core_hole": {
  "mo_removal_index_dict": {"0": 2},
  "mo_occ_filepath": "mo_occ.csv",
  "watch_indices": [74, 75, 76]
}
```

Replace `"0"` with the surveyed index before launching. Checkpointing is allowed.

CAP block when it is on. It does not replace the functional or $\mu$:

```json
"cap": {"type": "static", "gam0": 1, "xi": 0.5, "eps0": 0.003028, "clamp": 100}
```

### 5.3 Dynamics with the sphere (D3–D6)

`"driver": "plasmol"`. Same molecule block as D1. Omit `plasmon.source`. The only classical source is the molecular dipole, and only when `back_propagation` is true.

```json
"plasmon": {
  "simulation": {
    "cell_length": 0.2,
    "pml_thickness": 0.05,
    "surrounding_material_index": 1.33,
    "courant": 0.5
  },
  "nanoparticle": {"material": "Ag", "radius": 0.0175, "center": [0, 0, 0]},
  "molecule": {
    "position": [0.0325, 0, 0],
    "tolerance_field_e": 1e-20,
    "back_propagation": true
  }
}
```

D4 and D6 set `back_propagation` to false. Do not turn on mirror symmetries. The induced dipole will not respect them.

Plasmon runs cannot checkpoint. RT-TDDFT still runs when $|\mathbf{E}|$ is below `tolerance_field_e`, because a core hole is present. That is what lets the hole evolve before the radiated field grows. Do not point these jobs at the absorption driver.

### 5.4 δ-kick spectra (K1–K4)

`"driver": "absorption"` with `polarization: "full"`. No plasmon section. K3 and K4 add the same `core_hole` block as D1. K2 and K4 add the same `cap` block as D2. K1 and K2 omit `core_hole`.

```json
"settings": {
  "dt": 0.05,
  "t_end": 4000,
  "driver": {
    "name": "absorption",
    "polarization": "full",
    "spectrum_filepath": "spectrum.png",
    "npz_filepath": "absorption.npz",
    "observables": ["cross_section"],
    "min_ev": 1.5,
    "max_ev": 15.0,
    "gamma": 0.01
  }
}
```

`t_end` and `gamma` must match across K1–K4.

### 5.5 Gaussian hybrid spectra (G1–G8)

Same sphere, cell, PML, gap, and Gaussian in every file. `dt` 0.1 and `t_end` 10000 must match as well. Parallel mode overwrites the source component onto the nanoparticle–molecule axis.

```json
"settings": {
  "dt": 0.1,
  "t_end": 10000,
  "driver": {
    "name": "absorption",
    "polarization": "parallel",
    "spectrum_filepath": "spectrum_parallel.png",
    "npz_filepath": "fourier_parallel.npz",
    "field_e_ref_filepath": "field_e_ref_parallel.csv",
    "observables": ["cross_section"],
    "min_ev": 1.5,
    "max_ev": 5.0
  }
}
```

Empty cell (G1, G2, G5, G6): same simulation and source, no `nanoparticle` block, `"polarization": "single"`, component along x. DCH members add `molecule.core_hole`. The occupation CSV is written next to the production fields (`fields/mo_occ.csv`), not beside the vacuum reference. CAP members add `molecule.cap`. `back_propagation` is true on G3, G4, G7, and G8. The back-propagation-off control is Step 6, not another row of this table.

## 6. Still open

Status as of 2026-09-23.

- [x] The pair is chosen and written into the production JSON: trans-thioindigo, `Ag`, radius 0.0175 μm, molecule at `[0.0325, 0, 0]`, Gaussian at 0.417 μm.
- [x] Linear-response and Mie results for that pair are in Step 1.
- [ ] Survey Step 2 and replace MO 0 in every core-hole file.
- [ ] Run D1, then the orientation check, then paste the rotation into D3–D6 only.
- [ ] Bare-sphere Meep flux, `Step_1/d35/d35.json`, if the Mie cross section needs a FDTD confirmation. Not required before the survey.
- [ ] One short D2 (DCH + CAP) and one short D3 (DCH, sphere, no source) before the full 400 au and 10000 au propagations.
- [ ] Hybrid jobs cannot checkpoint. D3–D6 and G1–G8 have to finish in one process. D1, D2, and K1–K4 can restart.
- [ ] $\sigma_m$ is for shape comparison. Overlay peak-normalized curves when the claim is that a peak moved or broadened.
