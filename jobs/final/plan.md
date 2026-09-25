# Plan: nanoparticle and CAP effects on a double core hole

This is the working plan for the double-core-hole campaign. The run list and the current status are `jobs/final/progress.md`. Read those steps in order. This file states the goal, what was already checked on other systems, and how each remaining run is set up.

Step 1 is done. It is the double-core-hole literature and a valence-absorption screen of those molecules, plus trans-thioindigo (`jobs/final/Step_1/README.md`). trans-Thioindigo's bright root is 417 nm, on the 418.5 nm silver plasmon. No literature molecule in that set is. The production molecule is not chosen. The nanoparticle is not chosen. There is no separate step for choosing the pair.

Two pairs are still open. The resonant pair is a bright root and a plasmon on top of each other. Steps 2–7 use it. Step 7 is that same pair at three other surface gaps. The detuned pair is a bright root and a plasmon that do not overlap. Step 8 repeats the key runs on it, still at the 0.015 μm gap. Every JSON from Step 2 on names the molecule and the nanoparticle with placeholders. Do not launch a file that still contains `PLACEHOLDER_`.

DCH means two electrons are removed from one core MO, so the propagation stays closed-shell. Which MO that is comes from the survey of the molecule that was actually chosen. A survey of trans-thioindigo is on disk in `Step_2/thioindigo_trial/` and is not that assignment. On that candidate, MO 0 and MO 1 are sulfur 1s and MO 2 and MO 3 are oxygen 1s, and the two oxygen 1s are a canonical combination of both carbonyls rather than a hole on one of them. Equivalent atoms in the molecule you do choose need the same reading.

## 1. Goals

Investigate how two things change DCH dynamics and spectra of a molecule next to a nanoparticle whose plasmon sits on that molecule's bright root:

1. The silver sphere, through the local field and, when back-propagation is on, through the field the molecule itself radiates.
2. A complex absorbing potential (CAP; the Lopata non-Hermitian term on the Fock matrix), through lifetimes and peak widths.

The comparison that matters is the change produced by turning one of those pieces on, with everything else held fixed. A single “DCH + NP + CAP + pulse” job mixes the causes. Step 7 is the distance comparison on the resonant pair: the same nanoparticle effect at surface gaps of 0.005, 0.030, and 0.060 μm, against the 0.015 μm baseline. Step 8 is the further comparison off resonance, still at 0.015 μm. If the effect shrinks there, it was the shared resonance. If it does not, the sphere is doing something that does not need the plasmon on the dye.

Two observables:

- **Dynamics.** Time-dependent hole occupations on the neutral MO basis (`mo_occ.csv` and its plot). Sudden DCH is already a non-stationary initial condition, so the hole moves even when the external electric field is zero.
- **Spectra.** Real-time absorption of the isolated molecule (δ-kick) and the hybrid absorption cross section under a Gaussian pulse centered on the dye (Meep field divided by the vacuum incident field).

Field-free hybrid jobs must not be analyzed as a cross section. $\sigma_m$ divides by $|E_{\mathrm{inc}}|^2$. With no incident source that denominator is ~0. Those runs are dynamics: occupations, the local field, and the induced dipole.

## 2. Checks already done

These runs are not part of `jobs/final`. They are why the methods in this plan are trusted. Do not rerun them as a substitute for Steps 2–8, and do not overlay their outputs on the production traces. The old inputs use retired driver names.

### 2.1 Sudden DCH occupations

`jobs/DCH/nascimento_fig8` compares PlasMol hole occupations for 3-pentanone with the digitized curves from Nascimento and co-workers (`jobs/DCH/Nascimento2020.pdf`, Fig. 8a files under `Gold Data`) and with an NWChem real-time run of the same sudden double hole (`nwchem_dch/3p_dch.out`). The valence holes that were followed are MOs 21, 23, and 24. `parse_compare.py` measures the oscillation periods of those holes.

That comparison is what established the sudden-DCH path: neutral SCF, remove two electrons from one core MO without a second SCF, propagate, and project the density back onto the neutral orbitals. The same path is used here. The orbital indices are not. 21, 23, and 24 were the 3-pentanone valence set in that figure. They are not the frontier of the molecule chosen here.

Older traces in `jobs/DCH/mo_tracking_*` used `"driver": "dch"` and `dch_*` keys. Those names are not migrated. They are not valid production inputs.

### 2.2 Lopata CAP broadening

The CAP is the non-Hermitian term of Lopata and Govind, *J. Chem. Theory Comput.* **9**, 4939 (2013). Orbitals above a vacuum level $\varepsilon_0$ are damped; orbitals below it are not. PlasMol adds that term inside the Fock build.

Two checks sit on disk:

- Water, with and without the CAP, in `jobs/H2O` (`run_631g*`, `run_631g*_noCAP`, and the aug-cc-pVTZ pair). The methodology notes compare that kind of spectrum with the electron-energy-loss data Lopata and Govind used. The width depends on the functional and the basis, because those move the orbital energies relative to $\varepsilon_0$.
- A tune on 3-pentanone, in `jobs/DCH/broadening`, gave $\mu = 0.342720$ and $\varepsilon_0 = 0.003028$ Ha. That pair is not the production value. Both are placeholders from Step 2 on. $\mu$ is part of LC-ωPBE (Libxc’s default is 0.4), so every file for one molecule uses the same $\mu$, CAP on or off. $\varepsilon_0$ is tuned at that $\mu$ and is written only inside `molecule.cap`.

### 2.3 Hybrid FDTD against the point-dipole model

`jobs/Au_Na` is a sodium atom next to a gold sphere, in water. PlasMol’s hybrid Fourier spectra were computed in the parallel and perpendicular channels and compared with the quasistatic Gersten–Nitzan model (a point-polarizable molecule plus the sphere’s Mie multipoles). The write-up is `doc/docs/methodology.md` (validation section) and `jobs/Au_Na/PlasMol_Parallel/TeX/hybrid_spectra.tex`. The model notes are `jobs/Au_Na/Models/Gersten-Nitzan` and the two-oscillator reduction in `jobs/Au_Na/Models/COM`.

With `Au_JC_visible`, host index 1.33, and multipoles through $\ell_{\max}=25$, the joint fit is $\omega_m \approx 2.055$ eV, $\gamma_m \approx 0.076$ eV, $\alpha_0 \approx 0.43$ nm$^3$. Parallel and perpendicular peak positions agree to within a few tens of meV. The mean-squared error on the peak-normalized spectra is about $10^{-3}$. The same protocol with the Rakić gold dielectric reproduces both channels, so the match is not an artifact of one table. That is the alignment of the Meep hybrid loop (drive, local field, induced dipole, vacuum deconvolution, and the two orientations) against a closed classical dipole model. No separate Draine discrete-dipole-approximation program is in the tree.

This campaign does not reuse that sodium resonance or that gold sphere. One candidate sphere was a 35 nm silver particle in water, Mie peak 418.5 nm with Meep’s `Ag` dielectric. It is not selected. The Mie script and the bare-sphere flux jobs were removed with the pair-choice step. There is no flux input on disk.

## 3. Why the new runs are factored

| Factor | Off | On | What it isolates |
| --- | --- | --- | --- |
| DCH | Neutral molecule | Sudden double hole on the surveyed MO | The hole itself, rather than the response of the closed-shell molecule |
| CAP | No `molecule.cap` | Static CAP. $\varepsilon_0$ is the placeholder tuned for that molecule | Lifetime and broadening, rather than a coherent oscillation that does not decay |
| NP | No nanoparticle | The chosen sphere. Baseline gap 0.015 μm. Step 7 changes only the gap | The plasmonic scatterer. Material and radius are placeholders |
| Back-propagation | Molecule feels $\mathbf{E}$ but does not source it | Induced dipole is injected into Meep | Whether the sphere only reshapes the incident field, or also responds to the field the molecule radiates |
| Drive | None | δ-kick, or Gaussian on the molecule's bright root | External field versus free evolution of the sudden hole |

Back-propagation exists only inside a Meep cell.

### Shared electronic structure

If any of these change between two runs, the difference is no longer one factor.

| Item | Choice | Note |
| --- | --- | --- |
| Geometry | `PLACEHOLDER_RESONANT_MOLECULE.xyz`, or `_DETUNED_` in Step 8 | Do not edit the file once a step points at it. Survey, D1, D2, and K stay on that frame. D3–D6 gain a rotation only after the D1 check |
| Basis | `6-311G*`, Cartesian (6 $d$) | PlasMol default |
| Charge / spin | 0 / 0 | Singlet DCH stays restricted. Do not force UKS |
| XC | LC-ωPBE. $\mu$ is `PLACEHOLDER_*_MU` | Same value on every run of one molecule, CAP on or off. Do not omit `lrc_parameter`. Do not paste the pentanone 0.34272 |
| Core MO | Survey, then one index | The JSON key is `PLACEHOLDER_*_CORE_MO`. Two electrons. Not the survey's 0–5 list |
| Plot MOs | Three frontier indices | `watch_indices` is a placeholder until the molecule is chosen. Plot only. Logging still covers MO 0 through the neutral LUMO+1 |
| Molecule-only time | `dt` 0.05 au. Dynamics end at 400 au. Kicks end at 4000 au | |
| Hybrid time | `dt` 0.1 au, `t_end` 10000 au | PlasMol snaps `dt` to the Meep step (~0.10007 au) at Courant 0.5. Every hybrid job uses this pair. Do not overlay a 0.05 au trace on a 0.1 au trace as if they were one grid; compare only the overlapping window |
| Medium | 1.33 | Water. Every Meep cell. Molecule-only runs have no classical medium |
| Sphere | Placeholder material and radius, center at the origin | Baseline molecule x is radius + 0.015 μm. Step 7 uses radius plus 0.005, 0.030, or 0.060 μm. `cell_length` stays 0.2 unless that no longer fits |
| Gaussian | Wavelength is the molecule's bright root, `fwidth` 2.0 | Same source in every G file of a pair. The plot window is a placeholder on the Gaussian jobs |

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
| 1 | Done | Which molecules, including trans-thioindigo, have a bright valence line near a metallic plasmon |
| 2 | Placeholder | Which MO is the core on the resonant molecule |
| 3 | Placeholder | Field-free DCH on the resonant pair, one factor at a time (D1–D6) |
| 4 | Placeholder | δ-kick absorption of the resonant molecule, ± DCH, ± CAP (K1–K4) |
| 5 | Placeholder | Gaussian hybrid cross section of the resonant pair, ± DCH, ± NP, ± CAP (G1–G8) |
| 6 | Placeholder | Two controls on resonant G4, after G4 itself moves |
| 7 | Placeholder | Resonant pair at three other gaps, after G4 − G2 is nonzero |
| 8 | Placeholder | The key runs again, on a pair whose resonances do not overlap |

### Step 1 — literature and the valence screen

Done. Notes: `jobs/final/Step_1/README.md`. No production input. Each literature molecule is entered with the DOI, the question that paper asked, and the result. The folders under `Step_1/Molecules/` are the valence screen. trans-Thioindigo is `Molecules/transthioindigo/`: PySCF LC-ωPBE at the screen value μ = 0.34272, bright root 2.976 eV (417 nm, $f = 0.29$) on the existing PBE0 geometry. That μ is not copied into the later JSON. The root moves if the production μ changes. The benzene maximum is 543 nm. Most of that gap is the functional. If this dye is selected, match the calculated root. A 35 nm `Ag` sphere in water peaks at 418.5 nm. That candidate is not selected. No literature molecule in the screen is a bright line on that plasmon or on the 540 nm gold plasmon. Thioindigo is not in the double-core-hole literature. Its μ tune, `LC-wPBE mu/transthioindigo/tune.json`, has not been run. The sudden double hole used from Step 3 on is the pentanone initial condition in that note, not a two-site free-electron-laser spectrum.

### Step 2 — core survey

`jobs/final/Step_2/survey.json`. Driver `core_hole`. Neutral SCF, then the atoms that contribute to MOs 0 through 5, then exit. The table is written to `mo_survey.txt` in that directory. Launch with `--log log.out` so the run record is saved there too. No ionization and no propagation. `t_end` is unused. The geometry is `PLACEHOLDER_RESONANT_MOLECULE.xyz`.

An earlier survey of trans-thioindigo is in `Step_2/thioindigo_trial/`. It assigns MO 0 and MO 1 to sulfur (99.33% each), MO 2 and MO 3 to oxygen (99.20% each), and MO 4 and MO 5 to carbon (99.05% each). The two oxygen 1s are equivalent, so either index is a canonical combination of both carbonyls, not a hole on one of them. Do not copy those indices forward. Do not launch Step 3 until the resonant survey has been run and `_CORE_MO` is filled from it.

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

K1 is the ordinary spectrum of the resonant molecule. The plotted curve averages x, y, and z, so a rotation would not change it. K3 − K1 is the sudden hole with no CAP. K2 − K1 and K4 − K3 are the CAP on a given initial state. `gamma` is only the Fourier window (~0.27 eV half-width). It is not a substitute for the CAP. The dipole is already small by a few hundred au, so 4000 au does not sharpen the spectrum unless `gamma` is lowered. If the bright root is above 15 eV, raise `max_ev` on all four files together.

DCH kicks write `mo_occ.csv` under `x_dir/`, `y_dir/`, and `z_dir/`.

Step 1 is the linear-response check of each neutral molecule in the screen. It is not a substitute for K1.

### Step 5 — Gaussian hybrid cross sections

Same cell and the same Gaussian in every file. The sphere, when present, is the resonant placeholder (material and radius). The source wavelength is the molecule's bright root, `fwidth` 2.0. The plotted window is `PLACEHOLDER_RESONANT_WINDOW_MIN_EV` to `PLACEHOLDER_RESONANT_WINDOW_MAX_EV`. `gamma` is 0. The published curve is `cross_section`. Polarization `parallel` puts $\mathbf{E}$ along the nanoparticle–molecule axis and rearranges the source so $\mathbf{k}\perp\mathbf{E}$. Polarization `single` is the empty cell: no nanoparticle, source left as written. Back-propagation is on. No checkpoints.

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

Not part of the grid. Run them only after the resonant G4 itself shows a nanoparticle effect.

| File | Change from G4 |
| --- | --- |
| `G4_perpendicular` | Polarization `perpendicular` |
| `G4_no_backprop` | `back_propagation` false. The molecule feels the cell and does not source it |

One of each is enough. Do not repeat them across CAP and DCH until G4 has moved. Do not copy them into Step 8 unless the detuned G4 has moved and D3 − D4 does not already explain it. Do not repeat them across the Step 7 gaps.

Single core holes are out of this plan: removing one electron forces an open shell and is a different initial state.

### Step 7 — distance

Notes: `jobs/final/Step_7/README.md`.

Not part of the grid. Run it only after resonant G4 − G2 is nonzero at 0.015 μm. The baseline gap is not repeated. Each of 0.005, 0.030, and 0.060 μm has D3, D4, and G4, and nothing else. D1 and G2 have no sphere, so the differences are still D3 − D1 and G4 − G2.

The molecule x token is `PLACEHOLDER_RESONANT_X_GAP005_UM`, `_GAP030_`, or `_GAP060_`. It is the radius plus that gap, not the baseline x. Paste the Step 3 orientation into these D3 and D4 files. If the gap changes D3 and G4 but D4 stays with D1, the distance dependence is the radiated field. Widen `cell_length` before launching a gap that no longer fits between the PML.

### Step 8 — detuned pair

Notes: `jobs/final/Step_8/README.md`.

The resonant grid asks what the nanoparticle does when the plasmon sits on the bright root. This step asks whether that effect survives when it does not. The Gaussian is still centered on the molecule's bright root. The nanoparticle is a different resonance.

The key set is the survey, D1, D3, D4, K1, K3, and G1–G4. D3 − D1 is the sphere when the molecule radiates. D4 − D1 is the sphere when it does not. G4 − G2 is the same question in the cross section. K1 is the measurement that the plasmon misses the root. The CAP rows, the Step 6 controls, and the Step 7 gaps are not repeated. The gap stays 0.015 μm.

Every molecule field and every nanoparticle field uses `PLACEHOLDER_DETUNED` rather than `PLACEHOLDER_RESONANT`. If the detuned pair keeps the resonant molecule, copy the molecule tokens and change the nanoparticle, and do not rerun D1, K1, K3, G1, or G2. There is no bare-sphere flux job.

The orientation check is the Step 3 command, pointed at `Step_8/D1/field_p_D1.csv`. Paste the rotation into Step 8 D3 and D4 only.

## 5. Setup

Snippets match the JSON on disk. D3–D6 are the only jobs that gain `molecule.rotation`, and only after Step 3’s D1 check.

### 5.1 Survey

```json
{
  "settings": {"dt": 0.05, "t_end": 1.0, "driver": "core_hole"},
  "molecule": {
    "geometry": "PLACEHOLDER_RESONANT_MOLECULE.xyz",
    "geometry_units": "angstrom",
    "charge": 0,
    "spin": 0,
    "basis": "6-311G*",
    "xc": "HYB_GGA_XC_LC_WPBE",
    "lrc_parameter": "PLACEHOLDER_RESONANT_MU",
    "core_hole": {"mo_removal_index_dict": {"0": 2, "1": 2, "2": 2, "3": 2, "4": 2, "5": 2}}
  }
}
```

Electron counts in that dictionary are ignored. The driver prints atom contributions and exits.

### 5.2 Dynamics without a sphere (D1, D2)

`"driver": "quantum"`. No molecule source. No `plasmon` section. D2 adds `molecule.cap`. Both set:

```json
"core_hole": {
  "mo_removal_index_dict": {"PLACEHOLDER_RESONANT_CORE_MO": 2},
  "mo_occ_filepath": "mo_occ.csv",
  "watch_indices": ["PLACEHOLDER_RESONANT_HOMO_M1", "PLACEHOLDER_RESONANT_HOMO", "PLACEHOLDER_RESONANT_LUMO"]
}
```

Replace the core-MO token with the surveyed index, the three watch tokens with integer frontier indices, and `_MU` with the tuned range-separation parameter, before launching. Checkpointing is allowed. Step 8 uses the `PLACEHOLDER_DETUNED` spellings. Step 7 keeps the resonant spellings, except the molecule x token, which is the gap-specific one.

CAP block when it is on. It does not replace the functional or $\mu$. `_EPS0` is tuned at that $\mu$:

```json
"cap": {"type": "static", "gam0": 1, "xi": 0.5, "eps0": "PLACEHOLDER_RESONANT_EPS0", "clamp": 100}
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
  "nanoparticle": {"material": "PLACEHOLDER_RESONANT_NP", "radius": "PLACEHOLDER_RESONANT_NP_RADIUS_UM", "center": [0, 0, 0]},
  "molecule": {
    "position": ["PLACEHOLDER_RESONANT_MOLECULE_X_UM", 0, 0],
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
    "min_ev": "PLACEHOLDER_RESONANT_WINDOW_MIN_EV",
    "max_ev": "PLACEHOLDER_RESONANT_WINDOW_MAX_EV"
  }
}
```

Empty cell (G1, G2, G5, G6): same simulation and source, no `nanoparticle` block, `"polarization": "single"`, component along x. DCH members add `molecule.core_hole`. The occupation CSV is written next to the production fields (`fields/mo_occ.csv`), not beside the vacuum reference. CAP members add `molecule.cap`. `back_propagation` is true on G3, G4, G7, and G8. The back-propagation-off control is Step 6, not another row of this table.

## 6. Still open

Status as of 2026-09-24.

- [x] Step 1 literature note and valence screen, including trans-thioindigo at 417 nm. No literature molecule sits on the silver plasmon near 418 nm or the gold plasmon near 540 nm.
- [x] Thioindigo core survey in `Step_2/thioindigo_trial/`. Not selected, and not written into the JSON.
- [ ] Choose the resonant pair and the detuned pair. Then replace every `PLACEHOLDER_RESONANT` token in Steps 2–7. In Step 7 the x token is radius plus 0.005, 0.030, or 0.060 μm, not the baseline x. Replace every `PLACEHOLDER_DETUNED` token in Step 8.
- [ ] Run the resonant survey. Write `_CORE_MO` and the three frontier indices into the resonant DCH files. Do not reuse the thioindigo trial indices unless that molecule is the one selected.
- [ ] Run resonant D1, then the orientation check, then paste the rotation into D3–D6 only.
- [ ] One short D2 (DCH + CAP) and one short D3 (DCH, sphere, no source) before the full 400 au and 10000 au propagations.
- [ ] Hybrid jobs cannot checkpoint. D3–D6 and G1–G8 have to finish in one process. D1, D2, and K1–K4 can restart. Steps 7 and 8 have the same split.
- [ ] $\sigma_m$ is for shape comparison. Overlay peak-normalized curves when the claim is that a peak moved or broadened.
- [ ] After resonant G4 − G2 is nonzero, run Step 7 and compare D3 − D1 and G4 − G2 at each gap with the 0.015 μm baseline.
- [ ] Run Step 8 and compare those same differences between the resonant pair and the detuned pair. Both pairs stay at 0.015 μm.
