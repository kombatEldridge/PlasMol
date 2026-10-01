# Plan: nanoparticle and CAP effects on a double core hole

This is the working plan for the double-core-hole campaign. The run list and the current status are `jobs/final/progress.md`. Read those steps in order. This file states the goal, what was already checked on other systems, and how each remaining run is set up.

Step 1 is done. It is the double-core-hole literature and a valence-absorption screen of those molecules, plus trans-thioindigo (`jobs/final/Step_1/README.md`). trans-Thioindigo's bright root is 417 nm, on the 418.5 nm silver plasmon. No literature molecule in that set is. The resonant pair for Steps 2–7 is a sodium atom and a 25 nm gold sphere, Meep material `Au` (the Rakić table), in water. The detuned pair is still open. Step 8 names it with `PLACEHOLDER_DETUNED` and is not filled from the sodium job. Do not launch a file that still contains `PLACEHOLDER_`.

The sodium bright root at the production basis is 2.004849 eV, written into the Gaussian jobs as 0.618422 μm. A 25 nm Rakić sphere in water does not sit on that root. Its Mie absorption peaks at 2.340 eV, 0.335 eV higher (`jobs/final/Step_9/dielectric.txt`). The pair is the resonant series anyway. Step 9 is the later attempt to build a dielectric whose resonance does sit on the sodium line. That dielectric is `jobs/final/Step_9/na_drude.py`, loaded as `material_file`, not a built-in `meep.materials` entry. The jobs are written: D3, D4, G3, and G4. The Gaussian plot window is 1.5 to 4.5 eV. None of the four still contain a placeholder.

DCH means two electrons are removed from one core MO, with no second SCF, and the density is projected back onto the neutral orbitals. Sodium is charge 0, spin 1, so the job is unrestricted. A double hole on one spatial MO keeps spin 1 and zeros both the α and the β orbital of that index. The production index is `"0"`. That is the Na 1s, assigned from the orbital energy at the production μ (−38.175 Ha), not from the survey's atom table. On one atom every printed MO is sodium. The survey dictionary that lists MOs 0–5 is the print list, not the hole.

## 1. Goals

Investigate how two things change DCH dynamics and spectra of a sodium atom next to a nanoparticle:

1. The gold sphere, through the local field and, when back-propagation is on, through the field the atom itself radiates.
2. A complex absorbing potential (CAP; the Lopata non-Hermitian term on the Fock matrix), through lifetimes and peak widths.

The comparison that matters is the change produced by turning one of those pieces on, with everything else held fixed. A single “DCH + NP + CAP + pulse” job mixes the causes. Step 7 is the distance comparison on this pair: the same nanoparticle effect at surface gaps of 0.005, 0.030, and 0.060 μm, against the baseline. The baseline is the smallest gap the grid allows, one Yee pixel, with the atom at 0.02645 μm. Step 8 is the further comparison off resonance, at that same one-pixel gap, on a pair that is not chosen yet. If the effect shrinks there, it was the shared resonance. If it does not, the sphere is doing something that does not need the plasmon on the atom. Step 9 asks the same question of a dielectric built to peak on the sodium root, because the gold sphere misses it by 0.335 eV. The jobs are written. The Gaussian plot window is 1.5 to 4.5 eV, and none of D3, D4, G3, or G4 still contain a placeholder.

Two observables:

- **Dynamics.** Time-dependent hole occupations on the neutral MO basis (`mo_occ.csv` and its plot). Sudden DCH is already a non-stationary initial condition, so the hole moves even when the external electric field is zero.
- **Spectra.** Real-time absorption of the isolated atom (δ-kick) and the hybrid absorption cross section under a Gaussian pulse centered on the bright root (Meep field divided by the vacuum incident field).

Field-free hybrid jobs must not be analyzed as a cross section. $\sigma_m$ divides by $|E_{\mathrm{inc}}|^2$. With no incident source that denominator is ~0. Those runs are dynamics: occupations, the local field, and the induced dipole.

## 2. Checks already done

These runs are not part of `jobs/final`. They are why the methods in this plan are trusted. Do not rerun them as a substitute for Steps 2–8, and do not overlay their outputs on the production traces. The old inputs use retired driver names.

### 2.1 Sudden DCH occupations

`jobs/DCH/nascimento_fig8` compares PlasMol hole occupations for 3-pentanone with the digitized curves from Nascimento and co-workers (`jobs/DCH/Nascimento2020.pdf`, Fig. 8a files under `Gold Data`) and with an NWChem real-time run of the same sudden double hole (`nwchem_dch/3p_dch.out`). The valence holes that were followed are MOs 21, 23, and 24. `parse_compare.py` measures the oscillation periods of those holes.

That comparison is what established the sudden-DCH path: neutral SCF, remove two electrons from one core MO without a second SCF, propagate, and project the density back onto the neutral orbitals. The same path is used here. The orbital indices are not. 21, 23, and 24 were the 3-pentanone valence set in that figure. They are not the frontier of sodium, and the pentanone job was a closed-shell singlet. Sodium is a doublet.

Older traces in `jobs/DCH/mo_tracking_*` used `"driver": "dch"` and `dch_*` keys. Those names are not migrated. They are not valid production inputs.

### 2.2 Lopata CAP broadening

The CAP is the non-Hermitian term of Lopata and Govind, *J. Chem. Theory Comput.* **9**, 4939 (2013). Orbitals above a vacuum level $\varepsilon_0$ are damped; orbitals below it are not. PlasMol adds that term inside the Fock build.

Two checks sit on disk:

- Water, with and without the CAP, in `jobs/H2O` (`run_631g*`, `run_631g*_noCAP`, and the aug-cc-pVTZ pair). The methodology notes compare that kind of spectrum with the electron-energy-loss data Lopata and Govind used. The width depends on the functional and the basis, because those move the orbital energies relative to $\varepsilon_0$.
- A tune on 3-pentanone, in `jobs/DCH/broadening`, gave $\mu = 0.342720$ and $\varepsilon_0 = 0.003028$ Ha. That pair is not the production value. The sodium tune is `jobs/final/Step_2/tune.json`, run into `tune.log`: $\mu = 0.351340$ and $\varepsilon_0 = 0.009767$ Ha. $\mu$ is part of LC-ωPBE (Libxc’s default is 0.4), so every sodium file uses 0.351340, CAP on or off. $\varepsilon_0$ is tuned at that $\mu$ and is written only inside `molecule.cap`. The tune input itself still says `"tune"` for both fields, so it can be run again. Do not paste the numbers back into `tune.json`.

### 2.3 Hybrid FDTD against the point-dipole model

`jobs/Au_Na` is a sodium atom next to a gold sphere, in water. PlasMol’s hybrid Fourier spectra were computed in the parallel and perpendicular channels and compared with the quasistatic Gersten–Nitzan model (a point-polarizable molecule plus the sphere’s Mie multipoles). The write-up is `doc/docs/methodology.md` (validation section) and `jobs/Au_Na/PlasMol_Parallel/TeX/hybrid_spectra.tex`. The model notes are `jobs/Au_Na/Models/Gersten-Nitzan` and the two-oscillator reduction in `jobs/Au_Na/Models/COM`.

With `Au_JC_visible`, host index 1.33, and multipoles through $\ell_{\max}=25$, the joint fit is $\omega_m \approx 2.055$ eV, $\gamma_m \approx 0.076$ eV, $\alpha_0 \approx 0.43$ nm$^3$. Parallel and perpendicular peak positions agree to within a few tens of meV. The mean-squared error on the peak-normalized spectra is about $10^{-3}$. The same protocol with the Rakić gold dielectric reproduces both channels, so the match is not an artifact of one table. That is the alignment of the Meep hybrid loop (drive, local field, induced dipole, vacuum deconvolution, and the two orientations) against a closed classical dipole model. No separate Draine discrete-dipole-approximation program is in the tree.

This campaign uses a sodium atom and the Rakić gold dielectric, and it does not reuse that fit. The old sodium line was a 6-31G* root near 2.109 eV. The production root is the aug-cc-pVTZ linear-response value, 2.004849 eV. The sphere is 25 nm of `Au`, not `Au_JC_visible`, and $\omega_m$, $\gamma_m$, and $\alpha_0$ are not copied into any JSON. One candidate sphere was a 35 nm silver particle in water, Mie peak 418.5 nm with Meep’s `Ag` dielectric. It is not selected. The Mie script and the bare-sphere flux jobs were removed with the pair-choice step. There is no flux input on disk. The classical spectrum used to place Step 9 is `jobs/final/Step_9/dielectric.py`, dipole and quadrupole only.

## 3. Why the new runs are factored

| Factor | Off | On | What it isolates |
| --- | --- | --- | --- |
| DCH | Neutral atom | Sudden double hole on MO 0 | The hole itself, rather than the response of the neutral doublet |
| CAP | No `molecule.cap` | Static CAP. $\varepsilon_0 = 0.009767$ Ha | Lifetime and broadening, rather than a coherent oscillation that does not decay |
| NP | No nanoparticle | 25 nm `Au` sphere. Baseline gap one Yee pixel. Step 7 changes only the gap | The plasmonic scatterer |
| Back-propagation | Atom feels $\mathbf{E}$ but does not source it | Induced dipole is injected into Meep | Whether the sphere only reshapes the incident field, or also responds to the field the atom radiates |
| Drive | None | δ-kick, or Gaussian at 0.618422 μm | External field versus free evolution of the sudden hole |

Back-propagation exists only inside a Meep cell.

### Shared electronic structure

If any of these change between two runs, the difference is no longer one factor. Step 8 is the exception: it is a different pair, still unfilled. Step 9 changes only the sphere dielectric, from `Au` to `NaDrude`.

| Item | Choice | Note |
| --- | --- | --- |
| Geometry | `Na.xyz`, atom at the origin | Copied beside each resonant JSON. Paths resolve from the JSON's directory. Units angstrom. Do not edit the file. Survey, D1, D2, and K stay on that frame. The D1 orientation check was run. Its rotation is on the hybrid D3 and D4 copies and does not move the origin nucleus |
| Basis | `aug-cc-pVTZ`, Cartesian (6 $d$, 10 $f$) | `basis_coords` is `cartesian`. Not the campaign's old 6-311G*, and not the Au–Na 6-31G* |
| Charge / spin | 0 / 1 | Doublet. Parent spin forces UKS. The double hole keeps spin 1 |
| XC | LC-ωPBE. $\mu = 0.351340$ | Same value on every sodium file, CAP on or off. Do not omit `lrc_parameter`. Do not paste the pentanone 0.34272 or Libxc's default 0.4 |
| Core MO | Index 0, two electrons | Na 1s. The survey's 0–5 list is not the hole |
| Plot MOs | Every logged MO. `watch_indices` is omitted | Logging covers MO 0 through the neutral LUMO+1. Here that is 0 through 6, because the β LUMO is index 5. Step 8 still has the three placeholder strings |
| Molecule-only time | `dt` 0.05 au. Dynamics end at 400 au. Kicks end at 4000 au | |
| Hybrid time | `dt` 0.1 au, `t_end` 10000 au | PlasMol snaps `dt` to the Meep step (~0.10007 au) at Courant 0.5 and resolution 689. Every hybrid job uses this pair. Do not overlay a 0.05 au trace on a 0.1 au trace as if they were one grid; compare only the overlapping window |
| Medium | 1.33 | Water. Every Meep cell. Molecule-only runs have no classical medium |
| Sphere | `Au`, radius 0.025 μm, center at the origin | Baseline molecule x is 0.02645 μm. One exact pixel would be 0.026451379 μm. The field read interpolates; the coordinate is not moved. Step 7 uses 0.030, 0.055, and 0.085 μm. Those last two sit in the PML of `cell_length` 0.2 |
| Gaussian | Wavelength 0.618422 μm, `fwidth` 2.0. Plot window 1.5 to 4.5 eV | Linear-response 3s → 3p root, not 589 nm and not 2.109 eV. K1 is the real-time check. δ-kick jobs stay at 1.5 to 15 eV |

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
| 2 | Tuned. Survey reprinted at 0.351340 | Which MO is the Na 1s, and what μ and ε₀ are |
| 3 | Ready | Field-free DCH, one factor at a time (D1–D6) |
| 4 | Ready | δ-kick absorption, ± DCH, ± CAP (K1–K4) |
| 5 | Ready | Gaussian hybrid cross section, ± DCH, ± NP, ± CAP (G1–G8). Plot window 1.5 to 4.5 eV |
| 6 | Ready | Two controls on G4, after G4 itself moves. Plot window 1.5 to 4.5 eV |
| 7 | x written. Two cells still inside the PML | Three other gaps, after G4 − G2 is nonzero. Widen the 0.030 μm and 0.060 μm cells before launch |
| 8 | Placeholder | The key runs again, on a pair whose resonances do not overlap |
| 9 | Written. No placeholder | A sphere built to peak on the sodium root. `NaDrude` via `material_file`. Not launched |

### Step 1 — literature and the valence screen

Done. Notes: `jobs/final/Step_1/README.md`. No production input. Each literature molecule is entered with the DOI, the question that paper asked, and the result. The folders under `Step_1/Molecules/` are the valence screen. trans-Thioindigo is `Molecules/transthioindigo/`: PySCF LC-ωPBE at the screen value μ = 0.34272, bright root 2.976 eV (417 nm, $f = 0.29$) on the existing PBE0 geometry. That μ is not copied into the later JSON. The benzene maximum is 543 nm. Most of that gap is the functional. A 35 nm `Ag` sphere in water peaks at 418.5 nm. That candidate is not selected. No literature molecule in the screen is a bright line on that plasmon or on the 540 nm gold plasmon. Thioindigo is not in the double-core-hole literature. Its μ tune, `LC-wPBE mu/transthioindigo/tune.json`, has not been run. The sudden double hole used from Step 3 on is the pentanone initial condition in that note, adapted to the sodium doublet, not a two-site free-electron-laser spectrum.

### Step 2 — core survey and the tune

`jobs/final/Step_2/survey.json`. Driver `core_hole`. Neutral SCF, then the atoms that contribute to MOs 0 through 5, then exit. The table is written to `mo_survey.txt`. The log beside it is `log.out`. No ionization and no propagation. `t_end` is unused.

The log on disk is the μ = 0.351340 launch. It overrides Libxc's default 0.4. Every MO 0–5 is 99–101% Na: 100.17, 99.65, 100.85, 101.05, 101.38, 99.76. That table cannot name the shell. The shell assignment is a separate UKS at that μ, same basis and spin: index 0 is 1s (−38.175 Ha), 1 is 2s, 2–4 are 2p, 5 is the 3s HOMO (−0.192488 Ha). The production hole is index 0. SCF energy −162.20671104 Ha.

`tune.json` is the Koopmans scan, driver `tune`. It wrote μ = 0.351340 and ε₀ = 0.009767 Ha into `tune.log` (29.3 s) and was not overwritten with those numbers. For spin 1 the cation is closed-shell and the neutral α HOMO is the one in the Koopmans target. The first α virtual sits on ε₀ (+0.009767 Ha, the empty 3p). The CAP does not damp it. The first orbital above the cutoff is index 9.

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
- D3 − D1: sphere, atom allowed to radiate, no CAP. Compare only the first 400 au. The time steps differ (0.05 au versus the Meep step).
- D4 − D1: sphere present, atom does not source the cell. If D4 matches D1 and D3 does not, the nanoparticle effect is the radiated field coming back.
- D5 − D3 and D6 − D4: CAP on top of each nanoparticle case.
- D3 − D4 and D5 − D6: back-propagation at fixed CAP.

D1 and D2 stay on the stock geometry and can checkpoint. D3–D6 have no Meep source and cannot checkpoint. D1 has been run to 400 au. `jobs/final/orientation_check.txt` is the rotation pasted into Step 3 D3–D6 and into the Step 7 and Step 9 copies of D3 and D4: 84.1114 degrees about `[0.0, -0.89999759, 0.43589487]`. The D1 dipole is below 2e-10 au and the nucleus is at the origin, so the rotation does not aim a real emission axis and it does not move `Na.xyz`. K1’s three dipole files are the valence axis if a Gaussian job needs one; they are not this rotation.

These six files have no placeholders. `watch_indices` is omitted, so the occupation plot is every logged MO.

Do not treat D3–D6 as cross sections.

Neutral field-free dynamics are not in this matrix. With no hole and no field the density does not move.

### Step 4 — δ-kick spectra

Molecule only. Absorption driver, `polarization: "full"`. Kick strength 0.001 au. `t_end` 4000 au. Fourier window `gamma` 0.01, plotted from 1.5 to 15 eV. These jobs can checkpoint. The bright root at 2.005 eV is inside that window. Do not raise `max_ev` unless K1 shows weight above 15 eV, and then raise all four files together.

| ID | DCH | CAP |
| --- | --- | --- |
| K1 | off | off |
| K2 | off | on |
| K3 | on | off |
| K4 | on | on |

K1 is the ordinary spectrum of the sodium atom, and it is the real-time check of the 2.004849 eV linear-response root. The plotted curve averages x, y, and z, so a rotation would not change it. K3 − K1 is the sudden hole with no CAP. K2 − K1 and K4 − K3 are the CAP on a given initial state. `gamma` is only the Fourier window (~0.27 eV half-width). It is not a substitute for the CAP. The dipole is already small by a few hundred au, so 4000 au does not sharpen the spectrum unless `gamma` is lowered.

DCH kicks write `mo_occ.csv` under `x_dir/`, `y_dir/`, and `z_dir/`. K1 and K2 have no core hole. K3 and K4 omit `watch_indices`. None of the four still contains a placeholder.

Step 1 is the linear-response check of each neutral molecule in the screen. It is not a substitute for K1. The Gaussian wavelength was taken from a separate UKS linear-response calculation at this basis and this μ, not from Step 1.

### Step 5 — Gaussian hybrid cross sections

Same cell and the same Gaussian in every file. The sphere, when present, is 25 nm `Au`. The source wavelength is 0.618422 μm, `fwidth` 2.0. The plotted window is 1.5 to 4.5 eV. `gamma` is 0. The published curve is `cross_section`. Polarization `parallel` puts $\mathbf{E}$ along the nanoparticle–molecule axis and rearranges the source so $\mathbf{k}\perp\mathbf{E}$. Polarization `single` is the empty cell: no nanoparticle, source left as written. Back-propagation is on. No checkpoints. The empty-cell files still put the atom at 0.02645 μm, so the vacuum reference is the same coordinate.

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

The empty-cell Gaussian is not a substitute for Step 4. K1–K4 stay the real-time molecular reference. The empty cell checks that Meep plus deconvolution recovers that atom before the sphere is introduced. Do not reuse a vacuum reference computed with a different source or a different `dt`.

### Step 6 — two controls on G4

Not part of the grid. Run them only after the resonant G4 itself shows a nanoparticle effect. The plot window is 1.5 to 4.5 eV.

| File | Change from G4 |
| --- | --- |
| `G4_perpendicular` | Polarization `perpendicular` |
| `G4_no_backprop` | `back_propagation` false. The atom feels the cell and does not source it |

One of each is enough. Do not repeat them across CAP and DCH until G4 has moved. Do not copy them into Step 8 unless the detuned G4 has moved and D3 − D4 does not already explain it. Do not repeat them across the Step 7 gaps.

Single core holes are out of this plan: removing one electron forces a different open-shell initial state. The sodium ground state is already a doublet; the production hole is the double vacancy on MO 0, which keeps that spin.

### Step 7 — distance

Notes: `jobs/final/Step_7/README.md`.

Not part of the grid. Run it only after resonant G4 − G2 is nonzero at 0.02645 μm. The baseline gap is not repeated. Each of 0.005, 0.030, and 0.060 μm has D3, D4, and G4, and nothing else. D1 and G2 have no sphere, so the differences are still D3 − D1 and G4 − G2.

The molecule x values are written: 0.030, 0.055, and 0.085 μm. The G4 plot window is 1.5 to 4.5 eV. The non-PML interior of a 0.2 μm cell with a 0.05 μm PML ends at 0.05 μm. The 0.005 μm gap fits. The 0.030 and 0.060 μm gaps put the atom in the PML. Widen those two cells before launch. Do not change the gap values to make the cell fit.

Paste the Step 3 orientation into these D3 and D4 files. If the gap changes D3 and G4 but D4 stays with D1, the distance dependence is the radiated field.

### Step 8 — detuned pair

Notes: `jobs/final/Step_8/README.md`.

The resonant grid asks what the nanoparticle does when the plasmon sits on the bright root. This step asks whether that effect survives when it does not. The Gaussian is still centered on the molecule's bright root. The nanoparticle is a different resonance. Nothing in this directory was edited when the sodium series was filled. Every molecule field and every nanoparticle field is still `PLACEHOLDER_DETUNED`.

The key set is the survey, D1, D3, D4, K1, K3, and G1–G4. D3 − D1 is the sphere when the molecule radiates. D4 − D1 is the sphere when it does not. G4 − G2 is the same question in the cross section. K1 is the measurement that the plasmon misses the root. The CAP rows, the Step 6 controls, and the Step 7 gaps are not repeated. The surface gap stays one Yee pixel, the resonant baseline, so the comparison is not also a change of distance. It is not 0.015 μm.

If the detuned pair keeps the sodium atom, copy the molecule tokens, including μ, ε₀, the core index, and the omitted watch list, and change the nanoparticle. Do not rerun D1, K1, K3, G1, or G2 in that case. There is no bare-sphere flux job.

The orientation check is the Step 3 command, pointed at `Step_8/D1/field_p_D1.csv`. Paste the rotation into Step 8 D3 and D4 only.

### Step 9 — a dielectric on the sodium root

Notes: `jobs/final/Step_9/README.md`.

The gold sphere of Steps 2–7 misses the sodium root by 0.335 eV. This step replaces the material and nothing else, so a difference from G4 − G2 or D3 − D1 is the dielectric. The designed function is one Drude pole, ε∞ = 1, σ = 1, γ = 0.053 eV, ω_p = 4.505664 eV. γ is the Rakić free-electron damping and was not fitted. ω_p was raised from the quasistatic value 4.272 eV until the Mie absorption of a 25 nm sphere in water (n = 1.33) peaked at 2.004849 eV. Extinction peaks at 2.005787 eV. The script is `dielectric.py`. It is not a PlasMol driver.

The jobs are `D3/D3.json`, `D4/D4.json`, `G3/G3.json`, and `G4/G4.json`. Each is the Step 3 or Step 5 file with the nanoparticle block replaced. The material is `NaDrude`, and `material_file` is `../na_drude.py` (`doc/docs/custom_material.md`). That name is not an existing Meep material, so it is not `Au`. `Na.xyz` sits beside each JSON. The plot window on G3 and G4 is 1.5 to 4.5 eV. None of the four files contain a placeholder. Do not retune μ or ε₀, and do not change the radius. Do not use this dielectric as the Step 8 pair.

## 5. Setup

Snippets match the resonant JSON on disk. D3–D6 are the only jobs that gain `molecule.rotation`, and only after Step 3’s D1 check. Step 8 still uses the `PLACEHOLDER_DETUNED` spellings and is not shown here.

### 5.1 Survey

```json
{
  "settings": {"dt": 0.05, "t_end": 1.0, "driver": "core_hole"},
  "molecule": {
    "geometry": "Na.xyz",
    "geometry_units": "angstrom",
    "charge": 0,
    "spin": 1,
    "basis": "aug-cc-pVTZ",
    "basis_coords": "cartesian",
    "xc": "HYB_GGA_XC_LC_WPBE",
    "lrc_parameter": 0.351340,
    "core_hole": {"mo_removal_index_dict": {"0": 2, "1": 2, "2": 2, "3": 2, "4": 2, "5": 2}}
  }
}
```

Electron counts in that dictionary are ignored. The driver prints atom contributions and exits. The 0–5 list is the print list.

### 5.2 Dynamics without a sphere (D1, D2)

`"driver": "quantum"`. No molecule source. No `plasmon` section. D2 adds `molecule.cap`. Both set:

```json
"core_hole": {
  "mo_removal_index_dict": {"0": 2},
  "mo_occ_filepath": "mo_occ.csv"
}
```

There is no `watch_indices` key. Omitting it plots every logged MO. Checkpointing is allowed.

CAP block when it is on. It does not replace the functional or $\mu$:

```json
"cap": {"type": "static", "gam0": 1, "xi": 0.5, "eps0": 0.009767, "clamp": 100}
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
  "nanoparticle": {"material": "Au", "radius": 0.025, "center": [0, 0, 0]},
  "molecule": {
    "position": [0.02645, 0, 0],
    "tolerance_field_e": 1e-20,
    "back_propagation": true
  }
}
```

D4 and D6 set `back_propagation` to false. Do not turn on mirror symmetries. The induced dipole will not respect them. At `dt` 0.1 and Courant 0.5 the resolution is 689 and one pixel is 0.001451379 μm. The surface gap at x = 0.02645 is 0.00145 μm, which the position check accepts. The coordinate is stored as written. The field sample is an interpolation of the surrounding Yee points, not a snap to a pixel center.

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

Same sphere, cell, PML, gap, and Gaussian in every file. `dt` 0.1 and `t_end` 10000 must match as well. Parallel mode overwrites the source component onto the nanoparticle–molecule axis. The wavelength is 0.618422 μm. The plot window is 1.5 to 4.5 eV.

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
    "max_ev": 4.5
  }
}
```

The source block, in every G file of Steps 5–7 and in both Step 6 controls:

```json
"additional_parameters": {"wavelength": 0.618422, "fwidth": 2.0}
```

Empty cell (G1, G2, G5, G6): same simulation and source, no `nanoparticle` block, `"polarization": "single"`, component along x. DCH members add `molecule.core_hole`. The occupation CSV is written next to the production fields (`fields/mo_occ.csv`), not beside the vacuum reference. CAP members add `molecule.cap`. `back_propagation` is true on G3, G4, G7, and G8. The back-propagation-off control is Step 6, not another row of this table.

## 6. Still open

Status as of 2026-10-01.

- [x] Step 1 literature note and valence screen, including trans-thioindigo at 417 nm. No literature molecule sits on the silver plasmon near 418 nm or the gold plasmon near 540 nm.
- [x] Thioindigo core survey was not selected and is not in Step 2.
- [x] Resonant pair: Na, aug-cc-pVTZ, Cartesian, spin 1, LC-ωPBE, μ = 0.351340, ε₀ = 0.009767 Ha, core MO 0, 25 nm `Au`, atom at 0.02645 μm. `Na.xyz` copied beside each resonant JSON.
- [x] Linear-response bright root 2.004849 eV written as wavelength 0.618422 μm on the Gaussian jobs. Threefold 3s → 3p, oscillator strength 0.327 on each component.
- [x] `watch_indices` removed on the resonant DCH files, so the plot is every logged MO (0 through 6).
- [x] Step 9 dielectric specified classically and written. ω_p = 4.505664 eV, γ = 0.053 eV. `na_drude.py` assigns `meep.materials.NaDrude`. D3, D4, G3, and G4 are on disk and contain no placeholder. Not launched.
- [x] Resonant survey reprinted at μ = 0.351340. MOs 0–5 are 99–101% Na. The 1s index stays 0.
- [x] Gaussian plot window set to 1.5–4.5 eV on Steps 5–7, both Step 6 controls, and Step 9 G3 and G4. Not launched.
- [x] Step 7 molecule x written: 0.030, 0.055, and 0.085 μm.
- [ ] Widen `cell_length` on the Step 7 gaps of 0.030 μm and 0.060 μm before launch. At 0.055 and 0.085 μm the atom is inside the PML of `cell_length` 0.2.
- [x] Resonant D1 finished at 400 au. The orientation check is pasted into Step 3 D3–D6 and the Step 7 and Step 9 D3/D4 copies. The dipole is below 2e-10 au, and the origin nucleus does not move.
- [ ] One short D2 (DCH + CAP) and one short D3 (DCH, sphere, no source) before the full 400 au and 10000 au propagations.
- [ ] Hybrid jobs cannot checkpoint. D3–D6 and G1–G8 have to finish in one process. D1, D2, and K1–K4 can restart. Steps 7 and 8 have the same split.
- [ ] K1 and K2 can launch now. They are the real-time check of the 0.618422 μm line.
- [ ] $\sigma_m$ is for shape comparison. Overlay peak-normalized curves when the claim is that a peak moved or broadened.
- [ ] After resonant G4 − G2 is nonzero, run Step 7 and compare D3 − D1 and G4 − G2 at each gap with the 0.02645 μm baseline.
- [ ] Choose the detuned pair and replace every `PLACEHOLDER_DETUNED` token in Step 8. Do not reuse the Step 9 dielectric as that pair. The gap stays one Yee pixel.
- [ ] Run Step 9 D3, D4, G3, and G4 against the gold results. D3 and D4 have no placeholder. G3 and G4 wait on the resonant window. The material name is `NaDrude`.
