# Progress

This directory is the run list for the campaign in `plan.md`. Read the steps in order. Each `Step_n` is one stage. A stage that has several runs keeps those runs in folders named by the plan (`D1`, `K3`, `G4`, …).

Step 1 is done. The resonant pair is a sodium atom and a 25 nm gold sphere (Meep `Au`, Rakić). Steps 2–7 and Step 9 are filled and contain no `PLACEHOLDER_`. The Gaussian plot window is 1.5 to 4.5 eV. Step 7 molecule x is written: 0.030, 0.055, and 0.085 μm. The 0.005 μm gap fits in the present cell. The 0.030 μm and 0.060 μm gaps put the atom in the PML of `cell_length` 0.2, so those two cells still have to be widened before launch. Step 9 is the dielectric `NaDrude` from `Step_9/na_drude.py`, on D3, D4, G3, and G4, and it has not been launched. The detuned pair is not chosen. Step 8 is the only directory that still contains `PLACEHOLDER_`. Do not launch a file that still contains `PLACEHOLDER_`.

Run a JSON from the directory that contains it, and pass `--log log.out` so the run record is saved beside the input. Geometry paths are relative to that file. Do not launch two runs in the same directory.

```bash
python -m plasmol.main survey.json --log log.out
```

Spectra, occupations, and fields are the files named in each JSON. The core-orbital survey also writes `mo_survey.txt`. Nothing from these steps is left only in the terminal.

## Where things stand

| Step | Status | What it answers |
| --- | --- | --- |
| `Step_1` | Done | Which molecules have a bright valence line, including trans-thioindigo, and whether any of them sits on a metallic plasmon |
| `Step_2` | Tuned. Survey reprinted at 0.351340 | Na 1s index, and the production μ and ε₀ |
| Check | Done | D1 dipole is below 2e-10 au. The reported rotation is pasted on the hybrid D3 and D4 copies. The nucleus is at the origin, so it does not move |
| `Step_3` | D1–D6 and `D1_10k` finished | Field-free DCH, one factor at a time |
| `Step_4` | Ready | δ-kick absorption, ± DCH, ± CAP. K1 checks the 2.004849 eV root in real time |
| `Step_5` | Ready | Gaussian hybrid cross section, ± DCH, ± NP, ± CAP. Plot window 1.5 to 4.5 eV |
| `Step_6` | Ready | Two controls on G4, not part of the main grid. Plot window 1.5 to 4.5 eV |
| `Step_7` | x written. Two cells still inside the PML | How the nanoparticle effect depends on the surface gap. Widen the 0.030 μm and 0.060 μm cells before launch |
| `Step_8` | Placeholder | The key runs again, on a pair that does not share a resonance |
| `Step_9` | Written. No placeholder | A sphere whose resonance is the sodium root. `NaDrude`, not launched |

trans-Thioindigo is `Step_1/Molecules/transthioindigo/`. Its bright root is 417 nm (f = 0.29). A 35 nm silver sphere in water peaks at 418.5 nm. That pair is not selected. The thioindigo core survey was not selected and is not in Step 2. The μ tune input is `Step_1/LC-wPBE mu/transthioindigo/tune.json` and has not been run.

## What is filled

Sodium, Steps 2–7 and Step 9. Step 8 was not touched. `Na.xyz` is the reference geometry (`jobs/Au_Na/Na_Reference/Na.xyz`): one atom at the origin. A copy sits beside every resonant JSON, and beside each Step 9 JSON, because the path is resolved from that file's directory. `geometry_units` is `angstrom`. The origin makes the unit choice irrelevant to the coordinate.

| Field | Value |
| --- | --- |
| Basis | `aug-cc-pVTZ`, `basis_coords` `cartesian` |
| Charge / spin | 0 / 1. UKS doublet |
| XC | `HYB_GGA_XC_LC_WPBE`, `lrc_parameter` 0.351340 |
| CAP, where present | static, `gam0` 1, `xi` 0.5, `eps0` 0.009767, `clamp` 100 |
| Core hole | `{"0": 2}`. Na 1s. `watch_indices` omitted |
| Nanoparticle | `Au`, radius 0.025 μm, center at the origin. Step 9 replaces `Au` with `NaDrude` and changes nothing else |
| Baseline position | `[0.02645, 0, 0]` μm. Steps 3–6 and Step 9 |
| Step 7 position | `[0.030, 0, 0]`, `[0.055, 0, 0]`, `[0.085, 0, 0]` μm. Radius plus gaps of 0.005, 0.030, and 0.060 μm |
| Gaussian | `wavelength` 0.618422, `fwidth` 2.0. Plot window 1.5 to 4.5 eV. δ-kick jobs stay at 1.5 to 15 eV |

`tune.json` still contains the strings `"tune"` for μ and for ε₀. The run is `tune.log`: optimal μ = 0.351340, vacuum level ε₀ = 0.009767 Ha. Leave the input as `"tune"` so it can be repeated. The 3-pentanone pair, μ = 0.34272 and ε₀ = 0.003028 Ha, is not this tune. Libxc's default μ = 0.4 is not this tune either.

The survey log and `mo_survey.txt` are the launch at μ = 0.351340. Every MO 0–5 is about 100% Na (100.17, 99.65, 100.85, 101.05, 101.38, 99.76), which does not identify 1s. At the production μ the α ladder is index 0 = 1s (−38.175 Ha), 1 = 2s, 2–4 = 2p, 5 = 3s HOMO (−0.192488 Ha), 6–8 = empty 3p sitting on ε₀. SCF energy −162.20671104 Ha. The production hole stays index 0 either way.

Logging covers MO 0 through LUMO+1. The β LUMO is index 5, so the plotted set is 0 through 6. That is why `watch_indices` was removed instead of being filled with three integers. Step 8 still has the three detuned placeholder strings.

The bright root is a UKS linear-response TDDFT at this basis, this μ, and spin 1. States 1–3 are the 3s → 3p line at 2.004849 eV, each with oscillator strength 0.327 (sum 0.980). The wavelength written into the JSON is 0.618422 μm. It is not the experimental 589 nm line and not the old 6-31G* root near 2.109 eV. K1 is the real-time measurement of the same root. The linear-response number is only the Gaussian center.

The baseline gap is one Yee pixel, not 0.015 μm. At `dt` 0.1 au and Courant 0.5 the resolution is 689, so the pixel is 0.001451379 μm and radius plus one pixel is 0.026451379 μm. The files use 0.02645. That is 1.4 pm short of a pixel and inside the position check's tolerance. The coordinate is stored as written. The field sample interpolates the surrounding Yee points and does not snap the atom onto a pixel center.

## Still a placeholder

The resonant window tokens and the three Step 7 x tokens are filled. What remains is the detuned pair.

| Token | Where | What replaces it |
| --- | --- | --- |
| Every `PLACEHOLDER_DETUNED` | Step 8 only | The detuned pair, when it is chosen. Not the Step 9 dielectric |

K1, K2, D1–D6, K3, K4, Step 5 G1–G8, both Step 6 controls, Step 7, and all of Step 9 contain no placeholder. The Gaussian plot window is 1.5 to 4.5 eV. δ-kick jobs stay at 1.5 to 15 eV. Step 7 molecule x is 0.030, 0.055, and 0.085 μm. The 0.005 μm gap fits in the present cell. The 0.030 μm and 0.060 μm gaps still need a larger cell before launch, because 0.055 and 0.085 sit in the PML.

## Locked for this series

These are the same in every resonant JSON from Step 2 on.

| Item | Value |
| --- | --- |
| Basis | `aug-cc-pVTZ`, Cartesian Gaussians |
| Charge / spin | 0 / 1. The double hole on one MO keeps spin 1 and clears both spin-orbitals of that index |
| XC | LC-ωPBE. μ = 0.351340 on every file, CAP on or off |
| CAP, when present | static, `gam0` 1, `xi` 0.5, `clamp` 100, `eps0` 0.009767 Ha |
| Core hole, when present | 2 electrons from MO 0 |
| Quantum time | `dt` 0.05 au. Field-free dynamics `t_end` 400 au. δ-kick spectra `t_end` 4000 au |
| Hybrid time | JSON `dt` 0.1 au, `t_end` 10000 au. PlasMol snaps `dt` to the Meep step (about 0.10007 au) at Courant 0.5. Every hybrid job uses this same pair |
| Medium | refractive index 1.33 (water). Every Meep cell |
| Gap | One Yee pixel in Steps 3–6 and in Step 9. Step 7 uses 0.005, 0.030, and 0.060 μm. Step 8, once filled, uses the same one-pixel gap, not 0.015 μm |

μ is the range-separation parameter of LC-ωPBE. It was tuned for this atom and it is the same number on every sodium file. A CAP-off run with a different μ is not the partner of the CAP-on run. ε₀ is the CAP threshold, in hartree, tuned at that μ. It appears only in `molecule.cap`. The first empty 3p lies on ε₀, so the CAP does not damp it. Switching the series to another functional means editing every JSON from Step 2 on and retuning ε₀.

D1 and D2 have no `molecule.rotation` and stay on `Na.xyz`. The orientation check has been run. Its block is pasted into Step 3 D3–D6, and into the Step 7 and Step 9 copies of D3 and D4. Do not edit the geometry file. The sodium nucleus is at the origin, so that rotation does not move it.

### Step 1 — DCH literature, and the valence screen

Done. The note is `Step_1/README.md`. No production JSON. For each molecule it records the DOI, what the paper was trying to measure or calculate, and what it found. The folders under `Step_1/Molecules/` are the valence-absorption screen, and trans-thioindigo is one of them. Its bright root is 417 nm, on the 418.5 nm silver plasmon. None of the double-core-hole literature molecules are. The sudden double hole used from Step 3 on is the pentanone initial condition, on the sodium doublet rather than a closed-shell singlet, not a two-site free-electron-laser spectrum.

### Step 2 — core MO survey, resonant molecule

`survey.json`. Driver `core_hole`. It builds the neutral atom, writes the atoms that contribute to MOs 0 through 5 into `mo_survey.txt`, and exits. The same table is in `log.out` when the job is launched with `--log log.out`. It does not remove electrons and it does not propagate. `t_end` is unused.

The log on disk is the μ = 0.351340 launch. Every MO 0–5 is about 100% Na. That print still does not name the shell. The 1s index is 0 from the orbital energy.

### Step 3 — field-free DCH dynamics, resonant pair

No external electric field. The drive is the sudden hole. D1 and D2 are molecule-only (`quantum`) and can checkpoint. D3–D6 are hybrid (`plasmol`), have no Meep source, and cannot checkpoint. The sphere is 25 nm `Au`. The atom is at 0.02645 μm.

| File | CAP | Nanoparticle | Back-propagation |
| --- | --- | --- | --- |
| `D1/D1.json` | off | no | — |
| `D2/D2.json` | on | no | — |
| `D3/D3.json` | off | yes | on |
| `D4/D4.json` | off | yes | off |
| `D5/D5.json` | on | yes | on |
| `D6/D6.json` | on | yes | off |

Read D2 − D1 as the CAP on the free hole. D3 − D1 as the nanoparticle when the atom is allowed to radiate. D4 − D1 as the nanoparticle when it is not. If D4 matches D1 and D3 does not, the nanoparticle effect is the radiated field coming back.

D4 finished at 10007 au with no error. The local field stayed at zero, so the sphere did not drive the atom. The dipole does not match D1: it stays under 1e-6 until 1089 au, then reaches 0.91 near 2850 au. The three 2p holes, equal on D1, are split on D4.

D5 finished at 10007 au with no error. Back-propagation is on, and the local field stays below 4e-8. The dipole peaks at 0.002. The three 2p holes stay equal, and the 3s hole ends at 0.98. The plotted holes sum to 2.08 at the start and 4.11 at the end. CAP is on, so that increase can be absorption as well as promotion out of MOs 0–6.

D3 finished at 10007 au with no error. Back-propagation is on. The local field peaks at 1.1e-5. Through 400 au the dipole stays under 3e-10, as on D1. After that it follows D4: it passes 0.1 near 2060 au and peaks at 0.81 near 7230 au, and the 2p holes split. The plotted holes sum to 2.07 at the start and 3.26 at the end. There is no CAP, so those extra holes are electrons that left MOs 0–6.

D6 finished at 10007 au with no error. The half-step retry never fired. Back-propagation is off and the local field is zero. The dipole peaks at 0.002. The logged holes match D5 at the 0.001-electron level: the 2p holes stay equal, the 3s hole ends at 0.98, and the plotted holes sum to 4.11. Turning the radiated field off does not change this CAP run.

D1 finished at 400 au with no error. The 1s hole stays near 2. The 2p hole populations stay equal to each other, and the dipole stays below 2e-10 au, so there is no radiating axis. `orientation_check.txt` still reports a noise-direction rotation, 84.1114 degrees about `[0.0, -0.89999759, 0.43589487]`. That object is `molecule.rotation` on D3–D6. D1 and D2 stay on the stock frame, so D2 − D1 remains a same-frame CAP comparison. The same block is on the Step 7 and Step 9 D3 and D4 copies. It is not on the Gaussian jobs.

`Step_3/D1_10k` is that same D1 input with `t_end` 10000 instead of 400. It does not replace the 400 au run. It finished at 10000 au with no error. Through 400 au the holes match D1 to 3e-12, and the dipole stays under 1e-10. After that the free atom does what D3 and D4 do: the dipole passes 0.1 near 3140 au, peaks at 0.86 near 7970 au, and the 2p holes split. There is no sphere in this run. D3 and D4 reach a dipole of 0.1 about 1000 au earlier. The CAP runs, D5 and D6, do not grow this dipole. D2 also finished at 400 au with no error. D3–D6 are the hybrid runs, out to about 10007 au. Their first launch stopped when the Magnus residual leveled off near 2e-12, just above the requested 1e-12. A stalled residual below 1e-8 is accepted; the JSON value of `pc_convergence` is still 1e-12. D6 still died at 2730 au, once at 1.3e-9 and again at 9.4e-7. That one interval is retried as two steps of `dt/2`. A half-step that also fails is still an error. The partial traces are `D6/partial_t2730/` and `D6/partial_t2730_rerun/`. `rotated_geometry.xyz` is the same origin atom as `Na.xyz`.

Do not treat D3–D6 as cross sections. There is no incident field to divide by. The outputs are `mo_occ.csv`, `field_e_*.csv`, and `field_p_*.csv`. The occupation plot is MOs 0 through 6.

D1 and D2 run at `dt` 0.05 to 400 au. D3–D6 run at the Meep step, `dt` 0.1 to 10000 au. Compare occupations only on the overlapping window. Do not treat the two time grids as one trace.

Older traces in `jobs/DCH/mo_tracking_*` used `"driver": "dch"` and are not copied here. They are not valid production inputs. Do not overlay those CSV files on D1 or D2.

### Step 4 — δ-kick spectra, resonant molecule

Isolated atom, absorption driver, polarization `full` (x, y, and z). Kick strength 0.001 au. `t_end` 4000 au. Fourier window `gamma` 0.01, plotted from 1.5 to 15 eV. These jobs can checkpoint. A core hole writes `mo_occ.csv` under `x_dir/`, `y_dir/`, and `z_dir/`.

`gamma` = 0.01 au is a Lorentzian of about 0.27 eV half-width. The dipole is already small by a few hundred au, so 4000 au does not make the spectrum sharper than that window. It does keep the time signal if you want it. Lower `gamma` before relying on the longer trace for resolution. The linear-response root is at 2.005 eV, inside the window.

| File | DCH | CAP |
| --- | --- | --- |
| `K1/K1.json` | off | off |
| `K2/K2.json` | off | on |
| `K3/K3.json` | on | off |
| `K4/K4.json` | on | on |

K1 is the ordinary spectrum of the sodium atom and the real-time check of the Gaussian wavelength. The plotted curve is the average of the three kicks, so rotating the nuclei would not change it. The separate dipole files are the valence axis. Use them only if a Gaussian job needs that axis on +x. They are not the rotation for D3–D6. K3 − K1 is the sudden hole with no CAP. K2 − K1 and K4 − K3 are the CAP on a given initial state. `gamma` is only the Fourier window. It is not a substitute for the CAP. K1 and K2 can launch now.

### Step 5 — Gaussian hybrid cross sections, resonant pair

Same cell and the same Gaussian in every file. The nanoparticle block is present only in G3, G4, G7, and G8. The wavelength is 0.618422 μm. The plot window is 1.5 to 4.5 eV. Polarization `parallel` puts the field along the nanoparticle–molecule axis. Polarization `single` is the empty cell: no nanoparticle, and the source is left as written (x-polarized, propagating in y). Back-propagation is on. No checkpoints. The atom is at 0.02645 μm in the empty cell as well.

`gamma` is left at 0. The published curve is `cross_section`.

| File | DCH | Nanoparticle | CAP | Polarization |
| --- | --- | --- | --- | --- |
| `G1/G1.json` | off | no | off | `single` |
| `G2/G2.json` | on | no | off | `single` |
| `G3/G3.json` | off | yes | off | `parallel` |
| `G4/G4.json` | on | yes | off | `parallel` |
| `G5/G5.json` | off | no | on | `single` |
| `G6/G6.json` | on | no | on | `single` |
| `G7/G7.json` | off | yes | on | `parallel` |
| `G8/G8.json` | on | yes | on | `parallel` |

G1–G4 isolate DCH and the nanoparticle. G5–G8 are the same four with the CAP. Run G1–G4 first if the full set is too many at once. G8 − G4 is the CAP on the production hybrid (DCH + nanoparticle).

The empty-cell Gaussian is not a substitute for Step 4. K1–K4 stay the real-time reference.

### Step 6 — two controls on resonant G4

Not part of the grid. Run them only after G4 itself shows a nanoparticle effect. The plot window is 1.5 to 4.5 eV.

| File | What changes from G4 |
| --- | --- |
| `G4_perpendicular/G4_perpendicular.json` | Polarization `perpendicular` instead of `parallel` |
| `G4_no_backprop/G4_no_backprop.json` | `back_propagation` false. The atom feels the cell and does not source it |

One perpendicular run and one back-propagation-off run are enough. Do not repeat them across CAP and DCH until G4 has moved.

### Step 7 — distance, resonant pair

Not part of the grid. Run it only after G4 − G2 is nonzero at 0.02645 μm. Notes: `Step_7/README.md`.

Three surface gaps, each with D3, D4, and G4. D1 and G2 are not repeated: they have no nanoparticle. No CAP, and no perpendicular run. The detuned pair uses the one-pixel gap, not these three.

| Gap | Molecule x | Against the 0.02645 μm baseline |
| --- | --- | --- |
| 0.005 μm | 0.030 | Closer. A near field should get stronger. This cell fits |
| 0.030 μm | 0.055 | Larger gap. The atom is inside the PML until the cell is widened |
| 0.060 μm | 0.085 | Far enough that a near field should be gone. Also inside the PML |

Those x values are written in the JSON. If the gap changes D3 and G4 but D4 stays on D1, the distance dependence is the radiated field coming back. Paste the Step 3 orientation into these D3 and D4 files. The G4 plot window is 1.5 to 4.5 eV. The 0.030 μm and 0.060 μm cells are still the 0.2 μm cell, so do not launch them until that cell is widened.

### Step 8 — the same question, off resonance

The resonant grid asks what the nanoparticle and the CAP do for this sodium–gold pair. This step asks whether that nanoparticle effect is still there when the resonances do not overlap. The directory was not edited with the sodium numbers. Every token is still `PLACEHOLDER_DETUNED`.

It repeats only the runs that isolate the nanoparticle on the sudden hole. It does not repeat the CAP rows, the Step 6 controls, or the Step 7 gaps. The surface gap stays one Yee pixel, so this comparison is resonance and not distance. It is not 0.015 μm, and it is not the Step 9 dielectric. Add the controls here only if the detuned G4 itself moves and the mechanism is no longer obvious from D3 − D4.

| File | Resonant counterpart | Why it is in the key set |
| --- | --- | --- |
| `survey/survey.json` | Step 2 | The core index belongs to this molecule |
| `D1/D1.json` | Step 3 D1 | Free hole, no nanoparticle |
| `D3/D3.json` | Step 3 D3 | Nanoparticle, molecule allowed to radiate |
| `D4/D4.json` | Step 3 D4 | Nanoparticle, molecule does not source the cell |
| `K1/K1.json` | Step 4 K1 | Where the bright root is |
| `K3/K3.json` | Step 4 K3 | The same root after the sudden hole |
| `G1`–`G4` | Step 5 G1–G4 | Cross section, ± DCH, ± nanoparticle, no CAP |

D3 − D1 against the resonant D3 − D1 is the dynamics comparison. G4 − G2 against the resonant G4 − G2 is the cross-section comparison. K1 is what "detuned" means: the root this Gaussian is centered on, and a plasmon that is not there.

If this pair keeps the sodium atom, copy the molecule block from Step 3, including μ, ε₀, MO 0, and the omitted watch list, and change the nanoparticle. D1, K1, K3, G1, and G2 can be reused. The orientation check is the same command as Step 3, pointed at `Step_8/D1/field_p_D1.csv`. Paste the rotation into the Step 8 D3 and D4 only.

### Step 9 — synthetic dielectric on the sodium root

Notes: `Step_9/README.md`. Jobs: `D3`, `D4`, `G3`, `G4`. Not launched.

The 25 nm Rakić sphere absorbs at 2.340 eV. The sodium root is 2.004849 eV. This step is a one-pole Drude sphere of the same radius in the same water, with γ = 0.053 eV taken from the Rakić free-electron term and ω_p = 4.505664 eV chosen so the Mie absorption peaks on the root. The quasistatic plasma frequency, 4.272 eV, is not used: at 25 nm the peak shifts, and ω_p was raised to put it back. Extinction is 1 meV higher. `dielectric.py` writes `dielectric.txt`. It is the design script, not the material PlasMol loads.

The hybrid comparison is D3, D4, G3, and G4 against the gold results. `na_drude.py` assigns `meep.materials.NaDrude`, and each JSON points at `../na_drude.py`. See `doc/docs/custom_material.md`. `Na.xyz` sits beside each JSON. The plot window on G3 and G4 is 1.5 to 4.5 eV. None of the four files contain a placeholder. Do not retune the atom. Do not change the radius. Do not use this dielectric as the Step 8 pair.
