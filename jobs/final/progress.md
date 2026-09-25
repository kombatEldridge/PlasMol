# Progress

This directory is the run list for the campaign in `plan.md`. Read the steps in order. Each `Step_n` is one stage. A stage that has several runs keeps those runs in folders named by the plan (`D1`, `K3`, `G4`, …).

Step 1 is done. The molecule is not chosen. The nanoparticle is not chosen. Two pairs are still open: one whose resonances overlap, and one whose resonances do not. Every JSON from Step 2 on names those fields with placeholders. Do not launch a file that still contains `PLACEHOLDER_`.

Run a JSON from the directory that contains it, and pass `--log log.out` so the run record is saved beside the input. Geometry paths are relative to that file. Do not launch two runs in the same directory.

```bash
python -m plasmol.main survey.json --log log.out
```

Spectra, occupations, and fields are the files named in each JSON. The core-orbital survey also writes `mo_survey.txt`. Nothing from these steps is left only in the terminal.

## Where things stand

| Step | Status | What it answers |
| --- | --- | --- |
| `Step_1` | Done | Which molecules have a bright valence line, including trans-thioindigo, and whether any of them sits on a metallic plasmon |
| `Step_2` | Placeholder | Which MO is the core on the resonant molecule |
| `Step_3` | Placeholder | Field-free DCH on the resonant pair, one factor at a time |
| Check | After D1 | Which rotation puts the core-hole dipole on +x for D3–D6 |
| `Step_4` | Placeholder | δ-kick absorption of the resonant molecule, ± DCH, ± CAP |
| `Step_5` | Placeholder | Gaussian hybrid cross section of the resonant pair, ± DCH, ± NP, ± CAP |
| `Step_6` | Placeholder | Two controls on the resonant G4 point, not part of the main grid |
| `Step_7` | Placeholder | How the resonant nanoparticle effect depends on the surface gap |
| `Step_8` | Placeholder | The key runs again, on a pair that does not share a resonance |

trans-Thioindigo is `Step_1/Molecules/transthioindigo/`. Its bright root is 417 nm (f = 0.29). A 35 nm silver sphere in water peaks at 418.5 nm. That pair is not selected, and it is not written into the JSON. A core survey of that molecule is in `Step_2/thioindigo_trial/`. The μ tune input is `Step_1/LC-wPBE mu/transthioindigo/tune.json` and has not been run. There is no separate step for choosing the pair.

## Placeholders

Fill every `PLACEHOLDER_RESONANT` token in Steps 2–6 in one pass, once the resonant pair is chosen. In Step 7, fill those same tokens, but the molecule’s x coordinate is one of the three gap tokens, not the baseline x. Fill every `PLACEHOLDER_DETUNED` token in `Step_8` only. Replace the whole string. Until then the core-MO key and `watch_indices` are strings; the key becomes the surveyed index as a string, and the three watch indices become integers.

The baseline surface gap is 0.015 μm in Steps 3–6 and in Step 8. The molecule's x coordinate there is the nanoparticle radius plus that gap, both in μm. Step 7 is the only step that changes the gap: 0.005, 0.030, and 0.060 μm. `cell_length` stays 0.2. If the sphere, the gap, and the PML no longer fit, widen the cell before launching.

The Gaussian wavelength is the molecule's bright root, in μm, on both pairs. On the resonant pair the nanoparticle plasmon sits on that root. On the detuned pair it does not. `fwidth` stays 2.0.

| Token suffix | Field |
| --- | --- |
| `_MOLECULE.xyz` | `molecule.geometry` |
| `_CORE_MO` | key in `core_hole.mo_removal_index_dict`. The value stays 2 |
| `_HOMO_M1`, `_HOMO`, `_LUMO` | `watch_indices`. Frontier MOs to plot, not the core |
| `_NP` | `nanoparticle.material` |
| `_NP_RADIUS_UM` | `nanoparticle.radius` |
| `_MOLECULE_X_UM` | `plasmon.molecule.position[0]` at the 0.015 μm gap. Not used in Step 7 |
| `_X_GAP005_UM`, `_X_GAP030_UM`, `_X_GAP060_UM` | Step 7 only. Radius plus 0.005, 0.030, or 0.060 μm |
| `_WAVELENGTH_UM` | Gaussian `wavelength` |
| `_WINDOW_MIN_EV`, `_WINDOW_MAX_EV` | Plot window on the Gaussian jobs. δ-kick jobs stay at 1.5–15 eV |
| `_MU` | `lrc_parameter`. One value on every file for that molecule, CAP on or off |
| `_EPS0` | `cap.eps0`, in Ha. Only the files that have a CAP. Tuned at that μ |

The prefix is `PLACEHOLDER_RESONANT` or `PLACEHOLDER_DETUNED`. The survey's `mo_removal_index_dict` lists MOs 0 through 5 on purpose. That list is the orbitals to print. It is not the production hole, and it is not a placeholder.

If the detuned pair reuses the resonant molecule, copy the molecule tokens, including `_MU` and `_EPS0`, and change the nanoparticle. Rerun a molecule-only job only when the molecule changes. D1, K1, K3, G1, and G2 can be reused from Steps 3–5 in that case. The Gaussian stays on the molecule's root either way.

## Locked for this series

These are the same in every JSON from Step 2 on. They are not the pair.

| Item | Value |
| --- | --- |
| Basis | `6-311G*`, Cartesian Gaussians |
| Charge / spin | 0 / 0. Singlet DCH stays restricted |
| XC | LC-ωPBE (`HYB_GGA_XC_LC_WPBE`). μ is `_MU`, not a fixed number |
| CAP, when present | static, `gam0` 1, `xi` 0.5, `clamp` 100. `eps0` is `_EPS0` |
| Core hole, when present | 2 electrons from one MO. The index is the `_CORE_MO` placeholder |
| Quantum time | `dt` 0.05 au. Field-free dynamics `t_end` 400 au. δ-kick spectra `t_end` 4000 au |
| Hybrid time | JSON `dt` 0.1 au, `t_end` 10000 au. PlasMol snaps `dt` to the Meep step (about 0.10007 au) at Courant 0.5. Every hybrid job uses this same pair |
| Medium | refractive index 1.33 (water). Every Meep cell |
| Gap | 0.015 μm in Steps 3–6 and Step 8. Step 7 uses 0.005, 0.030, and 0.060 μm. The coordinate is a placeholder until the radius is known |

μ is the range-separation parameter of LC-ωPBE. It is a placeholder because it has to be tuned for the molecule, and it has to be the same number on every file for that molecule. A CAP-off run with a different μ is not the partner of the CAP-on run. ε₀ is the CAP threshold, in hartree, and it is tuned at that μ. It appears only in `molecule.cap`. The 3-pentanone tune, μ = 0.34272 and ε₀ = 0.003028 Ha, is not written into these files. Step 1’s thioindigo root at 417 nm was computed at μ = 0.34272. That root moves if the production μ is different. The nuclei stay at the PBE0 geometry. Switching the series to PBE0 means editing every JSON from Step 2 on and retuning ε₀. Do not reuse a PBE0 threshold with LC-ωPBE, or the reverse.

No input is written with `molecule.rotation`. D1 stays on the geometry file you point it at. After D1, the orientation check prints the one rotation that puts the core-hole dipole on +x. Paste that into D3–D6 only. Do not edit the geometry file to aim the dipole.

The production hole is one double vacancy on a single MO, so the job remains closed-shell. Do not launch a DCH job until `_CORE_MO` is the index from that molecule's survey.

### Step 1 — DCH literature, and the valence screen

Done. The note is `Step_1/README.md`. No production JSON. For each molecule it records the DOI, what the paper was trying to measure or calculate, and what it found. The folders under `Step_1/Molecules/` are the valence-absorption screen, and trans-thioindigo is one of them. Its bright root is 417 nm, on the 418.5 nm silver plasmon. None of the double-core-hole literature molecules are. The production molecule is still open. The sudden double hole used from Step 3 on is the pentanone initial condition, not a two-site free-electron-laser spectrum.

### Step 2 — core MO survey, resonant molecule

`survey.json`. Driver `core_hole`. It builds the neutral molecule, writes the atoms that contribute to MOs 0 through 5 into `mo_survey.txt`, and exits. The same table is in `log.out` when the job is launched with `--log log.out`. It does not remove electrons and it does not propagate. `t_end` is unused.

`thioindigo_trial/` is an earlier survey of trans-thioindigo: MO 0 and MO 1 sulfur 1s, MO 2 and MO 3 oxygen 1s, MO 4 and MO 5 carbon 1s. Do not copy those indices into the production files. The two oxygen 1s there are equivalent, so a canonical orbital is a combination of both carbonyls. That warning applies to any molecule with equivalent atoms. Read the new survey before choosing `_CORE_MO`.

### Step 3 — field-free DCH dynamics, resonant pair

No external electric field. The drive is the sudden hole. D1 and D2 are molecule-only (`quantum`) and can checkpoint. D3–D6 are hybrid (`plasmol`), have no Meep source, and cannot checkpoint.

| File | CAP | Nanoparticle | Back-propagation |
| --- | --- | --- | --- |
| `D1/D1.json` | off | no | — |
| `D2/D2.json` | on | no | — |
| `D3/D3.json` | off | yes | on |
| `D4/D4.json` | off | yes | off |
| `D5/D5.json` | on | yes | on |
| `D6/D6.json` | on | yes | off |

Read D2 − D1 as the CAP on the free hole. D3 − D1 as the nanoparticle when the molecule is allowed to radiate. D4 − D1 as the nanoparticle when it is not. If D4 matches D1 and D3 does not, the nanoparticle effect is the radiated field coming back.

D1 and D2 stay on the geometry as given. After D1 finishes, run:

```bash
python -m plasmol.quantum.orientation_check
```

That reads `Step_3/D1/field_p_D1.csv` and writes `orientation_check.txt`. The report includes the `molecule.rotation` block that puts the moving core-hole dipole on +x. Paste it into D3–D6 only. If the motion is already along x, the report says to add nothing. D2 − D1 stays a same-frame CAP comparison.

Do not treat D3–D6 as cross sections. There is no incident field to divide by. The outputs are `mo_occ.csv`, `field_e_*.csv`, and `field_p_*.csv`.

D1 and D2 run at `dt` 0.05 to 400 au. D3–D6 run at the Meep step, `dt` 0.1 to 10000 au. Compare occupations only on the overlapping window. Do not treat the two time grids as one trace.

Older traces in `jobs/DCH/mo_tracking_*` used `"driver": "dch"` and are not copied here. They are not valid production inputs. Do not overlay those CSV files on D1 or D2.

### Step 4 — δ-kick spectra, resonant molecule

Isolated molecule, absorption driver, polarization `full` (x, y, and z). Kick strength 0.001 au. `t_end` 4000 au. Fourier window `gamma` 0.01, plotted from 1.5 to 15 eV. These jobs can checkpoint. A core hole writes `mo_occ.csv` under `x_dir/`, `y_dir/`, and `z_dir/`.

`gamma` = 0.01 au is a Lorentzian of about 0.27 eV half-width. The dipole is already small by a few hundred au, so 4000 au does not make the spectrum sharper than that window. It does keep the time signal if you want it. Lower `gamma` before relying on the longer trace for resolution. If the bright root is above 15 eV, raise `max_ev` on all four files together.

| File | DCH | CAP |
| --- | --- | --- |
| `K1/K1.json` | off | off |
| `K2/K2.json` | off | on |
| `K3/K3.json` | on | off |
| `K4/K4.json` | on | on |

K1 is the ordinary spectrum of the resonant molecule. The plotted curve is the average of the three kicks, so rotating the nuclei would not change it. The separate dipole files are the valence axis. Use them only if a Gaussian job needs that axis on +x. They are not the rotation for D3–D6. K3 − K1 is the sudden hole with no CAP. K2 − K1 and K4 − K3 are the CAP on a given initial state. `gamma` is only the Fourier window. It is not a substitute for the CAP.

### Step 5 — Gaussian hybrid cross sections, resonant pair

Same cell and the same Gaussian in every file. The nanoparticle block is present only in G3, G4, G7, and G8. The source wavelength and the plot window are the resonant placeholders. Polarization `parallel` puts the field along the nanoparticle–molecule axis. Polarization `single` is the empty cell: no nanoparticle, and the source is left as written (x-polarized, propagating in y). Back-propagation is on. No checkpoints.

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

The empty-cell Gaussian is not a substitute for Step 4. K1–K4 stay the real-time reference. Step 1 is the linear-response check of a neutral molecule in this screen, once that molecule is the one in these files.

### Step 6 — two controls on resonant G4

Not part of the grid. Run them only after G4 itself shows a nanoparticle effect.

| File | What changes from G4 |
| --- | --- |
| `G4_perpendicular/G4_perpendicular.json` | Polarization `perpendicular` instead of `parallel` |
| `G4_no_backprop/G4_no_backprop.json` | `back_propagation` false. The molecule feels the cell and does not source it |

One perpendicular run and one back-propagation-off run are enough. Do not repeat them across CAP and DCH until G4 has moved.

### Step 7 — distance, resonant pair

Not part of the grid. Run it only after G4 − G2 is nonzero at 0.015 μm. Notes: `Step_7/README.md`.

Three surface gaps, each with D3, D4, and G4. D1 and G2 are not repeated: they have no nanoparticle. No CAP, and no perpendicular run. The detuned pair stays at 0.015 μm.

| Gap | D3 − D1 and G4 − G2, compared with 0.015 μm |
| --- | --- |
| 0.005 μm | Closer. A near field should get stronger |
| 0.030 μm | Twice the baseline gap |
| 0.060 μm | Far enough that a near field should be gone |

If the gap changes D3 and G4 but D4 stays on D1, the distance dependence is the radiated field coming back. Paste the Step 3 orientation into these D3 and D4 files. Widen `cell_length` on a gap whose sphere no longer fits.

### Step 8 — the same question, off resonance

The resonant grid asks what the nanoparticle and the CAP do when the plasmon and the bright root sit on each other. This step asks whether that nanoparticle effect is still there when they do not.

It repeats only the runs that isolate the nanoparticle on the sudden hole. It does not repeat the CAP rows, the Step 6 controls, or the Step 7 gaps. The gap stays 0.015 μm, so this comparison is resonance and not distance. Add the controls here only if the detuned G4 itself moves and the mechanism is no longer obvious from D3 − D4.

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

The orientation check is the same command as Step 3, pointed at `Step_8/D1/field_p_D1.csv`. Paste the rotation into the Step 8 D3 and D4 only.
