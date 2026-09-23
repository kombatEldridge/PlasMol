# Final jobs

This directory is the run list for the campaign in `plan.md`. Read the steps in order. Each `Step_n` is one stage. A stage that has several runs keeps those runs in folders named by the plan (`D1`, `K3`, `G4`, …).

The molecule is trans-thioindigo and the nanoparticle is a 35 nm silver sphere. Step_1 is the record of that choice. Run a JSON from the directory that contains it, and pass `--log log.out` so the run record is saved beside the input. Geometry paths are relative to that file. Do not launch two runs in the same directory.

```bash
python -m plasmol.main survey.json --log log.out
```

Spectra, occupations, and fields are the files named in each JSON. The core-orbital survey also writes `mo_survey.txt`. The Step 1 scripts already write their tables and figures. Nothing from these steps is left only in the terminal.

The linear-response and Mie scripts in Step_1 have been run. Step 2 writes `mo_survey.txt`. The JSON files from Step_3 on have not been run. There is no neutral-orientation screen. K1's three dipole files are the valence axis, if a Gaussian job needs one. D1's dipole is the axis for D3–D6.

## Locked for this series

These are the same in every JSON from Step_2 on.

| Item | Value |
| --- | --- |
| Geometry | `Step_1/transthioindigo.xyz`. Do not edit it. Later JSON files point at it |
| Basis | `6-311G*`, Cartesian Gaussians |
| Charge / spin | 0 / 0. Singlet DCH stays restricted |
| XC | LC-ωPBE (`HYB_GGA_XC_LC_WPBE`), μ = 0.34272 on every run, CAP on or off |
| CAP, when present | static, `gam0` 1, `xi` 0.5, `eps0` 0.003028 Ha, `clamp` 100 |
| Core hole, when present | 2 electrons from one MO. **Placeholder MO 0.** Replace after Step_2 |
| Plot MOs | 74, 75, 76. HOMO−1, HOMO, and LUMO of the 76-electron closed shell. Plot only |
| Quantum time | `dt` 0.05 au. Field-free dynamics `t_end` 400 au. δ-kick spectra `t_end` 4000 au |
| Hybrid time | JSON `dt` 0.1 au, `t_end` 10000 au. PlasMol snaps `dt` to the Meep step (about 0.10007 au) at Courant 0.5. Every hybrid job uses this same pair |
| Medium | refractive index 1.33 (water). Every Meep cell |
| Nanoparticle | `Ag`, radius 0.0175 μm (diameter 35 nm), center at the origin. Molecule at `[0.0325, 0, 0]` (gap 0.015 μm) |
| Gaussian | wavelength 0.417 μm, `fwidth` 2.0. Centered on the Step_1 root |

μ = 0.34272 is the range-separation parameter of the functional. ε₀ = 0.003028 Ha is locked with it and has not been retuned on thioindigo. The nuclei were optimized at PBE0 and are not reoptimized. Switching the series to PBE0 means editing every JSON from Step_2 on and retuning `eps0`. Do not reuse 0.003028 Ha with PBE0.

No input is written with `molecule.rotation`. D1 stays on the stock geometry. After D1, the orientation check prints the one rotation that puts the core-hole dipole on +x. Paste that into D3–D6 only. Do not edit `Step_1/transthioindigo.xyz`.

The ionized MO is not known until Step_2 says so. Thioindigo has two sulfur 1s orbitals below the two oxygen 1s, so MO 0 is not an oxygen core. Do not launch a DCH job (any file with `core_hole` from Step_3 on) until that index is written in. One double hole stays on a single MO so the job remains closed-shell.

## Steps

| Step | Status | What it answers |
| --- | --- | --- |
| `Step_1` | LR and Mie already run | Which molecule and which sphere share one resonance |
| `Step_2` | Input only | Which MO is the core on the thioindigo frame |
| `Step_3` | Inputs only | Field-free DCH dynamics, one factor at a time |
| Check | After D1 | Which rotation puts the core-hole dipole on +x for D3–D6 |
| `Step_4` | Inputs only | δ-kick absorption of thioindigo, ± DCH, ± CAP |
| `Step_5` | Inputs only | Gaussian hybrid cross section, ± DCH, ± NP, ± CAP |
| `Step_6` | Inputs only | Two controls on the G4 point, not part of the main grid |

### Step_1 — molecule and nanoparticle

The notes are `Step_1/README.md`. Ten small dyes were considered for a bright band near 540 nm. Only trans-thioindigo was calculated. Gas-phase LC-ωPBE puts its bright root at 417 nm (f = 0.29). Meep material `Ag` in water peaks at 416.5 nm for a diameter of 32.6 nm. The series uses 35 nm, whose peak is 418.5 nm. `lr_tddft.py` and `ag_mie.py` have been run. `d35/d35.json` is the bare-sphere flux job for that diameter and has not been run.

### Step_2 — core MO survey

`survey.json`. Driver `core_hole`. It builds the neutral molecule, writes the atoms that contribute to MOs 0 through 5 into `mo_survey.txt`, and exits. The same table is in `log.out` when the job is launched with `--log log.out`. It does not remove electrons and it does not propagate. `t_end` is unused.

Those six orbitals are the window where the two sulfur 1s and the two oxygen 1s should sit. Leave this job on the stock geometry, the same frame as D1. Then read the log and replace `"0"` in every later `mo_removal_index_dict` with the one MO that is the oxygen core you intend to ionize. The two oxygen 1s are equivalent, so a canonical orbital is a combination of them, not a hole on one carbonyl. If the oxygen character is outside `{0, 1, 2, 3, 4, 5}`, add that index and run the survey again before changing the production files.

### Step_3 — field-free DCH dynamics

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

D1 and D2 stay on the stock geometry. After D1 finishes, run:

```bash
python -m plasmol.quantum.orientation_check
```

That reads `Step_3/D1/field_p_D1.csv` and writes `orientation_check.txt`. The report includes the `molecule.rotation` block that puts the moving core-hole dipole on +x. Paste it into D3–D6 only. If the motion is already along x, the report says to add nothing. D2 − D1 stays a same-frame CAP comparison.

Do not treat D3–D6 as cross sections. There is no incident field to divide by. The outputs are `mo_occ.csv`, `field_e_*.csv`, and `field_p_*.csv`.

D1 and D2 run at `dt` 0.05 to 400 au. D3–D6 run at the Meep step, `dt` 0.1 to 10000 au. Compare occupations only on the overlapping window. Do not treat the two time grids as one trace.

Older traces in `jobs/DCH/mo_tracking_*` used `"driver": "dch"` and are not copied here. They are not valid production inputs. Do not overlay those CSV files on D1 or D2.

### Step_4 — δ-kick spectra

Isolated molecule, absorption driver, polarization `full` (x, y, and z). Kick strength 0.001 au. `t_end` 4000 au. Fourier window `gamma` 0.01, plotted from 1.5 to 15 eV. These jobs can checkpoint. A core hole writes `mo_occ.csv` under `x_dir/`, `y_dir/`, and `z_dir/`.

`gamma` = 0.01 au is a Lorentzian of about 0.27 eV half-width. The dipole is already small by a few hundred au, so 4000 au does not make the spectrum sharper than that window. It does keep the time signal if you want it. Lower `gamma` before relying on the longer trace for resolution.

| File | DCH | CAP |
| --- | --- | --- |
| `K1/K1.json` | off | off |
| `K2/K2.json` | off | on |
| `K3/K3.json` | on | off |
| `K4/K4.json` | on | on |

K1 is the ordinary thioindigo spectrum on the stock frame. The plotted curve is the average of the three kicks, so rotating the nuclei would not change it. The separate dipole files `x_dir/field_p_K1.csv`, `y_dir/field_p_K1.csv`, and `z_dir/field_p_K1.csv` are the valence axis. Use them only if a Gaussian job needs that axis on +x. They are not the rotation for D3–D6. K3 − K1 is the sudden hole with no CAP. K2 − K1 and K4 − K3 are the CAP on a given initial state. `gamma` is only the Fourier window. It is not a substitute for the CAP.

### Step_5 — Gaussian hybrid cross sections

Same cell and the same Gaussian in every file. The nanoparticle is the Step_1 sphere: `Ag`, radius 0.0175 μm. The source is centered at 0.417 μm with `fwidth` 2.0. Polarization `parallel` puts the field along the nanoparticle–molecule axis. Polarization `single` is the empty cell: no nanoparticle, and the source is left as written (x-polarized, propagating in y). Back-propagation is on. No checkpoints.

The plotted window is 1.5 to 5 eV. `gamma` is left at 0. The published curve is `cross_section`.

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

The empty-cell Gaussian is not a substitute for Step_4. K1–K4 stay the real-time reference. Step_1 is the linear-response check of the same neutral molecule.

### Step_6 — two controls on G4

Not part of the grid. Run them only after G4 itself shows a nanoparticle effect.

| File | What changes from G4 |
| --- | --- |
| `G4_perpendicular/G4_perpendicular.json` | Polarization `perpendicular` instead of `parallel` |
| `G4_no_backprop/G4_no_backprop.json` | `back_propagation` false. The molecule feels the cell and does not source it |

One perpendicular run and one back-propagation-off run are enough. Do not repeat them across CAP and DCH until G4 has moved.
