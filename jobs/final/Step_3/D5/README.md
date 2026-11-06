# D5

Field-free sudden double core hole next to the 25 nm gold sphere, with back-propagation on and the CAP on.

D3 is this hybrid with the CAP off. D6 is this hybrid with back-propagation off. D2 is the CAP on the free hole, with no sphere and a 400 au clock. D5 is the only Step 3 row with the sphere, radiation on, and the CAP on.

## What this run is

The sodium atom in the resonant series is charge 0, spin 1, `aug-cc-pVTZ` Cartesian, LC-ωPBE with μ = 0.351340. The geometry is `Na.xyz`, the nucleus at the origin. A sudden double hole, where it is on, removes two electrons from MO 0 (Na 1s) and keeps the neutral orbitals. `watch_indices` is omitted, so an occupation file is every logged MO, indices 0 through 6. The β LUMO is index 5, which is why the logged set runs through 6.

The CAP, on the runs that have one, is static: `gam0` 1, `xi` 0.5, `eps0` 0.009767 Ha, `clamp` 100. It does not replace μ. The sphere, on the runs that have one, is Meep `Au` (Rakić), radius 0.025 μm, center at the origin, in water (refractive index 1.33). The baseline atom position is 0.02645 μm, one Yee pixel off the surface. At `dt` 0.1 and Courant 0.5 the resolution is 689 and one pixel is 0.001451379 μm. The stored coordinate is 1.4 pm short of an exact pixel and inside the position check. The field sample interpolates. It does not snap the atom onto a pixel center.

Driver `plasmol`, same clock, sphere, rotation, and back-propagation as D3, plus the static CAP. No Meep source and no checkpoint. Outputs are `mo_occ.csv`, `field_e_D5.csv`, `field_p_D5.csv`, and `fields_D5.png`. Not a cross section.

The run finished at about 10007 au with no error. Back-propagation is on, and the local field stays below 4e-8. The dipole peaks at 0.002. The three 2p holes stay equal, and the 3s hole ends at 0.98. The plotted holes sum to 2.08 at the start and 4.11 at the end. The CAP is on, so that increase can be absorption as well as promotion out of MOs 0 through 6. D1_10k and the no-CAP hybrids grow a large dipole. This run does not.

D5 − D3 is the CAP on the radiating sphere. D5 − D6 is back-propagation with the CAP on. D5 − D2 mixes the sphere with a change of clock, so it is not the clean CAP comparison. That one is D2 − D1.

## The rest of Step 3

| Run | CAP | Nanoparticle | Back-propagation | Driver | Time |
| --- | --- | --- | --- | --- | --- |
| D1 | off | no | n/a | quantum | 400 au, dt 0.05 |
| D1_10k | off | no | n/a | quantum | 10000 au, dt 0.05 |
| D2 | on | no | n/a | quantum | 400 au, dt 0.05 |
| D3 | off | 25 nm Au | on | plasmol | 10000 au, dt 0.1 |
| D4 | off | 25 nm Au | off | plasmol | 10000 au, dt 0.1 |
| **D5** | on | 25 nm Au | on | plasmol | 10000 au, dt 0.1 |
| D6 | on | 25 nm Au | off | plasmol | 10000 au, dt 0.1 |

D1 and D2 can checkpoint. D3 through D6 cannot. D3 through D6 carry the orientation block. D1, D2, and D1_10k do not. None of these runs is a cross section: there is no incident field to divide by. Neutral field-free dynamics are not in this matrix. With no hole and no field the density does not move.

Older traces in `jobs/DCH/mo_tracking_*` used a retired driver name and are not these inputs. Do not overlay them on D1 or D2.
