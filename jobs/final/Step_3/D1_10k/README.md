# D1_10k

The D1 propagation extended from 400 au to 10000 au. Still no nanoparticle, no CAP, and no external field.

D1 is the same input with `t_end` 400. This folder does not replace it. It exists so the free atom can be compared with D3 through D6 on the hybrid window, without changing the Hamiltonian or adding a sphere. Nothing else in Step 3 is a molecule-only run past 400 au. D2 stays at 400 au, and D2 also has the CAP.

## What this run is

The sodium atom in the resonant series is charge 0, spin 1, `aug-cc-pVTZ` Cartesian, LC-ωPBE with μ = 0.351340. The geometry is `Na.xyz`, the nucleus at the origin. A sudden double hole, where it is on, removes two electrons from MO 0 (Na 1s) and keeps the neutral orbitals. `watch_indices` is omitted, so an occupation file is every logged MO, indices 0 through 6. The β LUMO is index 5, which is why the logged set runs through 6.

The CAP, on the runs that have one, is static: `gam0` 1, `xi` 0.5, `eps0` 0.009767 Ha, `clamp` 100. It does not replace μ. The sphere, on the runs that have one, is Meep `Au` (Rakić), radius 0.025 μm, center at the origin, in water (refractive index 1.33). The baseline atom position is 0.02645 μm, one Yee pixel off the surface. At `dt` 0.1 and Courant 0.5 the resolution is 689 and one pixel is 0.001451379 μm. The stored coordinate is 1.4 pm short of an exact pixel and inside the position check. The field sample interpolates. It does not snap the atom onto a pixel center.

Same driver, same molecule block, and the same checkpointable `quantum` setup as D1. The JSON differences are `t_end` 10000 and the `_10k` suffixes on the checkpoint and field files (`checkpoint_D1_10k.npz`, `field_e_D1_10k.csv`, `field_p_D1_10k.csv`, `fields_D1_10k.png`). There is still no rotation and no sphere.

The run finished at 10000 au. Through 400 au the holes match D1 to 3e-12 and the dipole stays under 1e-10. After that the free atom does what D3 and D4 do: the dipole passes 0.1 near 3140 au, peaks at 0.86 near 7970 au, and the 2p holes split. There is no sphere in this run. D3 and D4 reach a dipole of 0.1 about 1000 au earlier. The CAP runs, D5 and D6, do not grow this dipole.

Do not overlay this 0.05 au trace on a hybrid 0.1 au trace as if they were one grid. Compare the overlapping times.

## The rest of Step 3

| Run | CAP | Nanoparticle | Back-propagation | Driver | Time |
| --- | --- | --- | --- | --- | --- |
| D1 | off | no | n/a | quantum | 400 au, dt 0.05 |
| **D1_10k** | off | no | n/a | quantum | 10000 au, dt 0.05 |
| D2 | on | no | n/a | quantum | 400 au, dt 0.05 |
| D3 | off | 25 nm Au | on | plasmol | 10000 au, dt 0.1 |
| D4 | off | 25 nm Au | off | plasmol | 10000 au, dt 0.1 |
| D5 | on | 25 nm Au | on | plasmol | 10000 au, dt 0.1 |
| D6 | on | 25 nm Au | off | plasmol | 10000 au, dt 0.1 |

D1 and D2 can checkpoint. D3 through D6 cannot. D3 through D6 carry the orientation block. D1, D2, and D1_10k do not. None of these runs is a cross section: there is no incident field to divide by. Neutral field-free dynamics are not in this matrix. With no hole and no field the density does not move.

Older traces in `jobs/DCH/mo_tracking_*` used a retired driver name and are not these inputs. Do not overlay them on D1 or D2.
