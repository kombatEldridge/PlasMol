# D2

Field-free sudden double core hole with the complex absorbing potential on. No nanoparticle and no external field.

D1 is the same propagation with the CAP off. That pair is the CAP on the free hole. D5 and D6 also have the CAP, but they add the sphere and they run on the hybrid clock. D2 has no sphere, and it stops at 400 au, so it is not those runs.

## What this run is

The sodium atom in the resonant series is charge 0, spin 1, `aug-cc-pVTZ` Cartesian, LC-ωPBE with μ = 0.351340. The geometry is `Na.xyz`, the nucleus at the origin. A sudden double hole, where it is on, removes two electrons from MO 0 (Na 1s) and keeps the neutral orbitals. `watch_indices` is omitted, so an occupation file is every logged MO, indices 0 through 6. The β LUMO is index 5, which is why the logged set runs through 6.

The CAP, on the runs that have one, is static: `gam0` 1, `xi` 0.5, `eps0` 0.009767 Ha, `clamp` 100. It does not replace μ. The sphere, on the runs that have one, is Meep `Au` (Rakić), radius 0.025 μm, center at the origin, in water (refractive index 1.33). The baseline atom position is 0.02645 μm, one Yee pixel off the surface. At `dt` 0.1 and Courant 0.5 the resolution is 689 and one pixel is 0.001451379 μm. The stored coordinate is 1.4 pm short of an exact pixel and inside the position check. The field sample interpolates. It does not snap the atom onto a pixel center.

Driver `quantum`. `dt` 0.05 au, `t_end` 400 au. The molecule block matches D1, plus the static CAP. No sphere, no rotation, checkpoint allowed (`checkpoint_D2.npz`). Outputs are `mo_occ.csv`, `field_e_D2.csv`, `field_p_D2.csv`, and `fields_D2.png`.

The run finished at 400 au with no error. Read it against D1, not against D5 or D6. Those two add the sphere and a different time step. The CAP is the Lopata non-Hermitian term. `gamma` in a Fourier window is not this object, and this folder has no spectrum.

## The rest of Step 3

| Run | CAP | Nanoparticle | Back-propagation | Driver | Time |
| --- | --- | --- | --- | --- | --- |
| D1 | off | no | n/a | quantum | 400 au, dt 0.05 |
| D1_10k | off | no | n/a | quantum | 10000 au, dt 0.05 |
| **D2** | on | no | n/a | quantum | 400 au, dt 0.05 |
| D3 | off | 25 nm Au | on | plasmol | 10000 au, dt 0.1 |
| D4 | off | 25 nm Au | off | plasmol | 10000 au, dt 0.1 |
| D5 | on | 25 nm Au | on | plasmol | 10000 au, dt 0.1 |
| D6 | on | 25 nm Au | off | plasmol | 10000 au, dt 0.1 |

D1 and D2 can checkpoint. D3 through D6 cannot. D3 through D6 carry the orientation block. D1, D2, and D1_10k do not. None of these runs is a cross section: there is no incident field to divide by. Neutral field-free dynamics are not in this matrix. With no hole and no field the density does not move.

Older traces in `jobs/DCH/mo_tracking_*` used a retired driver name and are not these inputs. Do not overlay them on D1 or D2.
