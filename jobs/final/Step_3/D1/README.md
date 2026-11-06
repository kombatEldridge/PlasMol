# D1

Field-free sudden double core hole on the sodium atom. No nanoparticle, no complex absorbing potential, and no external field.

This is the reference row of Step 3. D2 adds the CAP and nothing else. D1_10k is this same Hamiltonian run out to the hybrid end time. D3 through D6 add the gold sphere, and they split on the CAP and on back-propagation. D1 has none of those.

## What this run is

The sodium atom in the resonant series is charge 0, spin 1, `aug-cc-pVTZ` Cartesian, LC-ωPBE with μ = 0.351340. The geometry is `Na.xyz`, the nucleus at the origin. A sudden double hole, where it is on, removes two electrons from MO 0 (Na 1s) and keeps the neutral orbitals. `watch_indices` is omitted, so an occupation file is every logged MO, indices 0 through 6. The β LUMO is index 5, which is why the logged set runs through 6.

The CAP, on the runs that have one, is static: `gam0` 1, `xi` 0.5, `eps0` 0.009767 Ha, `clamp` 100. It does not replace μ. The sphere, on the runs that have one, is Meep `Au` (Rakić), radius 0.025 μm, center at the origin, in water (refractive index 1.33). The baseline atom position is 0.02645 μm, one Yee pixel off the surface. At `dt` 0.1 and Courant 0.5 the resolution is 689 and one pixel is 0.001451379 μm. The stored coordinate is 1.4 pm short of an exact pixel and inside the position check. The field sample interpolates. It does not snap the atom onto a pixel center.

Driver `quantum`. `dt` 0.05 au, `t_end` 400 au. No `plasmon` section and no `molecule.rotation`. The job can checkpoint (`checkpoint_D1.npz`). Outputs are `mo_occ.csv`, `field_e_D1.csv`, `field_p_D1.csv`, and `fields_D1.png`.

The run finished at 400 au. The 1s hole stays near 2, the three 2p holes stay equal to each other, and the dipole stays below 2e-10 au, so there is no radiating axis. The orientation check still reported a noise-direction rotation, 84.1114 degrees about `[0.0, -0.89999759, 0.43589487]`. That object is `molecule.rotation` on D3 through D6. It is not on this file, and it is not on D2, so D2 − D1 stays in one frame. The nucleus is at the origin, so the rotation would not move `Na.xyz`.

Compare D2 − D1 on this 400 au window. Compare D3 − D1 and D4 − D1 only on the overlap. D3 and D4 use the Meep step, about 0.10007 au, not 0.05 au.

## The rest of Step 3

| Run | CAP | Nanoparticle | Back-propagation | Driver | Time |
| --- | --- | --- | --- | --- | --- |
| **D1** | off | no | n/a | quantum | 400 au, dt 0.05 |
| D1_10k | off | no | n/a | quantum | 10000 au, dt 0.05 |
| D2 | on | no | n/a | quantum | 400 au, dt 0.05 |
| D3 | off | 25 nm Au | on | plasmol | 10000 au, dt 0.1 |
| D4 | off | 25 nm Au | off | plasmol | 10000 au, dt 0.1 |
| D5 | on | 25 nm Au | on | plasmol | 10000 au, dt 0.1 |
| D6 | on | 25 nm Au | off | plasmol | 10000 au, dt 0.1 |

D1 and D2 can checkpoint. D3 through D6 cannot. D3 through D6 carry the orientation block. D1, D2, and D1_10k do not. None of these runs is a cross section: there is no incident field to divide by. Neutral field-free dynamics are not in this matrix. With no hole and no field the density does not move.

Older traces in `jobs/DCH/mo_tracking_*` used a retired driver name and are not these inputs. Do not overlay them on D1 or D2.
