# D4

Field-free sudden double core hole next to the 25 nm gold sphere, with back-propagation off and the CAP off.

The atom feels the cell and does not source it. D3 is the same hybrid with back-propagation on. D6 is this run plus the CAP. D1 is the same hole with no sphere. If D4 matches D1 and D3 does not, the nanoparticle effect is the radiated field coming back. D4 is the run that can show that, because it is the only no-CAP hybrid in which the atom does not source the cell.

## What this run is

The sodium atom in the resonant series is charge 0, spin 1, `aug-cc-pVTZ` Cartesian, LC-ωPBE with μ = 0.351340. The geometry is `Na.xyz`, the nucleus at the origin. A sudden double hole, where it is on, removes two electrons from MO 0 (Na 1s) and keeps the neutral orbitals. `watch_indices` is omitted, so an occupation file is every logged MO, indices 0 through 6. The β LUMO is index 5, which is why the logged set runs through 6.

The CAP, on the runs that have one, is static: `gam0` 1, `xi` 0.5, `eps0` 0.009767 Ha, `clamp` 100. It does not replace μ. The sphere, on the runs that have one, is Meep `Au` (Rakić), radius 0.025 μm, center at the origin, in water (refractive index 1.33). The baseline atom position is 0.02645 μm, one Yee pixel off the surface. At `dt` 0.1 and Courant 0.5 the resolution is 689 and one pixel is 0.001451379 μm. The stored coordinate is 1.4 pm short of an exact pixel and inside the position check. The field sample interpolates. It does not snap the atom onto a pixel center.

Driver `plasmol`, same clock and same sphere as D3. `back_propagation` is false. No Meep source, so if the atom does not source the cell the local field has nothing to drive it. The D1 rotation is on this file and does not move the origin nucleus. No checkpoint. Outputs are `mo_occ.csv`, `field_e_D4.csv`, `field_p_D4.csv`, and `fields_D4.png`. Not a cross section.

The run finished at about 10007 au with no error. The local field stayed at zero, so the sphere did not drive the atom. The dipole does not match D1. It stays under 1e-6 until 1089 au, then reaches 0.91 near 2850 au, and the three 2p holes, equal on D1, split here. D1_10k says the free atom also grows this dipole, later: D1 itself had already ended at 400 au, which is why the short reference looked quiet.

D4 − D1 is the sphere when the atom does not radiate. D3 − D4 is back-propagation with the CAP off. D6 − D4 is the CAP on this non-radiating case.

## The rest of Step 3

| Run | CAP | Nanoparticle | Back-propagation | Driver | Time |
| --- | --- | --- | --- | --- | --- |
| D1 | off | no | n/a | quantum | 400 au, dt 0.05 |
| D1_10k | off | no | n/a | quantum | 10000 au, dt 0.05 |
| D2 | on | no | n/a | quantum | 400 au, dt 0.05 |
| D3 | off | 25 nm Au | on | plasmol | 10000 au, dt 0.1 |
| **D4** | off | 25 nm Au | off | plasmol | 10000 au, dt 0.1 |
| D5 | on | 25 nm Au | on | plasmol | 10000 au, dt 0.1 |
| D6 | on | 25 nm Au | off | plasmol | 10000 au, dt 0.1 |

D1 and D2 can checkpoint. D3 through D6 cannot. D3 through D6 carry the orientation block. D1, D2, and D1_10k do not. None of these runs is a cross section: there is no incident field to divide by. Neutral field-free dynamics are not in this matrix. With no hole and no field the density does not move.

Older traces in `jobs/DCH/mo_tracking_*` used a retired driver name and are not these inputs. Do not overlay them on D1 or D2.
