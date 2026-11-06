# D6

Field-free sudden double core hole next to the 25 nm gold sphere, with back-propagation off and the CAP on.

D5 is this hybrid with back-propagation on. D4 is this hybrid with the CAP off. D2 is the CAP without the sphere. D6 is the only Step 3 row with the sphere, radiation off, and the CAP on. D5 − D6 is back-propagation at fixed CAP. D6 − D4 is the CAP at fixed back-propagation.

## What this run is

The sodium atom in the resonant series is charge 0, spin 1, `aug-cc-pVTZ` Cartesian, LC-ωPBE with μ = 0.351340. The geometry is `Na.xyz`, the nucleus at the origin. A sudden double hole, where it is on, removes two electrons from MO 0 (Na 1s) and keeps the neutral orbitals. `watch_indices` is omitted, so an occupation file is every logged MO, indices 0 through 6. The β LUMO is index 5, which is why the logged set runs through 6.

The CAP, on the runs that have one, is static: `gam0` 1, `xi` 0.5, `eps0` 0.009767 Ha, `clamp` 100. It does not replace μ. The sphere, on the runs that have one, is Meep `Au` (Rakić), radius 0.025 μm, center at the origin, in water (refractive index 1.33). The baseline atom position is 0.02645 μm, one Yee pixel off the surface. At `dt` 0.1 and Courant 0.5 the resolution is 689 and one pixel is 0.001451379 μm. The stored coordinate is 1.4 pm short of an exact pixel and inside the position check. The field sample interpolates. It does not snap the atom onto a pixel center.

Driver `plasmol`, same clock, sphere, rotation, and CAP as D5. `back_propagation` is false, and there is no Meep source. No checkpoint. Outputs are `mo_occ.csv`, `field_e_D6.csv`, `field_p_D6.csv`, and `fields_D6.png`. Not a cross section.

The finished run reached about 10007 au with no error. The half-step retry never fired on that launch. The local field is zero. The dipole peaks at 0.002. The logged holes match D5 at the 0.001-electron level: the 2p holes stay equal, the 3s hole ends at 0.98, and the plotted holes sum to 4.11. Turning the radiated field off does not change this CAP run.

Two earlier attempts died at 2730 au, once at a Magnus residual of 1.3e-9 and again at 9.4e-7. Those traces are `partial_t2730/` and `partial_t2730_rerun/`. They are not the finished run. A half-step retry that also fails is still an error. The JSON value of `pc_convergence` is still 1e-12. A stalled residual below 1e-8 is what the later launches accept.

D6 − D4 is the CAP when the atom does not source the cell. D5 − D6 is back-propagation with the CAP on.

## The rest of Step 3

| Run | CAP | Nanoparticle | Back-propagation | Driver | Time |
| --- | --- | --- | --- | --- | --- |
| D1 | off | no | n/a | quantum | 400 au, dt 0.05 |
| D1_10k | off | no | n/a | quantum | 10000 au, dt 0.05 |
| D2 | on | no | n/a | quantum | 400 au, dt 0.05 |
| D3 | off | 25 nm Au | on | plasmol | 10000 au, dt 0.1 |
| D4 | off | 25 nm Au | off | plasmol | 10000 au, dt 0.1 |
| D5 | on | 25 nm Au | on | plasmol | 10000 au, dt 0.1 |
| **D6** | on | 25 nm Au | off | plasmol | 10000 au, dt 0.1 |

D1 and D2 can checkpoint. D3 through D6 cannot. D3 through D6 carry the orientation block. D1, D2, and D1_10k do not. None of these runs is a cross section: there is no incident field to divide by. Neutral field-free dynamics are not in this matrix. With no hole and no field the density does not move.

Older traces in `jobs/DCH/mo_tracking_*` used a retired driver name and are not these inputs. Do not overlay them on D1 or D2.
