# D3

Field-free sudden double core hole next to the 25 nm gold sphere, with back-propagation on and the CAP off.

The atom is allowed to radiate into the cell. There is no Meep source. D4 is the same hybrid with back-propagation off. D1 is the same hole with no sphere and a shorter clock. D5 is this run plus the CAP. D3 is the only Step 3 row with the sphere, radiation on, and the CAP off.

## What this run is

The sodium atom in the resonant series is charge 0, spin 1, `aug-cc-pVTZ` Cartesian, LC-ωPBE with μ = 0.351340. The geometry is `Na.xyz`, the nucleus at the origin. A sudden double hole, where it is on, removes two electrons from MO 0 (Na 1s) and keeps the neutral orbitals. `watch_indices` is omitted, so an occupation file is every logged MO, indices 0 through 6. The β LUMO is index 5, which is why the logged set runs through 6.

The CAP, on the runs that have one, is static: `gam0` 1, `xi` 0.5, `eps0` 0.009767 Ha, `clamp` 100. It does not replace μ. The sphere, on the runs that have one, is Meep `Au` (Rakić), radius 0.025 μm, center at the origin, in water (refractive index 1.33). The baseline atom position is 0.02645 μm, one Yee pixel off the surface. At `dt` 0.1 and Courant 0.5 the resolution is 689 and one pixel is 0.001451379 μm. The stored coordinate is 1.4 pm short of an exact pixel and inside the position check. The field sample interpolates. It does not snap the atom onto a pixel center.

Driver `plasmol`. `dt` 0.1 au, `t_end` 10000 au. PlasMol snaps `dt` to the Meep step, about 0.10007 au, and the end time to about 10007 au. No `plasmon.source`. `back_propagation` is true, so the only classical source is the molecular dipole. `molecule.rotation` is the D1 orientation check. The nucleus does not move. Hybrid jobs cannot checkpoint. Outputs are `mo_occ.csv`, `field_e_D3.csv`, `field_p_D3.csv`, and `fields_D3.png`. Do not divide by an incident field. There is not one.

The finished run reached about 10007 au with no error. The local field peaks at 1.1e-5. Through 400 au the dipole stays under 3e-10, as on D1. After that it follows D4: it passes 0.1 near 2060 au and peaks at 0.81 near 7230 au, and the 2p holes split. The plotted holes sum to 2.07 at the start and 3.26 at the end. There is no CAP, so those extra holes are electrons that left MOs 0 through 6.

`partial_t3486/` is an earlier launch, not a second simulation. That log dies when a Magnus step fails to converge within 200 iterations (last residual about 8.3e-11). The progress lines stop just after 3002 au. The finished traces are the files in this directory.

D3 − D1 is the sphere when the atom radiates, on the first 400 au only. D3 − D4 is back-propagation with the CAP off. D5 − D3 is the CAP on this radiating case. D1_10k is the free-atom trace over the same long window.

## The rest of Step 3

| Run | CAP | Nanoparticle | Back-propagation | Driver | Time |
| --- | --- | --- | --- | --- | --- |
| D1 | off | no | n/a | quantum | 400 au, dt 0.05 |
| D1_10k | off | no | n/a | quantum | 10000 au, dt 0.05 |
| D2 | on | no | n/a | quantum | 400 au, dt 0.05 |
| **D3** | off | 25 nm Au | on | plasmol | 10000 au, dt 0.1 |
| D4 | off | 25 nm Au | off | plasmol | 10000 au, dt 0.1 |
| D5 | on | 25 nm Au | on | plasmol | 10000 au, dt 0.1 |
| D6 | on | 25 nm Au | off | plasmol | 10000 au, dt 0.1 |

D1 and D2 can checkpoint. D3 through D6 cannot. D3 through D6 carry the orientation block. D1, D2, and D1_10k do not. None of these runs is a cross section: there is no incident field to divide by. Neutral field-free dynamics are not in this matrix. With no hole and no field the density does not move.

Older traces in `jobs/DCH/mo_tracking_*` used a retired driver name and are not these inputs. Do not overlay them on D1 or D2.
