# G4

The production hybrid in this step: sudden double hole, gold sphere, back-propagation on, no CAP.

G2 is the hole without the sphere. G4 − G2 is the nanoparticle at fixed hole. G3 is the sphere without the hole. G4 − G3 is the hole at fixed cell. G8 − G4 is the CAP on this hybrid. Step 6 copies this file twice, once with polarization `perpendicular` and once with back-propagation off. Step 7 copies it again at three other gaps. Those copies are not rows of this table.

Run G1 through G4 first if the full set is too many at once.

## What this run is

The sodium atom in the resonant series is charge 0, spin 1, `aug-cc-pVTZ` Cartesian, LC-ωPBE with μ = 0.351340. The geometry is `Na.xyz`, the nucleus at the origin. A sudden double hole, where it is on, removes two electrons from MO 0 (Na 1s) and keeps the neutral orbitals. `watch_indices` is omitted, so an occupation file is every logged MO, indices 0 through 6. The β LUMO is index 5, which is why the logged set runs through 6.

The CAP, on the runs that have one, is static: `gam0` 1, `xi` 0.5, `eps0` 0.009767 Ha, `clamp` 100. It does not replace μ. The sphere, on the runs that have one, is Meep `Au` (Rakić), radius 0.025 μm, center at the origin, in water (refractive index 1.33). The baseline atom position is 0.02645 μm, one Yee pixel off the surface. At `dt` 0.1 and Courant 0.5 the resolution is 689 and one pixel is 0.001451379 μm. The stored coordinate is 1.4 pm short of an exact pixel and inside the position check. The field sample interpolates. It does not snap the atom onto a pixel center.

Same cell and the same Gaussian in every file. `dt` 0.1 au, `t_end` 10000 au. Wavelength 0.618422 μm, `fwidth` 2.0, amplitude 1, source centered at `[0, -0.04, 0]`. The plot window is 1.5 to 4.5 eV. `gamma` is omitted. The published curve is `cross_section` (`spectrum_G*.png`, `absorption_G*.npz`, and `field_e_ref_G*.csv` for the vacuum incident field). Production fields are `field_e_G*.csv` and `field_p_G*.csv`. Hole runs also write `mo_occ.csv` next to those fields, not beside the vacuum reference.

Polarization `single` is the empty cell: no `nanoparticle` block, source left x-polarized and propagating in y. Polarization `parallel` requires the sphere and puts the electric field along the nanoparticle-molecule axis. Back-propagation is true on all eight, including the empty cells. No checkpoints. No `molecule.rotation`. The atom is at 0.02645 μm in the empty cell as well. None of these files has been launched.

G1 through G4 isolate the hole and the nanoparticle. G5 through G8 are those four with the CAP.

## The rest of Step 5

| Run | Double core hole | Nanoparticle | CAP | Polarization |
| --- | --- | --- | --- | --- |
| G1 | off | no | off | single |
| G2 | on | no | off | single |
| G3 | off | 25 nm Au | off | parallel |
| **G4** | on | 25 nm Au | off | parallel |
| G5 | off | no | on | single |
| G6 | on | no | on | single |
| G7 | off | 25 nm Au | on | parallel |
| G8 | on | 25 nm Au | on | parallel |

G3 − G1 and G4 − G2 are the nanoparticle at fixed initial state. G2 − G1 and G4 − G3 are the hole at fixed cell. G8 − G4 is the CAP on the production hybrid. Step 4 remains the real-time molecular reference. This step does not replace it.
