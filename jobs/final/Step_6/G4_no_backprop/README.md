# G4_no_backprop

Step 5 G4 with `back_propagation` set to false.

The polarization stays `parallel`. The atom feels the cell and does not source it. This is the same mechanism check as Step 3 D4, moved onto the Gaussian cross section: if G4 moves and this run stays on G2, the sphere is answering the field the atom radiates, not only reshaping the incident Gaussian.

## What this run is

Both folders are Step 5 G4 with one switch thrown. They are not rows of the Step 5 grid. Run them only after that G4 shows a nanoparticle effect. Do not repeat either control across the CAP and across the hole until G4 has moved, and do not copy them into every Step 7 gap.

Shared with G4, and with each other: sudden double hole on MO 0, 25 nm `Au`, no CAP, atom at 0.02645 μm, Gaussian wavelength 0.618422 μm, `fwidth` 2.0, `dt` 0.1 au, `t_end` 10000 au, plot window 1.5 to 4.5 eV, `gamma` omitted, observable `cross_section`. No `molecule.rotation`. Neither file has been launched.

The sodium atom in the resonant series is charge 0, spin 1, `aug-cc-pVTZ` Cartesian, LC-ωPBE with μ = 0.351340. The geometry is `Na.xyz`, the nucleus at the origin. A sudden double hole, where it is on, removes two electrons from MO 0 (Na 1s) and keeps the neutral orbitals. `watch_indices` is omitted, so an occupation file is every logged MO, indices 0 through 6. The β LUMO is index 5, which is why the logged set runs through 6.

Neither folder uses the CAP. The sphere and the 0.02645 μm position are Step 5 G4's, and neither control changes them.

## The two controls

| Run | Polarization | Back-propagation |
| --- | --- | --- |
| G4_perpendicular | perpendicular | on |
| **G4_no_backprop** | parallel | off |

They differ from each other in both switches. One changes the polarization and leaves back-propagation on. The other changes back-propagation and leaves the polarization parallel. A difference between these two folders is not one factor. Each one has to be read against Step 5 G4, not against its neighbor alone.
