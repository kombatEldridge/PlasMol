# K1

Neutral sodium, no CAP. The ordinary real-time absorption spectrum, and the check that the 2.004849 eV linear-response root (the Gaussian wavelength 0.618422 μm) is in this window.

## What this run is

The sodium atom in the resonant series is charge 0, spin 1, `aug-cc-pVTZ` Cartesian, LC-ωPBE with μ = 0.351340. The geometry is `Na.xyz`, the nucleus at the origin. A sudden double hole, where it is on, removes two electrons from MO 0 (Na 1s) and keeps the neutral orbitals. `watch_indices` is omitted, so an occupation file is every logged MO, indices 0 through 6. The β LUMO is index 5, which is why the logged set runs through 6.

The CAP, on the runs that have one, is static: `gam0` 1, `xi` 0.5, `eps0` 0.009767 Ha, `clamp` 100. It does not replace μ.

No file in this step has a nanoparticle or a Meep cell. The atom position and the gold sphere show up in Steps 3 and 5, not here.

Molecule only. Driver `absorption`, `polarization` `full`, kick strength 0.001 au, one step wide, peak at 0. `dt` 0.05 au, `t_end` 4000 au. Fourier window `gamma` 0.01, plotted from 1.5 to 15 eV. The observable is `cross_section`. These jobs can checkpoint. None of them has been launched. `t_end` and `gamma` have to stay matched across K1 through K4. Do not raise `max_ev` unless K1 shows weight above 15 eV, and then raise all four together.

The linear-response root, 2.004849 eV, sits inside the window. Step 1's linear-response screen is a different basis, a different μ, and a different molecule. It is not a substitute for K1.

No `core_hole` block, so this job writes no `mo_occ.csv`. The plotted curve averages the x, y, and z kicks, so a nuclear rotation would not change it. The separate dipole files are the valence axis if a later Gaussian job needs that axis on +x. They are not the rotation pasted onto Step 3 D3 through D6.

K2 − K1 is the CAP on the neutral atom. K3 − K1 is the sudden hole with no CAP. K4 turns both on. None of the four has a nanoparticle. The empty-cell Gaussians in Step 5 are not a substitute for this spectrum.

## The rest of Step 4

| Run | Double core hole | CAP |
| --- | --- | --- |
| **K1** | off | off |
| K2 | off | on |
| K3 | on | off |
| K4 | on | on |

K1 and K2 have no core hole. K3 and K4 have the hole and omit `watch_indices`. No file in this step contains a nanoparticle, a Meep cell, or a placeholder.
