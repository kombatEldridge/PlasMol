# G4

Step 5 G4 with the sphere material changed from `Au` to `NaDrude`. Sudden hole, polarization `parallel`, back-propagation on, no CAP.

G4 − G2 uses Step 5 G2. G3 is this spectrum without the hole. D3 is the field-free radiating case of the same dielectric. If this sphere moves G4 and the gold sphere does not, the gold result was the 0.335 eV miss. If both move by about the same amount, the sphere is doing something that does not need its plasmon on the atom.

## What this run is

The sodium atom in the resonant series is charge 0, spin 1, `aug-cc-pVTZ` Cartesian, LC-ωPBE with μ = 0.351340. The geometry is `Na.xyz`, the nucleus at the origin. A sudden double hole, where it is on, removes two electrons from MO 0 (Na 1s) and keeps the neutral orbitals. `watch_indices` is omitted, so an occupation file is every logged MO, indices 0 through 6. The β LUMO is index 5, which is why the logged set runs through 6.

The CAP, on the runs that have one, is static: `gam0` 1, `xi` 0.5, `eps0` 0.009767 Ha, `clamp` 100. It does not replace μ. No run in this step uses it. The comparison with a CAP on is Steps 3 and 5, not a missing file here.

The nanoparticle block is the only intended change from Steps 3 and 5. `material` is `NaDrude`, and `material_file` is `../na_drude.py`. That name has to be new. `load_meep_material` rejects anything that is already a `Medium` in Meep's library, including `Au`. The poles are not written in the JSON. `na_drude.py` assigns `meep.materials.NaDrude`: one Drude pole, ε∞ = 1, σ = 1, plasma frequency 3.6340635899 μm⁻¹ (4.505664 eV), damping 0.0427473847 μm⁻¹ (0.053 eV, the Rakić free-electron γ, not refitted). `dielectric.py` is the classical design script, not the file PlasMol loads. It raised ω_p from the quasistatic 4.272 eV until the Mie absorption of this 25 nm sphere in water landed on the sodium root, 2.004849 eV. Extinction sits about 1 meV higher. The Rakić sphere of the same radius absorbs at 2.340 eV.

Radius 0.025 μm, atom at 0.02645 μm, `cell_length` 0.2, water index 1.33. No CAP, no Step 6 controls, no Step 7 gaps. D3 and D4 carry the Step 3 orientation block. G3 and G4 do not. The Gaussian wavelength is 0.618422 μm, `fwidth` 2.0, plot window 1.5 to 4.5 eV. None of the four files contains a placeholder. None of them has been launched. Do not retune μ or ε₀. Do not change the radius to chase the peak. Do not use this dielectric as the Step 8 pair.

D3 and D4 are dynamics. There is no incident field, so they are not cross sections. G3 and G4 publish `cross_section`.

## The rest of Step 9

| Run | Drive | Double core hole | Back-propagation | Driver | Polarization |
| --- | --- | --- | --- | --- | --- |
| D3 | none | on | on | plasmol | n/a |
| D4 | none | on | off | plasmol | n/a |
| G3 | Gaussian | off | on | absorption | parallel |
| **G4** | Gaussian | on | on | absorption | parallel |

Read D3 − D1 and G4 − G2 against the same differences with the gold sphere. D1 and G2 have no sphere, so they were not copied into this step.
