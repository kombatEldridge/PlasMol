# G4 at a surface gap of 0.030 μm

This is the Step 5 G4 job at a different distance. Sudden hole, 25 nm gold sphere, back-propagation on, no CAP, polarization `parallel`, Gaussian wavelength 0.618422 μm, `fwidth` 2.0, plot window 1.5 to 4.5 eV, `gamma` omitted. The published curve is `cross_section`. There is no `molecule.rotation` on the Gaussian jobs. D3 and D4 in this same gap carry the rotation. G4 does not.

G4 − G2 uses Step 5 G2, which has no sphere and is not copied here. Inside this gap, D3 and D4 are the field-free dynamics at the same x. They are not cross sections, and this file is not a field-free run. No perpendicular control and no CAP row were added at this gap.

## How this gap differs from the other Step 7 gaps

The sphere radius stays 0.025 μm. The surface gap here is 0.030 μm, so the molecule x is 0.055 μm (`position` `[0.055, 0, 0]`). The other gaps in this step are 0.005 μm, molecule x 0.030 μm and 0.060 μm, molecule x 0.085 μm. Do not launch this file yet. The atom sits in the PML of `cell_length` 0.2. The non-PML interior, with a 0.05 μm PML, ends at 0.05 μm, and 0.055 μm is past that. Widen the cell. Do not pull the atom back in by shrinking the gap. The 0.005 μm gap is the one that already fits.

The baseline gap, one Yee pixel with the atom at 0.02645 μm, is not repeated in Step 7. Those runs are Steps 3 and 5. Step 8, once it is filled, uses that one-pixel gap rather than any of these three. The nuclei do not change with the gap. Only x changes.

None of the nine Step 7 files has been launched.

## Jobs at this gap

| Job | Drive | Back-propagation | CAP | Driver | Polarization |
| --- | --- | --- | --- | --- | --- |
| D3 | none | on | off | plasmol | n/a |
| D4 | none | off | off | plasmol | n/a |
| **G4** | Gaussian, 0.618422 μm | on | off | absorption | parallel |

## Gaps in Step 7

| Folder | Surface gap | Molecule x | Cell |
| --- | --- | --- | --- |
| gap_005um | 0.005 μm | 0.030 μm | fits in `cell_length` 0.2 |
| **gap_030um** | 0.030 μm | 0.055 μm | atom sits in the PML of `cell_length` 0.2 |
| gap_060um | 0.060 μm | 0.085 μm | atom sits in the PML of `cell_length` 0.2 |

The sodium atom in the resonant series is charge 0, spin 1, `aug-cc-pVTZ` Cartesian, LC-ωPBE with μ = 0.351340. The geometry is `Na.xyz`, the nucleus at the origin. A sudden double hole, where it is on, removes two electrons from MO 0 (Na 1s) and keeps the neutral orbitals. `watch_indices` is omitted, so an occupation file is every logged MO, indices 0 through 6. The β LUMO is index 5, which is why the logged set runs through 6.

No Step 7 run uses the CAP. The sphere is the same 25 nm Meep `Au` sphere in water as in Steps 3 and 5. The atom is not at 0.02645 μm in this file. It is at the molecule x in the table above.

D3 and D4 here are dynamics. Do not treat them as cross sections. G4 is the cross section. D1 and G2 are intentionally absent.
