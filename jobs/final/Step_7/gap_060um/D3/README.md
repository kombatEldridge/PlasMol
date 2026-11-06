# D3 at a surface gap of 0.060 μm

This is the Step 3 D3 job at a different distance. Field-free sudden hole, 25 nm gold sphere, back-propagation on, no CAP, no Meep source. The atom is allowed to radiate. `molecule.rotation` is the Step 3 orientation check, 84.1114 degrees about `[0.0, -0.89999759, 0.43589487]`. The nucleus stays at the origin.

Inside this gap, D4 turns back-propagation off and G4 replaces the field-free propagation with the Gaussian cross section. D3 is not either of those. There is no D1 in Step 7, because D1 has no sphere. D3 − D1 still uses Step 3 D1. Compare that difference with the same difference at 0.02645 μm.

## How this gap differs from the other Step 7 gaps

The sphere radius stays 0.025 μm. The surface gap here is 0.060 μm, so the molecule x is 0.085 μm (`position` `[0.085, 0, 0]`). The other gaps in this step are 0.005 μm, molecule x 0.030 μm and 0.030 μm, molecule x 0.055 μm. Do not launch this file yet. The atom sits in the PML of `cell_length` 0.2. The non-PML interior, with a 0.05 μm PML, ends at 0.05 μm, and 0.085 μm is past that. Widen the cell. Do not pull the atom back in by shrinking the gap. The 0.005 μm gap is the one that already fits.

The baseline gap, one Yee pixel with the atom at 0.02645 μm, is not repeated in Step 7. Those runs are Steps 3 and 5. Step 8, once it is filled, uses that one-pixel gap rather than any of these three. The nuclei do not change with the gap. Only x changes.

None of the nine Step 7 files has been launched.

## Jobs at this gap

| Job | Drive | Back-propagation | CAP | Driver | Polarization |
| --- | --- | --- | --- | --- | --- |
| **D3** | none | on | off | plasmol | n/a |
| D4 | none | off | off | plasmol | n/a |
| G4 | Gaussian, 0.618422 μm | on | off | absorption | parallel |

## Gaps in Step 7

| Folder | Surface gap | Molecule x | Cell |
| --- | --- | --- | --- |
| gap_005um | 0.005 μm | 0.030 μm | fits in `cell_length` 0.2 |
| gap_030um | 0.030 μm | 0.055 μm | atom sits in the PML of `cell_length` 0.2 |
| **gap_060um** | 0.060 μm | 0.085 μm | atom sits in the PML of `cell_length` 0.2 |

The sodium atom in the resonant series is charge 0, spin 1, `aug-cc-pVTZ` Cartesian, LC-ωPBE with μ = 0.351340. The geometry is `Na.xyz`, the nucleus at the origin. A sudden double hole, where it is on, removes two electrons from MO 0 (Na 1s) and keeps the neutral orbitals. `watch_indices` is omitted, so an occupation file is every logged MO, indices 0 through 6. The β LUMO is index 5, which is why the logged set runs through 6.

No Step 7 run uses the CAP. The sphere is the same 25 nm Meep `Au` sphere in water as in Steps 3 and 5. The atom is not at 0.02645 μm in this file. It is at the molecule x in the table above.

D3 and D4 here are dynamics. Do not treat them as cross sections. G4 is the cross section. D1 and G2 are intentionally absent.
