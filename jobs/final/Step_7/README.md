# Distance

How the nanoparticle effect depends on the surface gap. Run this only after resonant G4 − G2 is nonzero at the baseline gap of 0.015 μm. That baseline is Steps 3 and 5. It is not repeated here.

Three other gaps, on the resonant pair only. The detuned pair in Step 8 stays at 0.015 μm, so a change of pair is not also a change of distance.

| Folder | Surface gap | Molecule x |
| --- | --- | --- |
| `gap_005um/` | 0.005 μm | radius + 0.005 |
| `gap_030um/` | 0.030 μm | radius + 0.030 |
| `gap_060um/` | 0.060 μm | radius + 0.060 |

The x token in each file is `PLACEHOLDER_RESONANT_X_GAP005_UM`, `PLACEHOLDER_RESONANT_X_GAP030_UM`, or `PLACEHOLDER_RESONANT_X_GAP060_UM`. Replace it with the radius plus that gap, in μm. The other resonant tokens match Steps 3 and 5. `cell_length` stays 0.2 until the sphere, the gap, and the PML no longer fit. Widen the cell on that gap’s files before launching. Do not change the gap to make the cell fit.

| File | What it repeats |
| --- | --- |
| `D3/D3.json` | Field-free hole, nanoparticle, back-propagation on |
| `D4/D4.json` | The same hole, nanoparticle, back-propagation off |
| `G4/G4.json` | DCH plus nanoparticle, Gaussian on the bright root, no CAP |

D3 − D1 and G4 − G2, against the same differences at 0.015 μm, are the distance effect. D1 and G2 have no nanoparticle, so they are not copied here. D4 is the check on mechanism: if the gap changes D3 and G4 but not D4, the distance dependence is the field coming back, not the molecule simply sitting closer to the metal.

No CAP rows. No perpendicular run. The orientation pasted into Step 3 D3 and D4 is pasted into these D3 and D4 files as well. The nuclei do not change with the gap.
