# Distance

How the nanoparticle effect depends on the surface gap. Run this only after resonant G4 − G2 is nonzero at the baseline. That baseline is Steps 3 and 5: the molecule at 0.02645 μm, one Yee pixel off the 0.025 μm sphere. It is not repeated here.

Three other gaps, on the resonant pair only. Step 8 uses that same one-pixel gap, so a change of pair is not also a change of distance.

| Folder | Surface gap | Molecule x |
| --- | --- | --- |
| `gap_005um/` | 0.005 μm | 0.030 |
| `gap_030um/` | 0.030 μm | 0.055 |
| `gap_060um/` | 0.060 μm | 0.085 |

The molecule x in each file is that number, the 0.025 μm radius plus the gap. The other resonant tokens match Steps 3 and 5. The non-PML interior of `cell_length` 0.2 ends at 0.05 μm, so 0.030 fits and 0.055 and 0.085 sit in the PML. Widen the cell on those two gaps before launching. Do not change the gap to make the cell fit.

| File | What it repeats |
| --- | --- |
| `D3/D3.json` | Field-free hole, nanoparticle, back-propagation on |
| `D4/D4.json` | The same hole, nanoparticle, back-propagation off |
| `G4/G4.json` | DCH plus nanoparticle, Gaussian on the bright root, no CAP |

D3 − D1 and G4 − G2, against the same differences at 0.02645 μm, are the distance effect. D1 and G2 have no nanoparticle, so they are not copied here. D4 is the check on mechanism: if the gap changes D3 and G4 but not D4, the distance dependence is the field coming back, not the molecule simply sitting closer to the metal.

No CAP rows. No perpendicular run. The orientation pasted into Step 3 D3 and D4 is pasted into these D3 and D4 files as well. The nuclei do not change with the gap.
