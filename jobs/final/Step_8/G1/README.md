# G1

Neutral Gaussian cross section in an empty cell. The counterpart of Step 5 G1.

Polarization `single`, no nanoparticle, back-propagation on, no hole, no CAP. The wavelength, the plot window, and the molecule x are placeholders. `fwidth` is 2.0, as in the resonant Gaussians. G2 adds the hole. G3 adds the sphere. This file is the empty-cell reference for those two, not a δ-kick. K1 remains the real-time molecular spectrum.

## What is still a placeholder

Do not launch this file. Every molecule field and every nanoparticle field that depends on the pair is still a `PLACEHOLDER_DETUNED` token. Step 8 is the only step that still has those tokens.

The strings that are filled are a template, not the sodium production setup and not a decision about the detuned molecule. Charge 0, spin 0, basis `6-311G*` Cartesian, geometry `PLACEHOLDER_DETUNED_MOLECULE.xyz`, μ `PLACEHOLDER_DETUNED_MU`. The resonant series is spin 1, `aug-cc-pVTZ`, μ = 0.351340, core MO 0, 25 nm `Au`, atom at 0.02645 μm. If the detuned pair keeps the sodium atom, copy that molecule block, including the omitted watch list, and change the nanoparticle. In that case D1, K1, K3, G1, and G2 can be reused instead of rerun. Do not reuse the Step 9 `NaDrude` dielectric as this pair. The surface gap stays one Yee pixel, so this comparison is resonance and not distance.

The CAP grid (D2, D5, D6, K2, K4, G5 through G8) and the Step 6 and Step 7 controls are not here. They answer lifetime, polarization, and distance. This step asks whether the resonances have to overlap. The resonant campaign is a sodium atom and a 25 nm Rakić sphere, and that sphere already misses the sodium root by 0.335 eV. "Detuned" here means a pair chosen so they do not overlap. It does not mean rerunning that gold sphere.

## The rest of Step 8

| Run | Driver | Hole | Nanoparticle | Back-propagation | Polarization | Time |
| --- | --- | --- | --- | --- | --- | --- |
| survey | core_hole | print MOs 0-5 | no | n/a | no | 1 au, unused |
| D1 | quantum | one core MO | no | n/a | no | 400 au, dt 0.05 |
| D3 | plasmol | one core MO | yes | on | no | 10000 au, dt 0.1 |
| D4 | plasmol | one core MO | yes | off | no | 10000 au, dt 0.1 |
| K1 | absorption | no | no | n/a | no | 4000 au, dt 0.05 |
| K3 | absorption | one core MO | no | n/a | no | 4000 au, dt 0.05 |
| **G1** | absorption | no | no | on | single | 10000 au, dt 0.1 |
| G2 | absorption | one core MO | no | on | single | 10000 au, dt 0.1 |
| G3 | absorption | no | yes | on | parallel | 10000 au, dt 0.1 |
| G4 | absorption | one core MO | yes | on | parallel | 10000 au, dt 0.1 |

D3 − D1 against the resonant D3 − D1 is the dynamics comparison. G4 − G2 against the resonant G4 − G2 is the cross-section comparison. K1 is the root the Gaussian is centered on. The survey is how the core index is assigned for this molecule, which the sodium atom table could not do, because every sodium MO is sodium.
