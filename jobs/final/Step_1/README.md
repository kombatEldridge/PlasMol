# Choose the molecule and the nanoparticle

This is the first step. The pair it picks is what the later jobs use: trans-thioindigo, and a 35 nm silver sphere (Meep material `Ag`, host index 1.33). `transthioindigo.xyz` in this directory is the geometry those jobs read. Do not edit it.

The question is which small dye and which bare sphere have resonances on top of each other, with the same functional as the rest of the series and a Meep dielectric the FDTD jobs can load.

## What was run

Two calculations have been run. Both are in this directory.

The molecule is gas-phase linear-response TDDFT, `lr_tddft.py`. Geometry `transthioindigo.xyz` (PBE0/6-311G*, copied from `jobs/Geometries/`). Excitation functional is the campaign one, not the geometry functional: LC-ωPBE, μ = 0.34272, Cartesian 6-311G*, 30 singlets. Uniform Lorentzian, half-width 0.15 eV, area of each line equal to its oscillator strength. Results: `lr_thioindigo_states.txt`, `lr_thioindigo_spectrum.txt`, `lr_thioindigo_spectrum.png`.

The sphere is Mie theory, `ag_mie.py`, with Meep's `Ag` permittivity and host index 1.33. It writes the absorption of the 35 nm and 40 nm spheres (`ag_mie_spectrum.txt`, `ag_mie_spectrum.png`) and the peak wavelength against diameter for both `Ag` and `Ag_visible` (`ag_mie_size.txt`, `ag_mie_size.png`).

`d35/d35.json` is the series sphere as a Meep flux job (driver `np_abs_cross_sec`, material `Ag`, radius 0.0175 μm). The source band is 300–550 nm (`frequency` 2.575758 μm⁻¹, `fwidth` 1.515152), inside the Rakic fit. That JSON has not been run. Do not point it at `Ag_visible`.

```bash
conda activate meep
python jobs/final/Step_1/lr_tddft.py
python jobs/final/Step_1/ag_mie.py
```

The linear-response script is the expensive one (about an hour). It does not need to be repeated. The Mie script takes a few seconds.

## Dyes that were considered

The first target was a 50 nm gold sphere, whose plasmon in water sits near 540 nm. The dye had to have a bright transition near that wavelength and be small enough for this DFT setup. Rhodamine 6G (530 nm, ε about 1.2×10⁵) is the usual partner for that plasmon and was set aside: the cation is about 64 atoms. The Cy3 chromophore (546 nm, ε about 1.3×10⁵) is about 60 atoms and was set aside for the same reason. Pentacene's gas-phase origin is 536 nm, but that short-axis band is weak. The strong pentacene band is in the ultraviolet.

Ten smaller candidates were left. Experimental wavelengths below are solution or vapor maxima, not calculated roots. Only trans-thioindigo was calculated. "Active" means an allowed π→π* or charge-transfer band with ε of order 10⁴–10⁵, not a dark n→π*.

| Molecule | Atoms | Measured λmax | ε (M⁻¹ cm⁻¹) | Why it stopped here |
| --- | ---: | --- | --- | --- |
| trans-Thioindigo | 28 | 543 nm in benzene; 508 nm in vapor | ~1.4×10⁴ | Calculated. See below |
| Indigo | 30 | 539–546 nm in vapor; ~600 nm in ethanol | ~1×10⁴ | The vapor band is the 540 nm one. Any solution spectrum is the 600 nm band. Not calculated |
| Phenol blue | 31 | 552 nm in hexane; 668–684 nm in water | allowed charge-transfer band | Strongly solvatochromic. A gas-phase root would be compared with the hexane number, not the water number. Not calculated |
| Streptocyanine, n = 3 | 32 | 519 nm in dichloromethane | 2.07×10⁵ | Brightest small dye near 540 nm. A polymethine, so LC-ωPBE is expected to push it to the blue. Not calculated |
| Methyl red, red form | 35 | 523–526 nm in water at pH 4.2 | ~4×10⁴ | The yellow anion, at 430 nm, is a different protonation state. Not calculated |
| Pyronin Y | 39 | 546–552 nm in 50% ethanol | 1.17×10⁵ | Bright, and already large for a real-time run. Not calculated |
| Pararosaniline | 40 | 542–549 nm in 50% ethanol | ~8–10×10⁴ | Sits on 540 nm. Not calculated |
| Safranin O | 43 | 530–534 nm in 50% ethanol | ~4.4–5.5×10⁴ | Not calculated |
| Quinaldine red | 46 | 528 nm in ethanol | ≥5.8×10⁴ | Upper end of the size range. Not calculated |
| 3,3′-Diethylthiacarbocyanine | 46 | 559 nm in ethanol | 1.61×10⁵ | Brightest of the larger ones, 19 nm red of 540 nm. Not calculated |

Two more were kept only as backups if LC-ωPBE moved a cyanine off a gold plasmon: the n = 4 streptocyanine (36 atoms, 625 nm, ε 2.95×10⁵ in dichloromethane) and thionine (26 atoms, 602 nm, ε 7.8×10⁴ in ethanol). Resorufin (22 atoms, 572 nm, ε about 5–6×10⁴) is the smallest bright dye in the neighborhood and was left out because 572 nm is already outside the original 540 nm window.

Counterions are not part of any of these chromophores.

## What the thioindigo calculation did

The SCF energy is −1560.721851 Ha. The only bright root in the visible is state 1.

| State | Energy | Wavelength | f |
| --- | ---: | ---: | ---: |
| 1 | 2.976 eV | 417 nm | 0.286 |
| 6 | 4.856 eV | 255 nm | 0.452 |
| 10 | 5.800 eV | 214 nm | 0.633 |

States 2–5 (374 down to 339 nm) have oscillator strength zero. There is no second visible band hiding at 540 nm.

The benzene maximum is 543 nm (2.28 eV), 0.69 eV below the calculated root. That gap is not a missing state. The vapor maximum is 508 nm, so the solvent accounts for about 0.16 eV of it. The rest is the functional. LC-ωPBE puts full Hartree–Fock exchange on the long-range part of this transition, and the thioindigo π→π* has charge-transfer character across the central double bond. PBE0 with a continuum solvent reproduces the solution band to about 0.03 eV. The basis and the PBE0 geometry move an indigoid λmax by about 10 nm, which is not this discrepancy. LC-ωPBE puts this dye's color at 417 nm.

So the resonance that has to be matched is the calculated root, 417 nm, not the benzene maximum and not the 540 nm gold plasmon.

## The sphere that meets it

Material `Ag`, the Rakic Lorentz–Drude model (Applied Optics 37, 5271, 1998). Meep allows it from 248 nm to 12.4 μm. Host index 1.33.

`Ag_visible` was not used. It is a Palik fit allowed only from 400 to 800 nm, and Meep marks it unstable. At 417 nm the two dielectrics differ (Rakic −4.01 + 0.59i, Palik −4.62 + 0.74i), and the 400 nm edge of the Palik fit is 17 nm from the dye line.

| Sphere | Absorption peak | Cross section at 417 nm |
| --- | --- | --- |
| `Ag`, 40 nm | 422.5 nm | 92% of the peak |
| `Ag`, 32.6 nm | 416.5 nm | the peak (7.4×10³ nm²) |
| `Ag`, 35 nm | 418.5 nm | 99% of the peak. This is the diameter the later jobs use |
| `Ag_visible`, 48 nm | 416.5 nm | the peak, different material |

The Mie intersection is 32.6 nm. The jobs use a **35 nm** silver sphere, Meep material `Ag`, in water. That peak is at 418.5 nm, about 2 nm red of the thioindigo root. Later jobs read `transthioindigo.xyz` from this directory and use that sphere. A 48 nm sphere is the match only if the material is `Ag_visible`, which this step does not use. The two materials are not interchangeable.
