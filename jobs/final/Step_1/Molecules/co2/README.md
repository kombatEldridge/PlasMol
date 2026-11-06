# CO₂

Valence absorption of CO₂. It is a closed-shell singlet. O₂ is the only triplet in Step 1. The deck asks for 16 singlet roots (`notriplet`), the same count as Acetylene, Ethylene, Ethane, CS₂, Formamide, Formaldehyde, N₂O, O₂, and SO₂.

This is a screen, not a PlasMol production run. There is no nanoparticle here and no sudden core hole. It was not selected. From Step 2 on the pair is a sodium atom and a 25 nm gold sphere.

## How it differs from the other calculations in Step 1

The other folders under `Molecules/` are this same NWChem screen on a different species. They share the geometry level (PBE0/6-311G*, Cartesian) and the functional family (LC-ωPBE). Each molecule has its own tuned μ. The root count here matches Acetylene, Ethylene, Ethane, CS₂, Formamide, Formaldehyde, N₂O, O₂, and SO₂, and not the folders outside that list. The common-μ value 0.34272 is the one spectrum in which every molecule used the same μ. That table is in `../../README.md`.

The other kind of calculation in this step is the μ tune, `../../LC-wPBE mu/co2/`. That job finds the range-separation parameter and writes no roots. This folder is the spectrum. trans-Thioindigo's tune is the one that has not been run.

## What was calculated

Neutral CO₂ was optimized at PBE0/6-311G* (Cartesian) from an NCI CACTUS 3D model. `source.txt` records the fetch. PubChem returned HTTP 500 on 2026-09-23 and was not used. `opt.nw` is that optimization. `opt.xyz` is the geometry both spectra use.

The valence calculation is LC-ωPBE linear-response TDDFT, Cartesian 6-311G*, charge 0. Two spectra are stored.

`absorption_mu0.34272.txt` is the common-μ screen, μ = 0.34272. On that screen there is no bright line. The lowest root is 8.856 eV (140 nm, f = 0), and it is at or below 10 eV. The last root in the file is 15.754 eV. The other common-μ folders with roots at or below 10 eV and no bright line are Acetylene, N₂, and N₂O. The Step 1 table is this spectrum, not the tuned one. A line counts as bright when its dipole strength is at least 0.01 and its energy is at most 10 eV.

`tddft.nw` and `absorption.txt` are the rerun at this molecule's own μ, 0.521026. `tddft.out` is that run. On the tuned-μ rerun (μ = 0.521026, `absorption.txt`) there is no bright line. The lowest root is 8.967 eV (138 nm, f = 0), and it is at or below 10 eV. The last root in the file is 16.407 eV. Later steps do not inherit either μ. The sodium production value is 0.351340, from Step 2.

The μ was not computed in this directory. It comes from `../../LC-wPBE mu/co2/`.

## Why it is in the set

The case with a third atom that can donate valence density. The interatomic relaxation is positive for holes on neighboring atoms and negative when both holes sit on the terminal oxygens.

The papers are in `../../README.md`. The sudden double hole used from Step 3 on is the pentanone initial condition in that note, on the sodium doublet, not a two-site free-electron-laser spectrum, and not this linear-response file.
