# trans-Thioindigo

Valence absorption of trans-Thioindigo. It is a closed-shell singlet. O₂ is the only triplet in Step 1. The spectrum is 30 singlets from PySCF, not an NWChem `nroots` line. Adenine, the aminophenols, the pentanones, pentanal, purine, pyrimidine, and the other nucleobases ask NWChem for 30 roots. This folder does not.

This is a screen, not a PlasMol production run. There is no nanoparticle here and no sudden core hole. It was not selected. From Step 2 on the pair is a sodium atom and a 25 nm gold sphere.

## How it differs from the other calculations in Step 1

The other `Molecules/` folders are NWChem screens of a different species, at PBE0/6-311G* Cartesian geometries. This folder is the PySCF exception: the geometry was copied in, there is no `opt.nw`, and there is no tuned-μ rerun. The functional family is still LC-ωPBE, and the basis is still Cartesian 6-311G*. The common-μ value 0.34272 is the comparison in `../../README.md`. This dye is the one bright line in that table on the 35 nm silver plasmon. None of the NWChem molecules is.

The other kind of calculation in this step is the μ tune, `../../LC-wPBE mu/transthioindigo/`. That job has not been run. Every other molecule's tune has.

## What was calculated

`lr_tddft.py` is a PySCF linear-response TDDFT on the existing PBE0/6-311G* geometry in `opt.xyz`. Cartesian 6-311G*, charge 0, singlet, LC-ωPBE, μ = 0.34272, 30 singlets. CACTUS was not used, and this folder has no `opt.nw`. The geometry was copied in. The μ tune has not been run, so nothing here was recomputed at a molecule-specific μ. `absorption.txt` and `absorption_mu0.34272.txt` are the same list.

On that spectrum the lowest bright line is 2.976 eV (417 nm, f = 0.286). There are 9 bright lines at or below 10 eV. Only 2.976 eV (417 nm, f = 0.286) is longward of 380 nm. The other bright lines are ultraviolet. The last root in the file is 7.276 eV. The benzene maximum is 543 nm and the vapor maximum is 508 nm. The root to match, if this dye were selected, would be 417 nm, not 543 nm. It was not selected.

The matching tune directory is `../../LC-wPBE mu/transthioindigo/`. It has a `tune.json` and no `log.out`.

## Why it is in the set

Not in the double-core-hole papers. It is in the screen because a bright valence root falls on a 35 nm silver sphere in water (Mie peak 418.5 nm). That pair was not selected. The production pair is a sodium atom and a 25 nm gold sphere.

The double-core-hole papers are in `../../README.md`. trans-Thioindigo is not in them. The sudden double hole used from Step 3 on is the pentanone initial condition in that note, on the sodium doublet, not this linear-response file.
