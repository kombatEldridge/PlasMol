# Na

Valence absorption of the sodium atom. It is a doublet, spin 1. O₂ is the only triplet in Step 1. Every other folder is a closed-shell singlet. The deck asks for 24 roots, the same count as Benzene, BeO, and LiF. Those three need the extra roots to get past 10 eV. This one does not: the eighth root is already past 10 eV.

This is a screen, not a PlasMol production run. There is no nanoparticle here and no sudden core hole. The atom is the one used from Step 2 on, but not at this basis and not at this μ. The production calculation is `aug-cc-pVTZ`, μ = 0.351340, and its bright root is 2.004849 eV.

## How it differs from the other calculations in Step 1

The other folders under `Molecules/` are this same NWChem screen on a different species. They share the functional family (LC-ωPBE) and, except trans-thioindigo, the basis (Cartesian 6-311G*). Each species has its own tuned μ. The root count here matches Benzene, BeO, and LiF. The common-μ value 0.34272 is the one spectrum in which every molecule used the same μ. That table is in `../../README.md`.

The geometry is not one of theirs. There is no CACTUS file, no `opt.nw`, and no optimization. A single nucleus has no internal coordinate. `opt.xyz` places Na at the origin.

The other kind of calculation in this step is the μ tune, `../../LC-wPBE mu/na/`. That job finds the range-separation parameter and writes no roots. This folder is the spectrum.

## What was calculated

The valence calculation is LC-ωPBE linear-response TDDFT, Cartesian 6-311G*, charge 0, multiplicity 2. Two spectra are stored.

`absorption_mu0.34272.txt` is the common-μ screen, μ = 0.34272. `tddft_mu0.34272.out` is that run. On that screen the lowest bright line is 1.999 eV (620 nm, f = 0.329) on each component of the threefold 3s → 3p line. There is 1 bright line at or below 10 eV, counting that threefold line as one. Only that line is longward of 380 nm. The last root in the file is 28.836 eV. The Step 1 table is this spectrum, not the tuned one. A line counts as bright when its dipole strength is at least 0.01 and its energy is at most 10 eV.

`tddft.nw` and `absorption.txt` are the rerun at this atom's own μ, 0.352377. `tddft.out` is that run. On the tuned-μ rerun the lowest bright line is 1.996 eV (621 nm, f = 0.329) on each component of the same threefold line. There is 1 bright line at or below 10 eV. Only that line is longward of 380 nm. The last root in the file is 28.922 eV. `absorption_levels.png` draws this file, not the common-μ one. The plot also keeps weaker positive strengths out to 12 eV, so the faint line near 4.21 eV (f = 0.001) is on the figure and is not in the bright-line count.

Later steps do not inherit either μ. The sodium production value is 0.351340, from Step 2, on `aug-cc-pVTZ`. The production bright root is 2.004849 eV (618 nm, f = 0.327 on each component). This screen is about 0.01 eV below that root. It is not the experimental 589 nm D line.

The μ was not computed in this directory. It comes from `../../LC-wPBE mu/na/`.

## Why it is in the set

The screen asks which species has a bright valence line near a metallic plasmon. The resonant series uses this atom, so the comparison figure has to be able to show it at the same level as the other rows. The 620 nm line is red of the gold band on that figure, which ends at 580 nm, and it is not on the 418 nm silver peak. BeO is the only literature molecule with a visible line, and that line is weak. This one is not weak.

There is no double-core-hole section for sodium in `../../README.md`. The sudden double hole used from Step 3 on is the pentanone initial condition in that note, on this doublet, not a two-site free-electron-laser spectrum, and not this linear-response file.
