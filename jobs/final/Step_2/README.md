# Step 2

Two sodium calculations live in this directory, because this step is one molecule and it was not split into run folders. They are not two versions of the same job.

## survey.json

Driver `core_hole`. Neutral sodium at the production μ, then the atoms that contribute to MOs 0 through 5, then exit. The counts in `mo_removal_index_dict` are ignored. That list is what gets printed, not a hole. `t_end` is unused. The table is `mo_survey.txt`. `log.out` is the same launch.

Every MO 0 through 5 is about 100% Na (100.17, 99.65, 100.85, 101.05, 101.38, 99.76). On one atom that print cannot name the shell. The 1s assignment is the orbital energy at this μ: index 0 is 1s at −38.175 Ha, index 1 is 2s, indices 2 through 4 are 2p, and index 5 is the 3s HOMO at −0.192488 Ha. SCF energy −162.20671104 Ha. The production hole from Step 3 on is index 0. The survey dictionary is not the hole.

This job does not propagate and does not tune anything. μ is already the number 0.351340.

## tune.json

Driver `tune`. Koopmans scan of μ and of the CAP threshold ε₀. Both fields are still the string `"tune"` in the JSON. `tune.log` is the result: μ = 0.351340, ε₀ = 0.009767 Ha, in 29.3 s. Do not paste those numbers back into `tune.json`. The scan has to be repeatable.

For spin 1 the cation is closed shell, and the neutral α HOMO is the Koopmans target. The first α virtual sits on ε₀ (the empty 3p, +0.009767 Ha), so the CAP does not damp it. The first orbital above the cutoff is index 9.

## How the two differ

The survey asks which atoms are in MOs 0 through 5, at a μ that has already been chosen. The tune is how that μ and the CAP threshold were chosen, and it does not print the MO table. The survey has no CAP block. The tune has one, with `eps0` set to `"tune"`.

Both use `Na.xyz`, charge 0, spin 1, `aug-cc-pVTZ` Cartesian, LC-ωPBE. That is already different from every tune under `Step_1/LC-wPBE mu/`. Those use 6-311G*, spin 0 except O₂ at spin 2, and they do not tune ε₀. The 3-pentanone pair μ = 0.34272 and ε₀ = 0.003028 Ha is not this tune. Libxc's default μ = 0.4 is not this tune either.

Nothing in this directory is a time propagation, a nanoparticle, or a spectrum. Those start at Step 3.
