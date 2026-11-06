# trans-Thioindigo, μ tune

Koopmans tune of the LC-ωPBE range-separation parameter for trans-Thioindigo. Spin is 0, the closed-shell singlet. O₂ is the only tune here with spin 2.

This is not a spectrum, and it is not a propagation. The spectrum of the same molecule is `../../Molecules/transthioindigo/`.

## How it differs from the other tunes

Every folder under `LC-wPBE mu/` is the same `tune` driver. `dt` is 0.05 au and `t_end` is 1.0 au. Both are unused: the driver returns before any time step. The basis is 6-311G*, the charge is 0, the functional is LC-ωPBE, and `lrc_parameter` is the string `"tune"`. There is no CAP block, so this scan does not produce an ε₀.

The folders differ by the geometry they point at, and therefore by the μ they return. This one has not been run. There is no `log.out` and no row in `../mu.txt`. The geometry string is `../../Molecules/transthioindigo/opt.xyz`, which does resolve to `Molecules/transthioindigo/opt.xyz`. Leave `lrc_parameter` as `"tune"` if it is launched. Do not paste a number back into the JSON.

Until that happens, trans-thioindigo has only the μ = 0.34272 PySCF spectrum in `../../Molecules/transthioindigo/`. The other molecules have a tuned-μ NWChem rerun. This is the missing one.

## How it differs from Step 2

Step 2 tunes the sodium atom, not this molecule. That tune uses `aug-cc-pVTZ`, spin 1, and a CAP block whose `eps0` is also `"tune"`. It returned μ = 0.351340 and ε₀ = 0.009767 Ha. Those are the production numbers from Step 2 on. They are not this folder's result, and this folder's μ is not copied forward.

## How it differs from the spectrum folder

`Molecules/transthioindigo/` does not optimize a geometry. `opt.xyz` was copied in, and `lr_tddft.py` is the μ = 0.34272 spectrum. This folder would only report μ. It has not been run, so that spectrum was never rewritten.
