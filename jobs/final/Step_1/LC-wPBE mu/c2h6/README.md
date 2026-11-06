# Ethane, μ tune

Koopmans tune of the LC-ωPBE range-separation parameter for ethane. Spin is 0, the closed-shell singlet. O₂ is the only tune here with spin 2.

This is not a spectrum, and it is not a propagation. The spectrum of the same molecule is `../../Molecules/c2h6/`.

## How it differs from the other tunes

Every folder under `LC-wPBE mu/` is the same `tune` driver. `dt` is 0.05 au and `t_end` is 1.0 au. Both are unused: the driver returns before any time step. The basis is 6-311G*, the charge is 0, the functional is LC-ωPBE, and `lrc_parameter` is the string `"tune"`. There is no CAP block, so this scan does not produce an ε₀.

The folders differ by the geometry they point at, and therefore by the μ they return. This one has been run. `log.out` reports μ = 0.431186. The same value is the `c2h6` row of `../mu.txt` and the `cam` value in `../../Molecules/c2h6/tddft.nw`. The JSON was not overwritten with that number.

The geometry string is `../../c2h6/opt.xyz`. From this directory that is `Step_1/c2h6/opt.xyz`, which is not there now. The optimized geometry is `Step_1/Molecules/c2h6/opt.xyz`. The molecule directories sat directly under Step 1 when the tune ran. The result is already in `log.out`. A fresh launch of this JSON would not find the geometry.

## How it differs from Step 2

Step 2 tunes the sodium atom, not this molecule. That tune uses `aug-cc-pVTZ`, spin 1, and a CAP block whose `eps0` is also `"tune"`. It returned μ = 0.351340 and ε₀ = 0.009767 Ha. Those are the production numbers from Step 2 on. They are not this folder's result, and this folder's μ is not copied forward.

## How it differs from the spectrum folder

`Molecules/c2h6/` optimizes the geometry at PBE0/6-311G* and computes linear-response roots. This folder does neither. It only reports μ.
