# Na, μ tune

Koopmans tune of the LC-ωPBE range-separation parameter for the sodium atom. Spin is 1. The cation is closed shell. O₂ is the only tune here with spin 2. Every other tune uses spin 0.

This is not a spectrum, and it is not a propagation. The spectrum of the same atom is `../../Molecules/na/`.

## How it differs from the other tunes

Every folder under `LC-wPBE mu/` is the same `tune` driver. `dt` is 0.05 au and `t_end` is 1.0 au. Both are unused: the driver returns before any time step. The basis is 6-311G*, the charge is 0, the functional is LC-ωPBE, and `lrc_parameter` is the string `"tune"`. There is no CAP block, so this scan does not produce an ε₀.

The folders differ by the geometry they point at, and therefore by the μ they return. This one has been run. `log.out` reports μ = 0.352377. The same value is the `na` row of `../mu.txt` and the `cam` value in `../../Molecules/na/tddft.nw`. The JSON was not overwritten with that number.

The geometry string is `../../Molecules/na/opt.xyz`. That file is the nucleus at the origin, and the path resolves. A fresh launch of this JSON would find it.

## How it differs from Step 2

Step 2 tunes the same atom at a different basis. That tune uses `aug-cc-pVTZ`, spin 1, and a CAP block whose `eps0` is also `"tune"`. It returned μ = 0.351340 and ε₀ = 0.009767 Ha. Those are the production numbers from Step 2 on. They are not this folder's result. This folder's μ is not copied forward, even though the two values are close.

## How it differs from the spectrum folder

`Molecules/na/` computes the linear-response roots at this μ and at the common screen value 0.34272. This folder does neither. It only reports μ. The spectrum folder does not optimize a geometry either. One nucleus has nothing to optimize.
