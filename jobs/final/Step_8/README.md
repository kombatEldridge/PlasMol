# Detuned pair

Same protocol as the resonant campaign, on a nanoparticle–molecule pair whose resonances do not overlap. The plasmon is not on the bright root that the Gaussian is centered on.

These files are the key runs only. The CAP grid (D2, D5, D6, K2, K4, G5–G8) and the two G4 controls stay in Steps 3–6. The gap scan stays in Step 7. They answer lifetime, polarization, and distance, not whether the resonances have to overlap. The gap here stays 0.015 μm. Do not add the controls here until the detuned G4 has moved and D3 − D4 does not already explain it.

Every nanoparticle field and every molecule field is a `PLACEHOLDER_DETUNED` token. The list of tokens, and the rule for reusing the resonant molecule, is in `progress.md`. Do not launch a file that still contains `PLACEHOLDER_`.

| File | Comparison |
| --- | --- |
| `survey/survey.json` | Which MO is the core. MOs 0–5 are printed, not ionized |
| `D1/D1.json` | Free hole |
| `D3/D3.json` | Nanoparticle, back-propagation on |
| `D4/D4.json` | Nanoparticle, back-propagation off |
| `K1/K1.json` | Neutral δ-kick. This root is the Gaussian wavelength |
| `K3/K3.json` | The same kick after the sudden hole |
| `G1`–`G4` | Hybrid cross section, ± DCH, ± nanoparticle, no CAP |

Read D3 − D1 and G4 − G2 against the same differences on the resonant pair. If the nanoparticle effect shrinks or disappears here, the resonant result was the shared resonance. If it does not, the sphere is doing something that does not need the plasmon on the dye.

K1 is required even when the molecule is the resonant one. It is the check that this nanoparticle's plasmon misses that root. There is no separate bare-sphere flux job. The plasmon position is part of the nanoparticle placeholder.
