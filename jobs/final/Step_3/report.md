# Step 3: field-free double core hole on sodium

The sudden double K hole does not decay. On every finished run the 1s hole stays inside 1.9947–2.0000 for the whole propagation. What does change is the valence, and it changes in two different ways.

With no complex absorbing potential the atom develops a dipole of order 1 after a few thousand atomic units, the three 2p holes split, and electrons move out of MOs 0–6 and come back. That happens with no sphere at all. The density dumps show that this dipole is a parity-breaking instability of the propagation, seeded by roundoff and growing with a 131 au e-fold, and that locking the orbital parity removes it without changing the relaxation. Putting the gold sphere in the cell makes the same dipole appear about 1000 au earlier, including on the run whose local field is exactly zero. With the absorber on, the dipole stays near 0.002, the 2p holes stay equal, and the electrons that leave MOs 0–6 do not come back. Turning the molecule’s radiation on or off does not change that absorber run.

These jobs have no incident wave. D3–D6 are not absorption cross sections.

![Dipole envelope, mean hole sum, and 2p splitting for the five runs that reach 10000 au.](comparison.png)

The curves are 20 au windows: the maximum dipole, the mean hole sum, and the maximum spread of the three 2p holes. D5 and D6 lie on top of each other. The absorber runs are the flat dipole and the climbing hole sum.

## What was compared

There is no external electric field. The only drive is the sudden hole, plus whatever classical field that hole’s dipole later induces on the sphere. The atom is the production sodium doublet: charge 0, spin 1, LC-ωPBE with ω = 0.351340, Cartesian aug-cc-pVTZ. `remove_core_electrons` sets both spins of MO 0 to empty (`{"0": 2}`), raises the charge by 2, and does not reoptimize. The ion at the start of the propagation is Na²⁺ with nine electrons, an empty K shell, and the 3s electron still there.

D1 and D2 are molecule-only (`quantum`), with a time step of 0.05 au, and they stop at 400 au (9.7 fs). D1₁₀ₖ is the same D1 input continued to 10000 au (242 fs). It does not replace the 400 au run. D3–D6 are hybrid (`plasmol`) with no Meep source. Meep’s step is 0.100073 au, so 100000 steps land at 10007.26 au rather than 10000. The sphere is gold, radius 0.025 μm, in water (n = 1.33). The nucleus is placed at x = 0.02645 μm, one Yee pixel off the surface. D3–D6 also carry the orientation taken from D1’s dipole. That dipole is below 2×10⁻¹⁰, so the axis is a noise direction. The nucleus sits at the origin of `Na.xyz` and the rotation does not move it.

| Run | Absorber | Sphere | Molecule radiates | Driver | Time step | Length |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | off | no | — | `quantum` | 0.05 | 400 au |
| D2 | on | no | — | `quantum` | 0.05 | 400 au |
| D3 | off | yes | yes | `plasmol` | 0.100073 | 10007 au |
| D4 | off | yes | no | `plasmol` | 0.100073 | 10007 au |
| D5 | on | yes | yes | `plasmol` | 0.100073 | 10007 au |
| D6 | on | yes | no | `plasmol` | 0.100073 | 10007 au |

The absorber is the static Lopata form used everywhere else in this campaign: γ₀ = 1, ξ = 0.5, ε₀ = 0.009767 Ha, clamp 100. The requested Magnus tolerance is still 1×10⁻¹².

The plan reads D4 against D1, and D3 against D4. If the sphere mattered only by returning the atom’s own radiation, D4 would follow the free atom and D3 would not. That is not what the long runs do.

## What is plotted

`mo_occ.csv` is a hole occupation on the frozen neutral orbitals, α+β, so a full spatial orbital contributes 2. The hole on orbital k is the neutral occupation minus the projection of the density at that time. Positive means electrons have left that orbital. The file covers MOs 0 through 6 and nothing higher.

MO 0 is 1s, MO 1 is 2s, MOs 2–4 are the three 2p orbitals, and MO 5 is 3s. MO 6 is the first virtual. Its hole stays near zero, slightly negative (at most about −0.04), so the electrons that leave the plotted window are not sitting on that orbital. A rise in the sum over MOs 0–6 is population that has gone into MOs 7 and above, or that the absorber has removed. The two cannot be told apart from this file alone. They can be told apart by whether the sum comes back down, because without the absorber the electron count is conserved.

`field_p` is the molecular dipole. `field_e` is the classical field at the molecule. Both are in atomic units.

## The 1s hole

| Run | Minimum | Maximum | Mean | Peak to peak |
| --- | ---: | ---: | ---: | ---: |
| D1, through 400 au | 1.9948 | 2.0000 | 1.9976 | 0.0052 |
| D1₁₀ₖ | 1.9947 | 2.0000 | 1.9976 | 0.0053 |
| D3 | 1.9947 | 2.0000 | 1.9975 | 0.0053 |
| D4 | 1.9947 | 2.0000 | 1.9975 | 0.0053 |
| D5 | 1.9951 | 2.0000 | 1.9979 | 0.0049 |
| D6 | 1.9951 | 2.0000 | 1.9979 | 0.0049 |

The row labelled t = 0 already has a 1s hole of 1.9966 (D1) or 1.9967 (the hybrid runs). That row is the state after the first step, not the sudden hole; the files in this directory stamp each row one step early (see the end of the dipole section). The sudden hole itself is exactly 2. At 41 au, about one neon double-K lifetime, D1₁₀ₖ still has 1.9995. At 10000 au it has 1.9965.

The motion that is there is a beat, not a loss. Over the first 40 au of D1₁₀ₖ the strongest period in the 1s hole is 0.1465 au. That period is an energy of 2π / 0.1465 = 42.9 Ha, about 1170 eV, which is the gap from the core down to the valence. It is three steps of the 0.05 au grid. A decay of a resonance a few tenths of an eV wide would be a smooth drop on a scale of tens of atomic units, and it would not return. Nothing in this table drops.

D2, which is the free atom with the absorber and only 400 au of propagation, has the same 1s range: 1.9950 to 2.0000.

## Without the absorber the valence runs away

D1 through 400 au is quiet in the dipole and already busy in the holes. The dipole never exceeds 1.9×10⁻¹⁰, and the field is zero because there is no classical cell. The three 2p holes stay equal. The 3s hole swings from 0.0004 up to 0.996. The sum of holes on MOs 0–6 starts at 2.019, which is the double hole, and its mean over 200–400 au is already 3.21. The sum is oscillating. It is not a slow leak.

D1₁₀ₖ is that same trajectory. Through 400 au the holes match D1 to 3×10⁻¹² and the dipole stays under 1×10⁻¹⁰. After that the free atom does what the sphere runs do. The dipole passes 0.1 at 3140 au and reaches 0.859 at 7972 au. There is still no field and no sphere. The 2p holes, equal out to 400 au, first differ by more than 0.05 at 3540 au and are split by as much as 0.224 near 5000 au. The mean hole sum over the last 200 au is 3.215, the same value it had at 200–400 au. After 1000 au the instantaneous sum still falls as low as 2.226 and rises as high as 4.113. Electrons leave MOs 0–6 and return. The time average does not drift.

D4 has the sphere and does not let the molecule source the cell. The local field is exactly zero for all 10007 au, so the sphere is not driving the atom. The dipole still passes 0.1 at 2052 au and peaks at 0.911 at 2850 au. The 2p holes split by more than 0.05 at 2203 au. The mean hole sum over the last 200 au is 3.205. The sphere was not required for this motion, and on this run it supplied no field. It did move the onset earlier by about 1100 au relative to the free atom on the finer time step.

D3 lets the molecule radiate. Through 400 au it matches D4 to 7×10⁻¹² in every plotted hole, and the dipole on both is still below 3×10⁻¹⁰. The local field later peaks at 1.1×10⁻⁵, near 2931 au. That is the radiated field coming back, and it is small next to a dipole of 0.8. The dipole passes 0.1 at 2062 au, almost with D4, and peaks at 0.806 at 7232 au. Once the dipole is large, D3 and D4 stop being the same trajectory: the 3s hole differs by as much as 0.95, and the dipole traces differ by as much as 0.67. The 2p splitting and the flat mean hole sum are shared with D4 and with the free atom. The late disagreement is the valence motion after it has already become large, not a separate plasmon-driven channel.

D3 and D4 do not reproduce D1’s valence holes inside the first 400 au, even though all three dipoles are numerical noise there. The 2s hole differs from D1 by up to 0.23. The time steps are different (0.05 au against the Meep step), and the hybrid driver is not the quantum driver. The free-atom control for D3 and D4 is D1₁₀ₖ, not the 400 au file.

## Where the dipole comes from

D1₁₀ₖ_dmo is the D1₁₀ₖ input with the full density in the neutral MO basis written every 10 steps (`d_mo.bin`, 20000 frames of two 59×59 blocks). D1₁₀ₖ_dmo_even is the same input with `zero_odd_blocks` on, so every occupied orbital is stripped of its minority-parity atomic functions before each Fock build. The frames were split into two off-diagonal blocks by the parity of the neutral orbitals: same-parity pairs (s with s, s with d, p with p) and opposite-parity pairs (s with p, p with d). On one center the dipole operator connects only opposite-parity orbitals, so the second block is the dipole.

The two blocks do different things.

![Frobenius norm of the same-parity and opposite-parity off-diagonal blocks of the alpha density, with the dipole and the parity-locked run.](dmo_parity_blocks.png)

The same-parity block is at 1.0 by the first atomic unit and stays between 0.8 and 1.3 for the whole run, with no trend. It is the relaxation of the valence in the field of the double hole: the neutral 2s, 2p, and 3s are not eigenfunctions of the ion's Fock operator, so each propagated orbital becomes a superposition of neutral orbitals of the same parity, mostly 3s with the compact even virtuals 22, 38, and 54, and 2s with virtual 58. Its size is what a single determinant allows. For an idempotent density the off-diagonal weight in row k is n_k(1 − n_k), so an orbital whose occupation swings from 1 to 0, as the 3s does, must carry coherences up to 0.5. The largest single element in the file is 0.39.

The opposite-parity block starts at a floor of 3×10⁻¹⁰ and grows exponentially from about 600 au with an e-folding time of 131 au, reaches 0.1 at 3000 au, and saturates near 0.5. The 10 au window maximum of |p| follows it point for point on a log scale from 10⁻⁹ up to 1. The 2p splitting is the same thing on the diagonal, and it is second order: the spread of the three 2p occupations tracks the square of the 2p–3s coherence with slope 2 on log axes once it clears its 10⁻¹² floor.

| time (au) | opposite-parity norm | 10 au max of \|p\| |
| --- | ---: | ---: |
| 500 | 1×10⁻⁹ | 1×10⁻⁹ |
| 1000 | 1.2×10⁻⁷ | 1×10⁻⁷ |
| 2000 | 2.8×10⁻⁴ | 3×10⁻⁴ |
| 3000 | 0.14 | 0.19 |
| 4000 and later | 0.4 to 0.5 | 0.8 to 1.1 |

The seed is roundoff, not the initial state. The neutral SCF orbitals are parity-pure to 10⁻¹³ in amplitude and the even–odd block of the Fock matrix is at 4×10⁻¹². The 3×10⁻¹⁰ floor in both files is the parity impurity of the neutral virtuals used for the projection, chiefly MO 9, the 4s that sits 0.003 Ha from the 3p. Inversion is an exact symmetry of the grid, the basis, and the sudden hole, so an exact propagation would hold the block at zero forever. The nonlinear mean-field equations make the even-parity trajectory linearly unstable against an odd perturbation, and the roundoff is amplified until the dipole mode saturates. From 10⁻¹⁰ with a 131 au e-fold the onset lands near 3000 au, which is what D1₁₀ₖ shows. The hybrid runs started from a different seed on the Meep step and broke about 1000 au earlier.

The locked run confirms the reading. In D1₁₀ₖ_dmo_even the opposite-parity block stays at the 3×10⁻¹⁰ floor for all 10000 au, the same-parity block matches D1₁₀ₖ_dmo frame by frame to 10⁻⁹ through 1000 au and to 10⁻⁵ through 2000 au, and the hole columns of the two runs agree to 2×10⁻⁵ through 2500 au. They separate only once the odd block of the unlocked run is large. The relaxation physics is in the same-parity block of both runs. The large dipole, the 2p splitting, and the late disagreement between D3 and D4 are in the odd block of the unlocked ones.

![Single odd-parity elements of the alpha density: growth on a log scale, the 2p spread against the 2p–3s coherence, and 40 au of one element.](dmo_odd_elements.png)

Single elements show the same growth. Each odd-parity entry multiplies by about 1000 per 1000 au, the same rate for 2p–3s, 2p–virtual, and 2s–3p, and the ratio between the three 2p orbitals stays at 1 : 3 : 4.5 through the whole rise. That is one unstable mode, a fixed odd-parity direction in orbital space, which becomes the dipole axis. Under the growing envelope each element swings sign with a period of 1.6 to 2 au, the valence gap of the doubly ionized atom rather than the neutral 2p–3s gap of 5.9 au. The exponential turns over when the occupations it couples start to move: by 2800 au the 3s occupation has been pulled from 0.9 to 0.43, and the 2p–3s coherence settles at 0.1 to 0.16.

| window (au) | (2p_c, 3s) | (2p_c, v38) | (2p_a, 3s) | (2p_b, 3s) | (2s, 3p) | 2p spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–250 | 9×10⁻¹² | 3×10⁻¹¹ | 3×10⁻¹² | 6×10⁻¹² | 5×10⁻¹¹ | 1×10⁻¹² |
| 1500–1750 | 8×10⁻⁶ | 9×10⁻⁶ | 2×10⁻⁶ | 5×10⁻⁶ | 4×10⁻⁶ | 2×10⁻⁹ |
| 2000–2250 | 2×10⁻⁴ | 3×10⁻⁴ | 4×10⁻⁵ | 1×10⁻⁴ | 1×10⁻⁴ | 4×10⁻⁷ |
| 2500–2750 | 7×10⁻³ | 9×10⁻³ | 1×10⁻³ | 4×10⁻³ | 4×10⁻³ | 2×10⁻⁴ |
| 3000–3250 | 0.09 | 0.09 | 0.02 | 0.06 | 0.04 | 0.04 |
| 3500–3750 | 0.16 | 0.14 | 0.03 | 0.08 | 0.07 | 0.09 |
| 8000–8250 | 0.07 | 0.05 | 0.01 | 0.10 | 0.04 | 0.09 |

The entries are window maxima of |D_kl| in the alpha block. 2p_c is the neutral 2p that ends up most along the dipole axis in the basis the SCF chose for the degenerate shell. v38 is the even virtual the relaxed 3s mixes with most.

So the dipole of order 1 in D1₁₀ₖ, D3, and D4 is not a response of the atom to anything. It is a numerical symmetry breaking of the mean-field propagation, seeded at 10⁻¹⁰ and amplified at a fixed rate. The parity lock removes it without touching the relaxation. D2₁₀ₖ_dmo_even is the absorber input with the same dump and lock and has not been analysed here.

The density files and the hole CSVs of these runs were written with the time stamp one step early: the row labelled t is the state at t + 0.05 au. The frame labelled 0 in `d_mo.bin` is therefore not diagonal. That stamping is fixed in the code; the files here were not rewritten.

## With the absorber the valence does not come back

D5 and D6 agree. The largest hole difference on MOs 0–6 over the whole 10007 au is 1.9×10⁻⁵. The dipoles differ by at most 4×10⁻⁴. D6’s local field is exactly zero. D5’s peaks at 3.9×10⁻⁸. Radiation into the cell does not matter once the absorber is on, because there is never a dipole large enough to make a field.

The dipole peaks at 0.00204 (D5, 6208 au) and 0.00224 (D6, 6184 au). It never reaches 0.1. The three 2p holes stay equal: the spread never exceeds 1×10⁻⁵. At the last sample each 2p hole is 0.153, the 2s hole is 0.672, and the 3s hole is 0.978. The sum is 4.107.

The mean hole sum distinguishes this pair from the runs above. Over 200–400 au it is already 3.37, against 3.21 with no absorber. Over the last 200 au it is 4.105. After 1000 au the instantaneous sum never falls below 3.061, whereas the no-absorber runs still return to about 2.2. The absorber is removing amplitude that has reached orbitals above the cutoff, so the valence hole does not refill. MO 6 is not that reservoir. The missing population is above the plotted window, and it is not coming back.

D2 is the free atom with the same absorber, but only to 400 au. On that window the dipole is still 4×10⁻¹¹. The 1s hole matches D1. The valence holes do not: the 3s hole differs from D1 by up to 0.94 and the 2s and 2p holes by up to 0.3. The absorber is already changing the beats inside 400 au. D2 was not extended, so it does not test the late dipole. D5 and D6 are the long absorber runs.

## What this does to the comparison that was planned

D4 was supposed to be the free atom plus a silent sphere. It is not. Its dipole and its 2p splitting follow the free atom’s shape and start earlier, with a local field of zero. D3 was supposed to be the run that showed the radiated field. Its field does turn on, at 10⁻⁵, and after a few thousand atomic units its valence holes leave D4’s. The same departure from a quiet dipole happens with no sphere in the cell. The radiated field is not what starts the motion.

The absorber comparison is clean. D5 against D3, and D6 against D4, turn off the large dipole and the 2p splitting and stop the hole sum from returning. D5 against D6 does not. Back-propagation is not a second control once the absorber is on.

None of this is a KLL decay. A KLL step would fill one 1s hole, empty two L orbitals, and emit an electron. The 1s hole does not fill. The 2p holes on the absorber runs stay near 0.15 rather than rising to 1. The propagation has nine electrons when the absorber is off, so it cannot become the eight-electron and then seven-electron ions of the cascade. The reason that channel is absent from the Hamiltonian is written up separately in the documentation (`doc/docs/auger_decay.md`). Step 3 is the numerical statement of the same fact: ten thousand atomic units, several hundred core-hole lifetimes, and the 1s hole at the end is the 1s hole at the start.

## Pairwise figures

`comparison.png` puts five runs on one time axis. The files in `comparisons/` are one contrast each, so the pairs can be read separately. Each figure is orbital by orbital: 1s on a zoomed axis, then 2s, the three 2p orbitals, 3s, and MO 6 on a zoomed axis. The 2s, 2p, and 3s rows share the scale from −0.08 to 1.08. Where the two runs are on the same time grid, the right-hand column is the sample-by-sample hole difference. A difference smaller than 5×10⁻⁴ is drawn on a ±0.01 scale, so a null result stays flat. Every field panel is |E| × 10⁻⁵. Dipoles that stay below 10⁻² are on an expanded axis, and that is written on the panel. Lines are solid.

D1 against D1₁₀ₖ is not plotted. Through 400 au the holes agree to 3×10⁻¹².

D1 against D2 is the absorber on the free atom, inside 400 au. The 3s difference reaches 0.94. The dipole on both is still numerical noise.

![D1 vs D2](comparisons/D1_vs_D2.png)

D1 against D3, and D1 against D4, over the first 400 au. The beats look alike. The clocks differ, so there is no difference column. A 5 au envelope of the 3s hole disagrees by about 0.10; the larger pointwise numbers in the text above are a phase slip between the 0.05 au step and the Meep step.

![D1 vs D3, first 400 au](comparisons/D1_vs_D3_to_400au.png)

![D1 vs D4, first 400 au](comparisons/D1_vs_D4_to_400au.png)

D2 against the first 400 au of D5. The absorber runs with and without the sphere. The beats match at the same level as D1 against D3. The dipole is still noise.

![D2 vs D5, first 400 au](comparisons/D2_vs_D5_to_400au.png)

D1₁₀ₖ against D4, then against D3, over the full propagation. The dipole of the sphere runs turns on near 2000 au. The free atom turns on near 3100 au. D4’s field panel is zero. D3’s field reaches about 1.1×10⁻⁵. The 2p colors split on all three runs.

![D1_10k vs D4](comparisons/D1_10k_vs_D4.png)

![D1_10k vs D3](comparisons/D1_10k_vs_D3.png)

D3 against D4. The difference column stays flat until about 2000 au and then opens, mostly in the 3s hole. The field is present only on D3.

![D3 vs D4](comparisons/D3_vs_D4.png)

D3 against D5, and D4 against D6. Turning the absorber on removes the large dipole and the 2p splitting. The 2s and 3s holes stop oscillating and climb.

![D3 vs D5](comparisons/D3_vs_D5.png)

![D4 vs D6](comparisons/D4_vs_D6.png)

D5 against D6. The difference column is empty at the ±0.01 scale. The dipole that remains is about 0.002 on the expanded axis, and it is the same whether or not the molecule radiates.

![D5 vs D6](comparisons/D5_vs_D6.png)

## The finished files

The tables above are the finished logs. Those logs contain no `ERROR` line.

D6 died twice on the way there, at 2730 au. The first residual stuck at 1.3×10⁻⁹. The second, after the stall floor had been raised, stuck at 9.4×10⁻⁷ and was correctly rejected. Those partial traces are `D6/partial_t2730/` and `D6/partial_t2730_rerun/`. The finished D6 log never took the half-step retry. D3’s first attempt stopped at 3486 au with a residual of 8.3×10⁻¹¹; that partial is `D3/partial_t3486/`. The finished D3 log is the one plotted.

Each directory has `mo_occ.csv`, `mo_occ.png`, `field_e_*.csv`, `field_p_*.csv`, and `fields_*.png`. D1₁₀ₖ_dmo, D1₁₀ₖ_dmo_even, and D2₁₀ₖ_dmo_even also have `d_mo.bin`, 2.2 GB each, readable with `plasmol.utils.d_mo.load_d_mo`. The parity-block figures are `dmo_parity_blocks.png` and `dmo_odd_elements.png`. D1₁₀ₖ, D3, D4, D5, and D6 also have `mo_occ_panels.png`, one orbital per row. `comparison.png` is the five long runs on one time axis. The one-contrast figures are in `comparisons/`. D1 and D2 can checkpoint. D3–D6 cannot, because a hybrid run with no source has nothing to restart from.

D3–D6 use the same panel layout as D1₁₀ₖ. The left column is the minimum, mean, and maximum inside each 40 au bin. The middle column is eight cycles of the strongest period, ending at 400 au. The right column is a Welch spectrum after that 40 au average is removed. The Meep step is 0.10 au, so the 0.15 au core beat is not resolved. The line near 550 eV, period 0.31 au, is that beat aliased, not a second core transition.

![D3 orbital panels](D3/mo_occ_panels.png)

![D4 orbital panels](D4/mo_occ_panels.png)

![D5 orbital panels](D5/mo_occ_panels.png)

![D6 orbital panels](D6/mo_occ_panels.png)
