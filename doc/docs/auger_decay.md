# Why this RT-TDDFT cannot describe Auger decay

Auger decay is a correlated two-electron transition. One electron falls into a core hole and a second electron leaves the atom, carrying the energy difference as kinetic energy. The propagation in PlasMol cannot do that transition, for a sodium K hole or for any other Auger channel.

The equation that is actually integrated is a one-body Kohn–Sham matrix built from the density at that instant, plus a dipole coupling to a classical field, plus an optional absorber that acts only on orbitals above the vacuum level. That Hamiltonian does not contain the two-electron continuum coupling. The Gaussian basis cannot hold a kilovolt electron. Without the absorber the electron count is conserved, so the ion Auger decay would leave behind is not in the space being propagated. The flat 1s trace is that fact showing up in the occupation file. It is not the atom refusing to relax.

The worked example is the KLL cascade of a sodium double K hole, which is the initial condition used for the resonant double-core-hole runs. [Core-Hole Dynamics](core_hole.md) describes how that initial condition is built. The same missing pieces, listed in the section on other Auger families, rule out Coster–Kronig decay, core–valence–valence decay, double Auger decay, radiative Auger decay, and interatomic Coulombic decay.

## What an Auger decay is

An Auger decay (Auger–Meitner decay) is a radiationless transition of an ion that already has an inner-shell hole. Two electrons take part. One electron from a higher shell drops into the hole. The energy released by that drop is given to a second electron, which is ejected into the continuum. No photon is required. The kinetic energy of the ejected electron is fixed, to within the width of the resonance, by the difference of total energies of the initial and final ions.

The name of a channel says which shells supply the hole and the two electrons. In the X-ray notation, K is \(n = 1\), L is \(n = 2\), and M is \(n = 3\). A KLL decay has the initial hole in the K shell, and both participating electrons come from the L shell. The L shell splits further: L\(_1\) is 2s, and L\(_{2,3}\) is 2p. KL\(_1\)L\(_1\), KL\(_1\)L\(_{2,3}\), and KL\(_{2,3}\)L\(_{2,3}\) are three different final hole patterns, with three different electron energies.

The rate is set by a Coulomb matrix element between two configurations, not by a one-body potential. The piece of the Hamiltonian that does the decay is a two-electron operator connecting bound orbitals to a continuum orbital,

\[
\hat H_{\mathrm{Auger}}
=
\sum_{ijm\mu\,\sigma\sigma'}
v_{ijm\mu}
\left(
\hat c_{i\sigma}^{\dagger}
\hat c_{j\sigma'}^{\dagger}
\hat c_{m\sigma'}
\hat c_{\mu\sigma}
+
\mathrm{h.c.}
\right),
\]

where \(v_{ijm\mu}\) is the two-electron Coulomb integral. For the KL\(_{2,3}\)L\(_{2,3}\) path of sodium, one index is the empty 1s orbital, two indices are 2p orbitals, and \(\mu\) is a continuum orbital at the decay energy. The golden-rule width is \(2\pi\) times the square of that integral, times the density of continuum states at that energy.

That structure is shared by every Auger process. Coster–Kronig and super-Coster–Kronig decay change which shells the two electrons come from. Core–valence–valence decay changes the energy. Double Auger decay adds a third electron. Interatomic Coulombic decay puts one of the two electrons on a neighboring atom. All of them are absent here for the same reason. The radiative branch, in which the core hole is filled and a photon is emitted instead, is also absent: the propagation has no quantized radiation field. For a single sodium K hole that branch is the small one. The measured K-shell fluorescence yield is \(0.0234 \pm 0.0015\), so the Auger yield is \(0.9766 \pm 0.0015\) [1]. Those numbers are the single-hole branching. A double K hole is not expected to reverse the preference.

## The sodium double hole and the KLL cascade

Neutral sodium is a doublet, \(1s^{2}\,2s^{2}\,2p^{6}\,3s^{1}\): eleven electrons, charge 0, spin 1. The production setup is unrestricted Kohn–Sham with LC-ωPBE, range-separation parameter \(\omega = 0.351340\), and Cartesian aug-cc-pVTZ (59 functions). The neutral energy is \(-162.206711\) Ha.

The job does not remove the core electrons by a second self-consistent calculation. `remove_core_electrons` takes the neutral orbitals and sets the occupation of molecular orbital 0 to zero on both spins (`{"0": 2}`). Charge is raised by 2 and the spin is left at 1, because both electrons came out of one spatial orbital. The density is rebuilt from the neutral orbitals with those occupations and is not reoptimized. The ion at \(t = 0\) is Na\(^{2+}\) in the configuration \(1s^{0}\,2s^{2}\,2p^{6}\,3s^{1}\): nine electrons, an empty K shell, and the valence electron still present. On the α spin the occupied orbitals are 2s, the three 2p orbitals, and 3s. On the β spin the 3s orbital was already empty in the neutral doublet, so the β occupations that remain are 2s and the three 2p orbitals.

A physical KLL decay of that ion is a cascade of two steps. There are two K holes, and each Auger electron fills only one of them. Each step raises the charge by 1. One representative path, in which every L electron is a 2p electron, is below. KL\(_1\)L\(_{2,3}\) and KL\(_1\)L\(_1\) change which L orbitals are emptied. They do not change the charge.

| Stage | Configuration | Electrons | Charge |
| --- | --- | ---: | ---: |
| Neutral atom | \(1s^{2}\,2s^{2}\,2p^{6}\,3s^{1}\) | 11 | 0 |
| Sudden double K hole (the initial state) | \(1s^{0}\,2s^{2}\,2p^{6}\,3s^{1}\) | 9 | +2 |
| After the first KLL step | \(1s^{1}\,2s^{2}\,2p^{4}\,3s^{1}\) | 8 | +3 |
| After the second KLL step | \(1s^{2}\,2s^{2}\,2p^{2}\,3s^{1}\) | 7 | +4 |

The first step fills one of the two 1s holes, creates two L holes, and emits one electron. The second step fills the remaining 1s hole, creates two further L holes, and emits a second electron. The final ion is Na\(^{4+}\) with a full K shell. The two Auger electrons do not have the same kinetic energy. The second leaves an ion that already carries the holes made by the first, which is why a double-K-hole Auger spectrum (a hypersatellite spectrum) is not a copy of the ordinary single-hole spectrum.

The lifetime of the first step is femtoseconds, not the length of these runs. The measured lifetime broadening of the neon K\(^{-2}\)V states is \(701 \pm 11\) meV, about 2.9 times the single-K width of 242 meV [2]. Neon is the neighboring atom, and the measured state still has a valence spectator, but that width was common to the whole Rydberg series. A width of 0.70 eV is a lifetime \(\hbar/\Gamma\) of about 0.94 fs, about 39 atomic units of time. Sodium is not expected to move a femtosecond lifetime out to the length of the propagation. The 10000 au run is 242 fs, several hundred such lifetimes. If the channel existed in the Hamiltonian, the double hole would already be gone, and the ion being propagated after that would be the Na\(^{3+}\) or Na\(^{4+}\) product.

## What the propagation advances

Each step builds one matrix per spin and multiplies the orbital coefficients by its exponential. In `get_F_orth` the matrix is PySCF’s Kohn–Sham Fock matrix of the density at that instant, plus the dipole coupling to the classical electric field, minus \(i\) times the absorber when a complex absorbing potential is on:

\[
F_{\sigma}(t)
=
F_{\mathrm{Hxc},\sigma}[D(t)]
-
\mathbf{r}\cdot\mathbf{E}(t)
-
i\Gamma_{\sigma}.
\]

\(F_{\mathrm{Hxc}}\) is the core Hamiltonian, the Hartree potential of \(D(t)\), the range-separated exact-exchange matrix, and the ωPBE correlation potential. Every term is fixed once the density at that instant is known. The exchange matrix does contain two-electron integrals, but only after they have been contracted with the current density. The contraction is a one-body operator. It is not the transition integral \(v_{ijm\mu}\) between the initial two-hole configuration and a one-hole-plus-continuum configuration.

The Magnus step then advances every orbital of a given spin with the same matrix,

\[
C(t+\Delta t) = \exp(-i\,\Delta t\, F_{\mathrm{mid}})\, C(t),
\]

and rebuilds the density from the new coefficients with the original occupation list. That list is stored when the hole is made (`self.occ = self.mf.mo_occ` after `remove_core_electrons`) and is passed to `make_rdm1` on every iteration. The time-dependent orbital that began as the 1s orbital carries occupation zero on both spins for the entire run. The density is always a nine-electron density assembled from the 2s, 2p, and 3s time-dependent orbitals.

A later projection of that density onto the frozen neutral 1s function can be nonzero only if one of those occupied orbitals rotates onto the neutral 1s. The logged hole occupation is that projection. For this open-shell ion it is the α+β sum on the neutral orbital basis,

\[
n_k(t)
=
\sum_{\sigma}
\left[C_{n,\sigma}^{\dagger} S\, D_{\sigma}(t)\, S\, C_{n,\sigma}\right]_{kk},
\qquad
h_k(t)
=
n_k^{(0)} - n_k(t),
\]

with the neutral coefficients \(C_n\) kept from before the hole was made. Positive \(h_k\) means electrons have left that neutral orbital. The neutral 1s orbital held two electrons, so \(h_0\) starts at 2. The logger writes orbitals 0 through 6.

Two consequences follow before any occupation file is read.

The equation of motion is one-body, \(i\partial_t C = F[D]\,C\). A one-body operator can rotate orbitals into each other. It cannot take two L electrons and put one in the 1s orbital and the other in a continuum orbital as one transition.

Without the absorber, \(F\) is Hermitian, the Magnus exponential is unitary, and the electron count stays at nine. The first KLL step leaves eight electrons and the second leaves seven. Those states are not in the space being propagated.

This is the adiabatic approximation as it is actually implemented. The exact time-dependent Kohn–Sham potential is a functional of the history of the density, and in principle it could be imagined to make the core density fall at an Auger rate, because exact time-dependent density-functional theory reproduces the exact density. That potential is not known, it would have to remember the past density, and even then the Kohn–Sham continuum orbital is not the physical Auger electron. The potential this code evaluates is adiabatic LC-ωPBE of the density at the current instant. Adiabatic potentials do not produce Auger decay. A real-time local-density propagation of an atomic Auger process shows the limitation directly [3]. A propagation that omits the Auger self-energy leaves the core population constant, because the decay sits in a two-electron self-energy built from \(v_{ijm\mu}\), not in the instantaneous Hartree–exchange–correlation potential [4].

## Why the KLL amplitude is zero

### The integral that sets the rate is never formed

For the first KL\(_{2,3}\)L\(_{2,3}\) step the amplitude is the antisymmetrized Coulomb integral between the initial configuration (empty 1s, full 2p) and the final configuration (one 1s electron, two 2p holes, one continuum electron). Building \(F_{\mathrm{Hxc}}\) contracts every two-electron integral against the density and uses the result as a potential. The commutator of that potential with the density moves charge in the one-particle picture. It never evaluates the bare integral as a coupling between those two configurations, and it never opens a continuum channel at the Auger energy.

Range-separated exact exchange does not close the gap. Exact exchange is nonlocal in space, but it is still one matrix of the size of the basis, rebuilt from the instantaneous orbitals, and it still multiplies every orbital the same way inside the exponential. Nonlocal in space is not a two-configuration transition amplitude.

### The classical field does not supply the missing operator

The only addition to \(F\) that is not already fixed by \(D(t)\) is the dipole term \(-\mathbf{r}\cdot\mathbf{E}(t)\). That term drives ordinary one-electron transitions. It is how a plasmon field acts on the atom. A dipole operator cannot fill a 1s hole and eject a second electron.

The finished hybrid runs agree. On the run with the sphere, back-propagation, and no absorber, the 1s hole occupation over 10007 au has mean 1.9975 and stays inside 1.9947 to 2.0000. The run with the sphere present and back-propagation off is the same. The free-atom run is the same. The sphere changes the late valence motion. It does not touch the K hole.

### A one-body rotation is a beat, not a decay

The sudden hole is not a stationary state of its own Fock operator. Removing the 1s density deepens the potential felt by the outer electrons, so the initial density is not an eigenstate and the outer orbitals move. That motion is real. It is the mean-field part of shake-up: the valence cloud rearranges because the screening changed. It is a rotation generated by \(F\). Population that leaves a neutral orbital by such a rotation comes back at the Bohr period of the two orbitals involved, unless something absorbs it.

A decay is a different time dependence. A resonance of width \(\Gamma\) loses population as \(e^{-\Gamma t/\hbar}\) and does not return it. Producing that factor requires an imaginary part of the core self-energy, which is the on-shell value of the Auger amplitude. The Fock matrix of an adiabatic functional is Hermitian apart from the absorber below. The 1s diagonal has no imaginary part. Population that wobbles off the 1s orbital wobbles back.

## The basis cannot represent the ejected electron

Suppose the two-electron operator were added anyway. The ejected electron would still have to be expanded in the one-particle basis. Cartesian aug-cc-pVTZ on sodium has 59 functions: atom-centered Gaussians chosen for the valence and near-valence region, not continuum radial waves.

The instantaneous Kohn–Sham eigenvalues of the sudden double-hole density put a number on the gap. They were recomputed with PySCF at the production functional, range-separation parameter, basis, and occupation constraint. They are not ΔSCF binding energies and they are not a predicted Auger line. They are the spectrum of the unrelaxed density the job starts from, which is also the spectrum the static absorber is built from. Hole–hole interaction and orbital relaxation move a real hypersatellite line by tens of eV. They do not move it from a kilovolt down to the top of this basis.

| Orbital | Neutral α (eV) | Sudden double hole, α (eV) | Absorber \(\gamma\) |
| --- | ---: | ---: | --- |
| 0, 1s | −1038.8 | −1328.6 | 0 |
| 1, 2s | −62.2 | −156.2 | 0 |
| 2–4, 2p | −34.0 | −128.6 | 0 |
| 5, 3s | −5.2 | −28.4 | 0 |
| 6–7, next | +0.3 | −17.0 | 0 |
| Highest of the 59 | +91.2 | +46.1 | nonzero only above +0.27 eV |

A Koopmans-style reading of the sudden-hole column only places the line. Dropping a 2p electron from −128.6 eV into the 1s orbital at −1328.6 eV releases about 1200 eV. The second 2p electron is bound by 129 eV in that same spectrum, which leaves a kinetic energy of about 1071 eV (39 Ha). The estimate uses one unrelaxed Fock matrix for both the initial ion and the final ion. The highest function in that same matrix is at +46 eV. The highest function in the neutral spectrum, before the hole pulls every eigenvalue down, is at +91 eV. The electron a KLL decay emits sits more than twenty times above either ceiling.

There is no grid, no complex-scaled continuum, no Coulomb-wave final state, and no Stieltjes imaging that would rebuild a kilovolt density of states from these Gaussians. Adding diffuse functions adds Rydberg orbitals near the vacuum level. It does not add a radial wave with about 39 Ha of kinetic energy.

## The absorber is a different process, and it is closed on the core

The complex absorbing potential is the one non-Hermitian term in the propagation, so it is the obvious place to look for a decay. It is not an Auger decay, and on this ion it does not act on the orbitals that would have to decay.

`get_gamma_ao` follows Lopata and Govind [5]. Their absorber was built to mimic coupling to the missing continuum for excitations near and above the ionization threshold, over about 0 to 50 eV, so that a valence absorption spectrum is not dominated by artifacts of a finite Gaussian basis. It is diagonal in a Kohn–Sham orbital basis. With the production parameters (static, \(\gamma_0 = 1\), \(\xi = 0.5\), \(\varepsilon_0 = 0.009767\) Ha, clamp 100) the strength on orbital \(i\) is

\[
\gamma_i =
\begin{cases}
0, & \varepsilon_i \le \varepsilon_0, \\[0.4em]
\min\left\{100,\; \exp\!\big[\xi(\varepsilon_i-\varepsilon_0)\big]-1\right\},
& \varepsilon_i > \varepsilon_0.
\end{cases}
\]

The cutoff \(\varepsilon_0\) is 0.266 eV. It was tuned as a vacuum level for the neutral valence spectrum. Every orbital in the table through the 3s, and the next orbital after it, is tens to more than a thousand eV below that cutoff, so each has \(\gamma = 0\) exactly. On the sudden-hole Fock matrix only 15 of the 59 α orbitals and 20 of the 59 β orbitals sit above the cutoff, and the highest of those is at 46 eV. The code subtracts \(i\Gamma\) from the Fock matrix (`F_ao - 1j * Gamma`), so those high orbitals decay. The 1s orbital does not.

The static option, used by the production jobs, builds \(\Gamma\) once from the \(t = 0\) double-hole density and freezes it. It does not turn on when a core hole is ready to decay, and it does not depend on which hole is present. The dynamic option rebuilds \(\Gamma\) from the orbitals of the current density and uses the same formula. A 1s eigenvalue near −1300 eV still has \(\gamma = 0\). Switching from static to dynamic does not open a KLL channel.

What the absorber removes is density that has already reached an orbital above \(\varepsilon_0\). That is a model of a one-hole process: an electron that has climbed into the discretized continuum is not allowed to reflect off the end of the basis. Autoionization of a valence resonance, which is what reference [5] addresses, is one electron leaving an orbital that is already above the ionization threshold. Auger decay is two electrons changing orbitals, one of them by a thousand eV. Damping the top of a 46 eV Gaussian spectrum does not turn one into the other.

The runs with the absorber show the distinction in the occupations. With the absorber on, out to 10007 au, the logged holes on orbitals 0 through 6 grow from 2.08 to 4.11. Electrons leave the plotted set, and because the absorber is on they can leave the atom. The 1s hole does not take part. Its mean is 1.9979 and its range is 1.9951 to 2.0000, the same 0.005-electron wobble as the runs with no absorber. The lost electrons came out of the L and valence part of the plotted window and out of the higher virtuals. They did not come out of a KLL event, because the 1s projection never gained an electron.

## Charge, spin, and the occupation list

A one-body rotation that moved a whole 2p electron onto the neutral 1s function would still not be a KLL decay. It would fill one core hole by emptying one L orbital and it would leave the electron count at nine. The ion would still be Na\(^{2+}\). The physical step emits a second electron and leaves Na\(^{3+}\), with two L holes rather than one. The propagation has no variable that counts continuum electrons, no partial-wave flux, and no energy-resolved outgoing wave.

The occupation list makes the bookkeeping explicit. `remove_core_electrons` zeros both spins of orbital 0. The Magnus step calls `make_rdm1` with that same list at every iteration. The time-dependent orbital labeled 0 remains empty by construction. Charge can appear on the neutral 1s shape only by orbital mixing, and the occupation traces show that this mixing stays at about 0.005 electrons. The time loop cannot promote one L electron into the core and send another into the continuum, because those are two occupation changes and the occupation list is not a dynamical variable.

Spin does not open a hidden channel. The parent is a doublet and the double hole on one spatial orbital keeps spin 1. The α and β Fock matrices differ, which is why the three 2p hole traces can split once the valence motion becomes large. They are still two one-body matrices. An Auger amplitude between a pair of L electrons and a continuum electron is an off-diagonal element of the two-electron operator, not a difference between the α and β potentials.

## What the 10000 au run shows

The field-free run with no absorber is the cleanest trace. It is the free Na\(^{2+}\) ion, time step 0.05 au, 200000 steps, from 0 to 10000 au (242 fs). The hole occupations are projections onto the neutral orbitals.

| Neutral orbital | \(t = 0\) | \(t = 41\) au | \(t = 10000\) au | Peak to peak |
| --- | ---: | ---: | ---: | ---: |
| 0, 1s | 1.997 | 2.000 | 1.996 | 0.005 |
| 1, 2s | 0.010 |  | 0.106 | 0.360 |
| 2, 2p | 0.004 |  | 0.267 | 0.401 |
| 3, 2p | 0.004 |  | 0.202 | 0.345 |
| 4, 2p | 0.004 |  | 0.212 | 0.345 |
| 5, 3s | 0.000 |  | 0.609 | 0.998 |
| 6, empty 3p | 0.000 |  | −0.008 | 0.037 |

At 41 au, about one neon double-K lifetime, the 1s hole is 1.9999. At the end of the run it is 1.9965. The mean over all 200000 samples is 1.9976. The entire motion of the core hole spans 0.0053 electrons. Its strongest period is 0.147 au, an energy of 1166 eV, which is three time steps. That is the scale of a core-to-outer-shell orbital difference. A coherent beat from a coupling already present in \(F\) looks like that. An irreversible decay of a 0.7 eV resonance does not. It would be a smooth drop on a scale of about 40 au, with no return.

The other columns point the wrong way for a core being filled. The 2s hole and the three 2p holes grow. Electrons leave those orbitals. They do not fall from them into the 1s orbital. The 3s hole swings between 0 and 1 for the whole run, so the valence electron leaves the 3s orbital and comes back, rather than dropping into the core. The one plotted orbital that gains density is the empty 3p orbital, by at most 0.037 electrons. The holes on orbitals 0 through 6 sum to 2.019 at the start and 3.384 at the end. About 1.4 additional electrons have left the plotted window. This run has no absorber, so they have not left the atom. They sit in neutral orbitals 7 through 58. That is valence and L density moving into the diffuse virtual space under the deepened potential: the shake-up the sudden approximation produces. It is not a core being refilled.

The same 1s numbers appear with the absorber on and with the sphere present. Whatever is complicated about the late dipole of the runs that have no absorber, it is not an Auger cascade. The core hole those runs carry at 10000 au is the core hole they were given at \(t = 0\).

## The same wall stops every Auger process

KLL is the deepest case, because the energy that has to leave is a kilovolt and the core orbital is the one the absorber is guaranteed to ignore. The reason it is closed does not use the fact that the hole is a K hole. Any Auger process is a transition driven by the uncontracted two-electron integral above, with at least one outgoing continuum electron. This propagation has the contracted, instantaneous potential instead, and it has no continuum orbital at the relevant energy. Replacing 1s by 2s, or 2p by 3s, does not add that integral.

| Process | What has to happen | Why it does not happen here |
| --- | --- | --- |
| Diagram Auger (KLL, KLM, LMM) | One electron fills a deeper hole. A second electron from a higher shell leaves with the difference. | Needs \(v_{ijm\mu}\) between the two configurations, and a continuum orbital at that kinetic energy. \(F\) has neither. |
| Coster–Kronig | The electron that fills the hole comes from the same shell. The ejected electron comes from a higher shell. | Same operator as diagram Auger. A smaller energy does not create the operator. In the table above, the 2s and 2p holes both grow. Neither fills the other. |
| Super-Coster–Kronig | Both participating electrons come from the shell that has the hole. | Same two-electron continuum integral. The occupation list cannot move two electrons out of one shell and one electron back in. |
| Core–valence–valence (KVV, LVV) | A core or L hole is filled from the valence, and a second valence electron leaves. | The 3s hole grows toward 1 instead of dropping into the core. The valence electron is shaking into virtuals. |
| Double Auger | One electron fills the hole and two electrons leave. | A three-electron continuum process. One step further from a one-body \(F\). |
| Radiative Auger | One electron fills the hole, one leaves, and a photon is emitted. | Needs both the two-electron continuum coupling and a quantized radiation field. The classical \(\mathbf{E}(t)\) is an external field, not a spontaneous photon. |
| Interatomic Coulombic decay, electron-transfer-mediated decay | The energy, or one of the electrons, goes to a neighbor, which then emits. | Needs a second quantum system with its own electrons. The gold sphere is a classical dielectric. It has no electrons to ionize. |

Two near relatives are not counterexamples.

Shake-up and shake-off at the moment the hole is created are not Auger decay. The sudden removal of a core electron changes the potential, and outer electrons are promoted or ejected by that change. The non-stationary density at \(t = 0\) is that sudden approximation, and the one-body rearrangement that follows is the part of it this method can describe. Shake-up does not fill the hole. The hole was the cause.

Autoionization of a resonance that already lies above the vacuum level is not Auger decay either. One electron leaves. The absorber is a phenomenological model of that loss, restricted to orbitals above \(\varepsilon_0\) and to energies this Gaussian basis can roughly represent, which here means below about 46 eV. It was not a model of a two-electron core decay.

A plasmon-assisted Auger decay is out of reach for a more specific reason. Assistance would mean the classical field modulating an Auger amplitude, or opening a sideband of one. There is no Auger amplitude to modulate. Coupling the atom to the sphere adds \(-\mathbf{r}\cdot\mathbf{E}(t)\) and, when back-propagation is on, a classical polarization radiated by the dipole. Neither object is \(v_{ijm\mu}\).

## An imposed 1s lifetime would still not be Auger decay

The absorber formula cannot be retuned onto the core. \(\gamma\) is zero at and below \(\varepsilon_0\) by construction, and \(\varepsilon_0\) sits at the vacuum level so that the valence ionization continuum is what gets absorbed. Lowering \(\varepsilon_0\) until it passed −1329 eV would damp every orbital in the atom, including 2s, 2p, and 3s, and the damping would still be a one-body imaginary potential. It would delete amplitude. It would not transfer it.

Adding an imaginary part by hand to the 1s diagonal, \(-i\gamma_{1s}\) with \(\gamma_{1s}\) chosen as \(\hbar\) over a femtosecond, would make the projected 1s population fall. That fall would not be a KLL decay.

1. The electron removed from the 1s orbital disappears from the norm. It is not placed in a continuum orbital, and it has no kinetic energy.
2. No second electron is ejected, so the partner L holes are not created. The final configuration in the cascade table never appears.
3. The branching among KL\(_1\)L\(_1\), KL\(_1\)L\(_{2,3}\), and KL\(_{2,3}\)L\(_{2,3}\) is absent. One number \(\gamma\) cannot choose a final state.
4. The rate is an input. The calculation cannot predict it, and it cannot respond to the hole–hole shift that separates a hypersatellite electron from a diagram electron.
5. The cascade still cannot happen. After a prescribed drain had emptied the 1s label, the ion would not be Na\(^{3+}\) with a remaining 1s hole and two L holes. It would be the original nine-electron density with some amplitude missing.

Letting \(\gamma_{1s}(t)\) depend on the classical field repairs none of those five. The field would be scaling a prescribed drain. The Auger electron, the final ion, and the line positions would still be missing. Forcing the 1s trace to fall would hide the fact that the channel is absent.

## What a calculation of this decay needs

A description of the sodium double-hole KLL cascade has to contain the configurations in the cascade table as distinct states, coupled by the Coulomb operator, with a continuum normalized at the Auger energy.

1. The initial state is the two-hole state \(1s^{0}\), not a Slater determinant that has only been given a new one-body potential. The final state of each step is a specific L-hole pattern plus a continuum function at a definite kinetic energy.
2. The coupling is the antisymmetrized integral \(v_{ijm\mu}\), evaluated between those configurations. Algebraic diagrammatic construction, a Fano treatment, configuration interaction in the continuum, or a complex-scaled calculation on a correlated Hamiltonian are the usual ways to turn that integral into a width and a line position. An absorber, if one is used, has to act on the correlated continuum, not on the occupied Kohn–Sham spectrum of an adiabatic Fock matrix.
3. The continuum has to exist near a kilovolt. That means Coulomb waves, a grid, a B-spline or finite-element basis, or an explicit Stieltjes reconstruction. Enlarging aug-cc-pVTZ does not provide it.
4. Both steps of the cascade have to be present if the claim is about the double hole. The first electron is a hypersatellite electron. The second is the decay of the single-hole ion the first step left behind, and that ion is a different charge state with its own spectrum.
5. If a plasmon is to assist the decay, it has to act on that amplitude. A classical field added to a one-body potential, which is what the hybrid step does now, has nothing to assist.

None of those ingredients is a small change to `get_F_orth` or to the Magnus exponential. They replace the equation being integrated.

## What the present runs do describe

The limitation is specific. These propagations do describe the one-body response of the nine-electron ion to the sudden change in screening and, when the sphere is present, the additional one-body driving by the classical local field. The growth of the 2s, 2p, and 3s hole traces, the electrons that move into orbitals 7 through 58, and the dipole that develops on the runs without an absorber are that response. They are the motion of Na\(^{2+}\) with a K shell that cannot refill.

They stop being a description of the physical ion once the clock passes the Auger lifetime. A real double K hole in an atom next to sodium lives about a femtosecond. By 400 au, and certainly by 10000 au, that atom would already have emitted its Auger electrons and become a more highly charged ion with L holes. Reading the late dipole as a core-hole lifetime, or reading the flat 1s curve as evidence that the core does not relax, confuses a channel the Hamiltonian does not contain with a channel the atom does not use. The atom uses KLL. This propagation cannot.

## References

1. M.-L. Venzke, K. Frenzel, and P. Hönicke, “Experimental determination of the sodium K-shell atomic fundamental parameters for X-ray spectroscopy,” arXiv:2506.14352 (2025). Single-K fluorescence yield \(0.0234 \pm 0.0015\) and Auger yield \(0.9766 \pm 0.0015\).
2. G. Goldsztejn et al., “Double-Core-Hole States in Neon: Lifetime, Post-Collision Interaction, and Spectral Assignment,” *Phys. Rev. Lett.* **117**, 133001 (2016).
3. C. S. Cucinotta, D. Hughes, and P. Ballone, “Real-time real-space TD-DFT for atoms: Benchmark computations on a nonspherical logarithmic grid,” *Phys. Rev. B* **86**, 045114 (2012).
4. F. Covito, E. Perfetto, A. Rubio, and G. Stefanucci, “Real-time dynamics of Auger wave packets and decays in ultrafast charge migration processes,” *Phys. Rev. A* **97**, 061401(R) (2018).
5. K. Lopata and N. Govind, “Near and Above Ionization Electronic Excitations with Non-Hermitian Real-Time Time-Dependent Density Functional Theory,” *J. Chem. Theory Comput.* **9**, 4939 (2013). The absorber used in this code, tested on spectra from 0 to 50 eV.

The Fock matrix, the absorber, and the sudden hole are `get_F_orth`, `get_gamma_ao`, and `remove_core_electrons` in `plasmol/quantum/molecule.py`. The time step is the Magnus exponential in `plasmol/quantum/propagators/magnus2.py`. The neutral eigenvalues and the sudden-hole eigenvalues were recomputed at \(\omega = 0.351340\), Cartesian aug-cc-pVTZ, charge 0, spin 1, and the same occupation constraint. The neutral energy matches the production value. The occupation numbers are from the finished field-free 10000 au run and from the hybrid runs with and without the absorber.

## See also

- [Core-Hole Dynamics](core_hole.md) — sudden SCH/DCH initial conditions and hole-occupation logging
- [Theory & Methodology](methodology.md) — the hybrid loop and the Lopata absorber
