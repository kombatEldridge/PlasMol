# Molecules in the double-core-hole literature

This step has two parts. The note below is the literature. The folders under `Molecules/` (`n2/`, `co/`, `para_aminophenol/`, …, and `transthioindigo/`) are a valence-absorption screen: which of these molecules, if any, has a bright excitation near a metallic-nanoparticle plasmon. trans-Thioindigo is in that screen and is not in the papers below. There is no production PlasMol JSON. The μ-tune inputs are under `LC-wPBE mu/`, including `transthioindigo/tune.json`, which has not been run.

Starting coordinates are NCI CACTUS 3D models (`cactus.sdf`, `start.xyz`, `source.txt`). PubChem returned HTTP 500 on 2026-09-23. NWChem then optimizes each neutral molecule at PBE0/6-311G* (Cartesian), the same geometry level as the candidate trans-thioindigo structure, and computes linear-response TDDFT at LC-ωPBE, μ = 0.34272, the value used for this screen. Later steps do not inherit that μ. O₂ is the triplet. Everything else is a closed-shell singlet. Roots are in `absorption.txt`. The comparison with the silver and gold plasmons this screen was built against is the last section.

This step is a reading list plus that screen, not a PlasMol production run. There is no production JSON. The molecule and the nanoparticle for the later steps are not chosen. Thioindigo and indigo do not appear in the double-core-hole papers below. A 2025 study of hemithioindigo is single-hole XPS and NEXAFS, not a double core hole (DOI: [10.1063/5.0271164](https://doi.org/10.1063/5.0271164)).

A double core hole is two vacancies in core orbitals. Single-site means both vacancies on one atom (K⁻², or L⁻² for a 2p shell). Two-site means one vacancy on each of two atoms (K⁻¹K⁻¹). Cederbaum, Tarantelli, Sgamellotti, and Schirmer showed in 1986 that the two-site double ionization potential tracks the chemical environment more sharply than either a single core hole or a single-site double hole, because the two-site energy feels the valence density at both atoms (DOI: [10.1063/1.451432](https://doi.org/10.1063/1.451432)). Their numbers are for CH₄, C₂H₂, C₂H₄, and C₂H₆, below.

Two experiments make the state. A synchrotron photon can eject both core electrons at once. The cross section is small, and the photon mostly makes the single-site hole; the two-site hole needs a knockout of the neighbor. An x-ray free-electron laser pulse can ionize the same molecule twice before Auger decay, on a few-femtosecond scale, and then both single-site and two-site holes are open. Reviews of the two routes: Piancastelli, *Eur. Phys. J. Spec. Top.* **222**, 2035 (2013), DOI: [10.1140/epjst/e2013-01985-9](https://doi.org/10.1140/epjst/e2013-01985-9); Lablanquie, Penent, and Hikosaka, *J. Phys. B* **49**, 182002 (2016), DOI: [10.1088/0953-4075/49/18/182002](https://doi.org/10.1088/0953-4075/49/18/182002).

From Step 4 on, this campaign is neither of those measurements. It removes two electrons from one core MO of the neutral closed shell and propagates. That is the sudden single-site initial condition used for the pentanones at the end of this note. On two equivalent atoms a canonical MO is a combination of the sites. Cederbaum's 1986 paper is also the warning about that: a delocalized orbital picture does not relax the way a localized hole does. Step 3 says which thioindigo MO is which atom. It does not by itself make a localized hole.

Metal K-edge double-hole emission has been simulated for transition-metal complexes (DOI: [10.1063/1.5111141](https://doi.org/10.1063/1.5111141)). Those complexes are not given their own sections. The sections below are the molecules with a stated site, a stated question, and a stated result.

## N₂

**Tashiro, Ehara, Fukuzawa, Ueda, Buth, Kryzhevoi, and Cederbaum, *J. Chem. Phys.* 132, 184302 (2010).** DOI: [10.1063/1.3408251](https://doi.org/10.1063/1.3408251)

Purpose. CASSCF single and double ionization potentials for a set of small molecules, and a way to turn those potentials into the generalized relaxation energy and the interatomic relaxation energy that an x-ray two-photon photoelectron spectrum would measure.

Result. In a diatomic the interatomic relaxation energy is negative: the first hole pulls valence density onto its own atom and the second site relaxes less. In N₂ that suppression is weaker than in CO, because the triple bond holds the valence density in place. Static correlation in the double ionization potentials of the set reaches 5.6 eV. ΔSCF with delocalized core orbitals misses 27–35 eV of the relaxation that CASSCF keeps.

**Fang et al., *Phys. Rev. Lett.* 105, 083005 (2010).** DOI: [10.1103/PhysRevLett.105.083005](https://doi.org/10.1103/PhysRevLett.105.083005)

Purpose. Make a double K hole in N₂ at the Linac Coherent Light Source by absorbing two photons inside the Auger lifetime, and separate the single-site hole from the two-site hole with photoelectron and Auger spectra.

Result. The single-site double hole is observed and matches theory. The paper gives an upper bound on the two-site contribution. This is the first direct molecular single-site double core hole.

**Cryan et al., *Phys. Rev. Lett.* 105, 083004 (2010).** DOI: [10.1103/PhysRevLett.105.083004](https://doi.org/10.1103/PhysRevLett.105.083004)

Purpose. Auger decay of the single-site double vacancy in impulsively aligned N₂, in the molecular frame, with 1.1 keV photons.

Result. The single-site Auger spectrum sits near 413 eV. It is shifted from the ordinary Auger spectrum by 51 ± 7 eV; the ab initio shift of the highest line is about 49 eV. The calculated two-site Auger electrons fall in a window of about 342–353 eV.

**Lablanquie et al., *Phys. Rev. Lett.* 106, 063003 (2011).** DOI: [10.1103/PhysRevLett.106.063003](https://doi.org/10.1103/PhysRevLett.106.063003)

Purpose. Single-photon double K ionization at a synchrotron, detected as two photoelectrons in coincidence with the Auger electrons. N₂ is the hollow-molecule case; CO, CO₂, and O₂ are the oxygen chemical-shift cases.

Result. The N₂ K⁻² main line and its satellites are measured, including how they form and how they decay.

**Salén et al., *Phys. Rev. Lett.* 108, 153003 (2012).** DOI: [10.1103/PhysRevLett.108.153003](https://doi.org/10.1103/PhysRevLett.108.153003)

Purpose. Test the predicted chemical sensitivity of two-site holes by x-ray two-photon photoelectron spectroscopy of N₂, N₂O, and CO₂ on the same instrument, with CO as the earlier reference.

Result. The two-site energy shifts of the set follow the theoretical ordering. N₂ is the homonuclear member against which the inequivalent sites in N₂O and CO₂ are compared. The shift is decomposed into the hole–hole repulsion and the interatomic relaxation.

**Larsson et al., *J. Phys. B* 46, 164030 (2013).** DOI: [10.1088/0953-4075/46/16/164030](https://doi.org/10.1088/0953-4075/46/16/164030)

Purpose. Write up the same Linac Coherent Light Source photoelectron measurements on N₂, N₂O, and CO₂, and compare them with rate-equation simulations.

Result. The measured two-site positions agree with the chemical-sensitivity theory. The relative intensities of the nonlinear peaks, compared with the simulation, estimate the x-ray pulse duration.

**Tenorio, Decleva, and Coriani, *J. Chem. Phys.* 155, 131101 (2021).** DOI: [10.1063/5.0062130](https://doi.org/10.1063/5.0062130)

Purpose. Restricted-active-space calculations of K⁻² shake-up satellites, with the neutral and the double-hole wavefunctions optimized separately, for H₂O, N₂, CO, and C₂H₂ₙ (n = 1–3).

Result. Binding energies and intensities of the N₂ shake-up satellites agree with the measured spectra.

**Communications Physics (2024).** DOI: [10.1038/s42005-024-01804-5](https://doi.org/10.1038/s42005-024-01804-5)

Purpose. A resonant, neutral double core excitation in N₂ at the European XFEL: one few-femtosecond pulse promotes a 1σ electron on each nitrogen into 1πg*.

Result. The two-site K⁻¹K⁻¹V² state is identified by its single-participator and double-participator decay. RASSCF and RASPT2 decay spectra match the measurement. The ordinary single-hole absorption calculation matches the synchrotron ion-yield spectrum of the same edge.

## CO

**Tashiro et al., *J. Chem. Phys.* 132, 184302 (2010).** DOI: [10.1063/1.3408251](https://doi.org/10.1063/1.3408251)

Purpose. Same CASSCF survey as under N₂. CO is the heteronuclear diatomic in that set, and the worked example of a 1 keV two-photon spectrum.

Result. Ejecting the second core electron from the same atom costs about 70–90 eV more than the first ionization. Ejecting it from the other atom costs about 15 eV more. The interatomic relaxation energy is negative, and the suppression is stronger than in N₂.

**Berrah et al., *Proc. Natl. Acad. Sci. U.S.A.* 108, 16912 (2011).** DOI: [10.1073/pnas.1111380108](https://doi.org/10.1073/pnas.1111380108)

Purpose. First direct observation of a two-site double core hole, using sequential two-photon absorption in CO at the Linac Coherent Light Source. Single-site holes were already known; the two-site hole is the one the chemical analysis needs.

Result. Carbon binding energies, with the single-hole line calibrated to the known value. Single hole: 296.5 ± 0.5 eV measured, 298.2 eV calculated. Second photon, single-site: 371.4 ± 3.5 eV measured, 369.6 eV calculated. Second photon, two-site: 312.8 ± 0.7 eV measured, 314.2 eV calculated. The two-site shift from the single hole is 16.3 ± 1.2 eV measured and 16.0 eV calculated.

**Lablanquie et al., *Phys. Rev. Lett.* 106, 063003 (2011).** DOI: [10.1103/PhysRevLett.106.063003](https://doi.org/10.1103/PhysRevLett.106.063003)

Purpose. Single-photon oxygen K⁻² spectra of CO, CO₂, and O₂, as a chemical-shift series on the same detector.

Result. The oxygen double-hole binding energy moves with the ligand. CO is one end of that series.

**J. Phys. Chem. Lett. 11, 4359 (2020).** DOI: [10.1021/acs.jpclett.0c01167](https://doi.org/10.1021/acs.jpclett.0c01167)

Purpose. A selected-configuration-interaction method with non-orthogonal orbitals for the oxygen single-site double-hole spectrum of CO, aimed at the shake-up satellites rather than the main line alone.

Result. The computed spectrum matches the experiment, including the intense satellites. The strong satellites move charge in the direction opposite to the relaxation the core hole itself induces.

**Tenorio, Decleva, and Coriani, *J. Chem. Phys.* 155, 131101 (2021).** DOI: [10.1063/5.0062130](https://doi.org/10.1063/5.0062130)

Purpose. Same shake-up study as under N₂.

Result. The CO K⁻² shake-up energies and intensities agree with experiment.

## CO₂

**Tashiro et al., *J. Chem. Phys.* 132, 184302 (2010).** DOI: [10.1063/1.3408251](https://doi.org/10.1063/1.3408251)

Purpose. Same CASSCF survey. CO₂ is the case with a third atom that can donate valence density.

Result. The interatomic relaxation energy is positive when the two holes sit on neighboring atoms: the remaining atom supplies density. It is negative when both holes sit on the terminal oxygens.

**Lablanquie et al., *Phys. Rev. Lett.* 106, 063003 (2011).** DOI: [10.1103/PhysRevLett.106.063003](https://doi.org/10.1103/PhysRevLett.106.063003)

Purpose. Oxygen K⁻² chemical shift of CO₂ against CO and O₂ by single-photon coincidence spectroscopy.

Result. The oxygen double-hole line of CO₂ is resolved from the CO and O₂ lines. The shift is large compared with the ordinary oxygen 1s chemical shift.

**Salén et al., *Phys. Rev. Lett.* 108, 153003 (2012).** DOI: [10.1103/PhysRevLett.108.153003](https://doi.org/10.1103/PhysRevLett.108.153003)

Purpose. Two-site holes in CO₂, compared with CO so that the extra oxygen is the only change, and compared with N₂ and N₂O.

Result. The O⁻¹C⁻¹ two-site state is measured. Its shift from the CO two-site state is the effect of the second oxygen, and it agrees with the relaxation analysis.

**Larsson et al., *J. Phys. B* 46, 164030 (2013).** DOI: [10.1088/0953-4075/46/16/164030](https://doi.org/10.1088/0953-4075/46/16/164030)

Purpose. The CO₂ photoelectron spectrum from the same free-electron-laser run, focused beam minus unfocused beam, against the rate-equation model.

Result. Single-site and two-site peaks are assigned. Their intensity ratio is one of the inputs to the pulse-duration estimate.

## N₂O

**Tashiro et al., *J. Chem. Phys.* 132, 184302 (2010).** DOI: [10.1063/1.3408251](https://doi.org/10.1063/1.3408251)

Purpose. Same CASSCF survey. N₂O is the molecule in which a two-site hole might be expected to amplify a site shift and does not.

Result. The terminal and central nitrogen single ionization potentials differ by 4 eV. The two-site potentials N_t⁻¹O⁻¹ and N_c⁻¹O⁻¹ differ by only 2.3–2.8 eV. Bond lengths plus the single-hole shift would have suggested about 11 eV. The reduction is relaxation: the interatomic term is positive for a hole pair on neighboring atoms and negative for the terminal nitrogen paired with oxygen.

**Salén et al., *Phys. Rev. Lett.* 108, 153003 (2012).** DOI: [10.1103/PhysRevLett.108.153003](https://doi.org/10.1103/PhysRevLett.108.153003)

Purpose. Measure that site dependence. The two nitrogens are inequivalent, which is the point of choosing N₂O.

Result. The two-site peaks associated with the two nitrogen sites are separated in the two-photon photoelectron spectrum, and the extracted relaxation terms match the sign structure of the calculation.

**Larsson et al., *J. Phys. B* 46, 164030 (2013).** DOI: [10.1088/0953-4075/46/16/164030](https://doi.org/10.1088/0953-4075/46/16/164030)

Purpose. Full photoelectron account of the N₂O run.

Result. The spectrum supports the same assignment. N₂O, with N₂ and CO₂, is where the chemical-sensitivity prediction is checked rather than assumed.

## O₂

**Lablanquie et al., *Phys. Rev. Lett.* 106, 063003 (2011).** DOI: [10.1103/PhysRevLett.106.063003](https://doi.org/10.1103/PhysRevLett.106.063003)

Purpose. Put O₂ on the same oxygen K⁻² chemical-shift scale as CO and CO₂.

Result. The O₂ double-hole binding energy is measured and is distinct from the CO and CO₂ values.

**Kastirke et al., *Phys. Rev. Lett.* 125, 163201 (2020).** DOI: [10.1103/PhysRevLett.125.163201](https://doi.org/10.1103/PhysRevLett.125.163201)

Purpose. Molecular-frame angular distribution of the second photoelectron when an O₂ double core hole is made at the European XFEL. Two electrons and two ions are detected in coincidence, for breakup into two O²⁺.

Result. Single-site and two-site holes are both identified. Relaxed-core Hartree–Fock reproduces the single-site angular distribution; frozen-core Hartree–Fock does not. For the two-site hole the two approximations agree with each other and with the measurement.

## H₂O

**Tenorio, Decleva, and Coriani, *J. Chem. Phys.* 155, 131101 (2021).** DOI: [10.1063/5.0062130](https://doi.org/10.1063/5.0062130)

Purpose. K⁻² shake-up of water, which had not been fully assigned.

Result. The satellite binding energies and intensities are assigned. The paper reports this as the first complete characterization of that water shake-up spectrum.

**Ismail, Inhester, et al., *Phys. Rev. A* 110, 013108 (2024).** DOI: [10.1103/PhysRevA.110.013108](https://doi.org/10.1103/PhysRevA.110.013108)

Purpose. Whether the nuclei move during the lifetime of a double core hole in water. The comparison is the Auger spectrum of H₂O against D₂O after sequential two-photon absorption.

Result. The isotope shift in the Auger spectrum is the nuclear motion. The double-hole lifetime used in the analysis is 1.5 fs. Restricted Hartree–Fock calculations of K⁻² and K⁻²V states with XMOLECULE reproduce the isotope effect.

## NH₃

**Eland, Tashiro, Linusson, Ehara, Ueda, and Feifel, *Phys. Rev. Lett.* 105, 213005 (2010).** DOI: [10.1103/PhysRevLett.105.213005](https://doi.org/10.1103/PhysRevLett.105.213005)

Purpose. Single-photon energies of the hollow molecules NH₃²⁺ and CH₄²⁺, both vacancies in the 1s shell, by multi-electron coincidence at a synchrotron.

Result. The NH₃ double-hole energy agrees with high-level calculations and with a simple model. The main decay is two Auger steps. The first step leaves a triply charged ion with one core hole and two valence holes, and that intermediate is observed. Pre-edge states with two holes and one excited electron lie below the double-hole threshold and are assigned by their decay.

**Tashiro, Ehara, and Ueda, *J. Chem. Phys.* 135, 154307 (2011).** DOI: [10.1063/1.3651082](https://doi.org/10.1063/1.3651082)

Purpose. Auger electron energies from the two-step decay of a molecular double core hole. NH₃, CH₄, and H₂CO are the examples.

Result. The NH₃ double hole has an empty 1s orbital, so the Auger spectrum splits into two components: double hole to core-plus-two-valence-holes, then that state to four valence holes. The calculated NH₃ spectrum matches Eland's measurement.

## CH₄

**Cederbaum, Tarantelli, Sgamellotti, and Schirmer, *J. Chem. Phys.* 85, 6513 (1986).** DOI: [10.1063/1.451432](https://doi.org/10.1063/1.451432)

Purpose. Compare the energy of a double core vacancy with a single core vacancy, and separate the single-site case from the two-site case. CH₄ is the one-carbon molecule; the chemical-shift claim is carried by C₂H₂, C₂H₄, and C₂H₆.

Result. CH₄ fixes the single-site relaxation analysis. Second-order perturbation theory gives a localized double-hole relaxation four times the single-hole relaxation. With delocalized orbitals, relaxation and correlation are the same size and both have to be kept. The paper's chemical conclusion is stated under the C₂ hydrocarbons: two-site binding energies separate the three molecules, and single-site binding energies do not.

**Eland et al., *Phys. Rev. Lett.* 105, 213005 (2010).** DOI: [10.1103/PhysRevLett.105.213005](https://doi.org/10.1103/PhysRevLett.105.213005)

Purpose. Same coincidence experiment as NH₃.

Result. The CH₄²⁺ 1s⁻² energy is measured and matches the calculation. The decay path is the same two Auger steps, and the core-plus-two-valence-hole intermediate is identified. The pre-edge 2-hole–1-particle states are located.

**Tashiro, Ehara, and Ueda, *J. Chem. Phys.* 135, 154307 (2011).** DOI: [10.1063/1.3651082](https://doi.org/10.1063/1.3651082)

Purpose. Same Auger calculation as NH₃.

Result. CH₄ has the same two separated Auger components as NH₃, because the double hole empties one 1s orbital.

## H₂CO

**Tashiro, Ehara, and Ueda, *J. Chem. Phys.* 135, 154307 (2011).** DOI: [10.1063/1.3651082](https://doi.org/10.1063/1.3651082)

Purpose. Auger decay when the molecule has more than one double-hole configuration. H₂CO has C 1s⁻², O 1s⁻², and both the singlet and the triplet C 1s⁻¹O 1s⁻¹.

Result. The two-site first Auger component overlaps the second Auger component, so an experiment would have trouble isolating them. The C 1s⁻² and O 1s⁻² components are separated from that overlap and are the ones an experiment can pick out.

## C₂H₂

**Cederbaum et al., *J. Chem. Phys.* 85, 6513 (1986).** DOI: [10.1063/1.451432](https://doi.org/10.1063/1.451432)

Purpose. Show that a two-site double core vacancy is a sharper chemical probe than a single core hole. Acetylene, ethylene, and ethane are the series: triple, double, and single carbon–carbon bonds.

Result. The two-site double ionization potentials of the three molecules separate. The single-site double ionization potentials, and the ordinary C 1s ionization potentials, do not. XPS does not tell the three compounds apart; the two-site spectrum should.

**Tashiro et al., *J. Chem. Phys.* 132, 184302 (2010).** DOI: [10.1063/1.3408251](https://doi.org/10.1063/1.3408251)

Purpose. Same CASSCF survey. The C₂ hydrocarbons are where the interatomic relaxation changes sign relative to the diatomics.

Result. The interatomic relaxation energy of C₂H₂ is positive. Valence density flows from the C–H bonds onto the two carbon holes, so the second hole relaxes more, not less.

**Lablanquie et al., *Phys. Rev. Lett.* 107, 193004 (2011).** DOI: [10.1103/PhysRevLett.107.193004](https://doi.org/10.1103/PhysRevLett.107.193004)

Purpose. Detect a single-photon two-site core double ionization. Acetylene is the molecule, because the two carbons are equivalent and the two-site state had been predicted since 1986.

Result. At 770.5 eV the two-site channel is 1.6 ± 0.4% of the single-site channel. A knockout model, in which the first 1s photoelectron ejects the other carbon 1s, agrees. The double-hole spectroscopy and the Auger decay are measured.

**Nakano et al., *Phys. Rev. Lett.* 110, 163001 (2013).** DOI: [10.1103/PhysRevLett.110.163001](https://doi.org/10.1103/PhysRevLett.110.163001)

Purpose. Single-photon K⁻² and K⁻¹K⁻¹ spectra of C₂H₂, C₂H₄, C₂H₆, CO, and N₂, as a chemical-analysis test and as a test of the knockout mechanism.

Result. The two-site line tracks the carbon–carbon bond length across acetylene, ethylene, and ethane more than the ordinary C 1s line does. The two-site cross section falls from C₂H₂ to N₂ to CO in the way the knockout model requires. The two-site Auger spectrum is cleanest for CO.

**Tashiro, Ehara, et al., *J. Chem. Phys.* 137, 224306 (2012).** DOI: [10.1063/1.4769777](https://doi.org/10.1063/1.4769777)

Purpose. Auger decay of the acetylene single-site double hole, the two-site double hole, and the C 1s⁻²π⁻¹π*⁺¹ satellite. Coincidence measurement and a two-step Auger calculation.

Result. The angle-integrated two-site Auger spectrum matches the calculation. The first and second Auger electrons overlap, so individual peaks cannot be assigned to one step. A separate N₂ single-site calculation in the same paper shows that nuclear motion on the repulsive final curves changes the second Auger energy distribution.

**Tenorio, Decleva, and Coriani, *J. Chem. Phys.* 155, 131101 (2021).** DOI: [10.1063/5.0062130](https://doi.org/10.1063/5.0062130)

Purpose. K⁻² shake-up of C₂H₂, which the paper says had not been fully characterized.

Result. The satellite spectrum is assigned, and the main line sits with the measured Nakano spectrum.

## C₂H₄

**Cederbaum et al., *J. Chem. Phys.* 85, 6513 (1986).** DOI: [10.1063/1.451432](https://doi.org/10.1063/1.451432)

Purpose. The double-bond member of the hydrocarbon series.

Result. Its two-site double ionization potential falls between acetylene and ethane and is separated from both. Its single-site value is not a useful label for the bond.

**Tashiro et al., *J. Chem. Phys.* 132, 184302 (2010).** DOI: [10.1063/1.3408251](https://doi.org/10.1063/1.3408251)

Purpose. Interatomic relaxation for the double-bond case.

Result. The interatomic relaxation energy is positive, as in acetylene and ethane, from density leaving the C–H bonds.

**Nakano et al., *Phys. Rev. Lett.* 110, 163001 (2013).** DOI: [10.1103/PhysRevLett.110.163001](https://doi.org/10.1103/PhysRevLett.110.163001)

Purpose. The measured middle point of the C₂H₂ₙ two-site series.

Result. The two-site binding energy lies between acetylene and ethane and follows the bond length. The single-site main line is only about 2 eV from acetylene: Tenorio, Decleva, and Coriani (DOI: [10.1063/5.0062130](https://doi.org/10.1063/5.0062130)) place the measured C₂H₄ − C₂H₂ single-site shift at −2.1 ± 0.2 eV.

**Tenorio, Decleva, and Coriani, *J. Chem. Phys.* 155, 131101 (2021).** DOI: [10.1063/5.0062130](https://doi.org/10.1063/5.0062130)

Purpose. Assign the C₂H₄ K⁻² shake-up.

Result. The main line and the satellites match the measured single-site spectrum. The shake-up of the C₂H₂ₙ series is assigned there for the first time.

## C₂H₆

**Cederbaum et al., *J. Chem. Phys.* 85, 6513 (1986).** DOI: [10.1063/1.451432](https://doi.org/10.1063/1.451432)

Purpose. The single-bond member of the hydrocarbon series.

Result. The two-site potential is the third distinct value. Ordinary XPS and the single-site double hole do not provide that third value.

**Tashiro et al., *J. Chem. Phys.* 132, 184302 (2010).** DOI: [10.1063/1.3408251](https://doi.org/10.1063/1.3408251)

Purpose. Interatomic relaxation for the single-bond case.

Result. Positive interatomic relaxation, same direction as acetylene and ethylene.

**Nakano et al., *Phys. Rev. Lett.* 110, 163001 (2013).** DOI: [10.1103/PhysRevLett.110.163001](https://doi.org/10.1103/PhysRevLett.110.163001)

Purpose. Close the measured C₂H₂ₙ two-site series on ethane.

Result. The two-site line is the long-bond end of the series. The single-site main line is 650.5 ± 0.5 eV, and it lies 1.9 ± 0.2 eV below acetylene, as compared in Tenorio, Decleva, and Coriani (DOI: [10.1063/5.0062130](https://doi.org/10.1063/5.0062130)). That is the small single-site shift. The two-site shift is the large one.

**Tenorio, Decleva, and Coriani, *J. Chem. Phys.* 155, 131101 (2021).** DOI: [10.1063/5.0062130](https://doi.org/10.1063/5.0062130)

Purpose. K⁻² shake-up of ethane.

Result. RASPT2 reproduces the measured main line at 650.5 ± 0.5 eV (calculated 649.55 eV) and assigns the satellites.

## C₆H₆

**Carniato et al., *J. Phys. B* 53, 244010 (2020).** DOI: [10.1088/1361-6455/abc663](https://doi.org/10.1088/1361-6455/abc663)

Purpose. Single-photon core ionization plus core excitation (K⁻²V) of benzene, by magnetic-bottle coincidence at a synchrotron. The question is whether a hollow-carbon satellite spectrum can be assigned in a six-carbon ring.

Result. The spectrum is dense and is assigned with DFT and post-Hartree–Fock calculations in a localized picture, both holes on one carbon. The Auger decay of those K⁻²V states is measured separately, with a coincidence cut that improves the resolution.

## para-Aminophenol, and the ortho and meta isomers

**Santra, Kryzhevoi, and Cederbaum, *Phys. Rev. Lett.* 103, 013002 (2009).** DOI: [10.1103/PhysRevLett.103.013002](https://doi.org/10.1103/PhysRevLett.103.013002)

Purpose. Inner-shell single and double ionization spectra of para-aminophenol by many-body Green's functions. The molecule was chosen because C, N, and O are distinguishable sites, as a proposal for two-photon spectroscopy at a free-electron laser.

Result. The double-ionization spectrum, and especially the two-site pairs, moves more with the chemical environment and with many-body corrections than the single-ionization spectrum. A kinetic model for a 1 fs pulse at 1 keV gives the photoelectron spectrum that experiment would record, including the probability of forming the double hole inside the Auger lifetime.

**Kryzhevoi, Santra, and Cederbaum, *J. Chem. Phys.* 135, 084302 (2011).** DOI: [10.1063/1.3624393](https://doi.org/10.1063/1.3624393)

Purpose. Single and double inner-shell ionization potentials of the ortho, meta, and para isomers, so the isomer shift itself is the observable.

Result. The two-site potentials separate the isomers by more than the single-hole potentials. Later DFT numbers on the para isomer (below) agree with these single-hole values more closely than with these double-hole values.

**Zhaunerchyk et al., *J. Phys. B* 48, 244003 (2015).** DOI: [10.1088/0953-4075/48/24/244003](https://doi.org/10.1088/0953-4075/48/24/244003)

Purpose. The Linac Coherent Light Source experiment that Santra's paper proposed, on 4-aminophenol, with a magnetic-bottle spectrometer and covariance mapping.

Result. The molecule absorbs more than two photons. Double-hole production is mixed with photoelectron–Auger–photoelectron–Auger sequences, and covariance mapping is what separates true pairs from accidentals. Their DFT single-hole binding energies are 538.89 eV (O), 405.29 eV (N), and 289.7–291.4 eV (ring carbons). The single-site double-hole values are 1164.26 eV (O), 886.50 eV (N), and about 643–647 eV (carbons).

**Chem. Sci. 7, 5922 (2016).** DOI: [10.1039/C6SC01571A](https://doi.org/10.1039/C6SC01571A)

Purpose. Simulate the all-x-ray double-quantum-coherence spectrum of the three aminophenol isomers. The signal correlates a single core hole with a double core hole. SA-RASSCF treats valence, single-hole, and double-hole states in one calculation, for N 1s⁻², O 1s⁻², and N 1s⁻¹O 1s⁻¹.

Result. The three isomers do not give the same pattern. In the para isomer the N 1s / O 1s double-hole states that carry the signal lie near 934.2, 935.5, and 937.0 eV, fed by N 1s resonances at 401.5 and 402.7 eV. The ortho isomer has an additional feature at an N 1s energy of 404.5 eV that the other two isomers lack.

## H₂S

**Linusson, Takahashi, Ueda, Eland, and Feifel, *Phys. Rev. A* 83, 022506 (2011).** DOI: [10.1103/PhysRevA.83.022506](https://doi.org/10.1103/PhysRevA.83.022506)

Purpose. Single-site S 2p⁻² double holes in H₂S, SO₂, and CS₂. Ordinary S 2p ionization potentials of H₂S and CS₂ are almost equal (170.6 and 170.2 eV), so XPS does not show the different ground-state charge. The double-hole measurement is supposed to separate that initial-state shift from the relaxation.

Result. Coincidence spectra at 500 eV resolve ³P, ¹D, and ¹S. The statistically weighted H₂S double ionization potential is 376.7 eV measured and 377.2 eV from MCSCF. DIP − 2 IP is 35.5 eV measured and 35.8 eV calculated. The multiplet spacing is nearly the same as in SO₂ and CS₂, which is the atomic S 2p exchange. H₂S and CS₂, which look alike in XPS, do not give the same excess relaxation once the double-hole energy is used.

## SO₂

**Linusson et al., *Phys. Rev. A* 83, 022506 (2011).** DOI: [10.1103/PhysRevA.83.022506](https://doi.org/10.1103/PhysRevA.83.022506)

Purpose. The oxidized member of the same sulfur series. The ordinary S 2p ionization potential is already different: 175.2 eV.

Result. Weighted double ionization potential 385.1 eV measured, 386.7 eV calculated. DIP − 2 IP is 34.7 eV measured and 35.3 eV calculated. The double-hole shift from H₂S is mostly the same initial-state shift the single hole already shows; the excess relaxation is not what distinguishes SO₂ from H₂S.

## CS₂

**Linusson et al., *Phys. Rev. A* 83, 022506 (2011).** DOI: [10.1103/PhysRevA.83.022506](https://doi.org/10.1103/PhysRevA.83.022506)

Purpose. The molecule that XPS cannot tell from H₂S. S 2p ionization potential 170.2 eV against 170.6 eV for H₂S.

Result. Weighted double ionization potential 373.3 eV measured, 373.9 eV calculated. DIP − 2 IP is 32.9 eV measured and 33.1 eV calculated, about 2.6 eV below H₂S. That difference is the initial-state chemical shift that the single-hole line had cancelled against relaxation. The ³P, ¹D, and ¹S intervals still match the other two molecules.

## LiF, BeO, and BF

**Tashiro et al., *J. Chem. Phys.* 132, 184302 (2010).** DOI: [10.1063/1.3408251](https://doi.org/10.1063/1.3408251)

Purpose. The ionic end of the same CASSCF set, against the covalent diatomics.

Result. The interatomic relaxation energy is negative in all three, as in every diatomic in the paper. Its magnitude is smaller in LiF and BF than in BeO. The valence density is already on the fluorine, so the first hole cannot pull much more density across the bond. BeO behaves more like CO.

## Pyrimidine, purine, the nucleobases, and formamide

**Takahashi et al., *J. Phys. Chem. A* 115, 12070 (2011).** DOI: [10.1021/jp205923m](https://doi.org/10.1021/jp205923m)

Purpose. Whether the relaxation analysis still organizes double-hole states once the molecule is a nucleobase. DFT for pyrimidine, uracil, cytosine, thymine, purine, adenine, and guanine, and for formamide. Formamide is also done with CASSCF. Every single-site and two-site pair is included.

Result. One trend covers the set, rather than a separate spectroscopic assignment of each base. The generalized single-site relaxation energy correlates with the natural-bond-orbital charge on the ionized atom. The interatomic relaxation energy correlates with the distance between the two holes. The paper's conclusion is that those two correlations are enough to use double-hole energies for chemical analysis of a molecule this size.

## 3-Pentanone

**Nascimento, Zhang, Bergmann, and Govind, *J. Phys. Chem. Lett.* 11, 556 (2020).** DOI: [10.1021/acs.jpclett.9b03500](https://doi.org/10.1021/acs.jpclett.9b03500)

Purpose. Carbon K-edge absorption after a single or a double core hole on the oxygen of 3-pentanone, 2-pentanone, and pentanal. The neutral carbon pre-edge lines of these isomers sit on top of each other. The oxygen hole is meant to pull them apart. The same sudden hole drives a valence charge migration, which is the dynamics traced in the paper's Figure 8. That figure is the 3-pentanone occupation trace compared in `jobs/DCH`. The computed spectra in the paper are shifted by +11.4 eV.

Result. Neutral 3-pentanone (C₂ᵥ) has three C 1s → π* energies inside 0.15 eV, with the main pre-edge intensity at 288.0 eV on the carbonyl and the methylenes. One oxygen hole separates methyl from methylene by about 1 eV and puts a methylene doublet at 284.9 and 285.2 eV. The double hole spreads the pre-edge features by at least 2 eV. The methylene line moves to 283.4 eV and becomes the strong one, and a methyl line appears at 281.5 eV. The LUMO, which started on the carbonyl and the methylenes, picks up amplitude on the methyl groups. The sudden oxygen hole also moves valence hole density among the occupied orbitals on an attosecond period. The occupations of that migration are the comparison used to check the sudden-double-hole propagator.

## 2-Pentanone

**Nascimento et al., *J. Phys. Chem. Lett.* 11, 556 (2020).** DOI: [10.1021/acs.jpclett.9b03500](https://doi.org/10.1021/acs.jpclett.9b03500)

Purpose. The same oxygen-hole carbon-edge spectra for the isomer in which all five carbons are inequivalent.

Result. The neutral pre-edge is still unresolved. The double hole splits it into five transitions. The band near 283.3 eV is two lines, at 283.26 and 283.38 eV, from the two carbons that sit almost equally far from the carbonyl. C₁ and C₃ are not separated from each other by the hole. The LUMO remains on the carbonyl and the methylenes, with a changed balance between them.

## Pentanal

**Nascimento et al., *J. Phys. Chem. Lett.* 11, 556 (2020).** DOI: [10.1021/acs.jpclett.9b03500](https://doi.org/10.1021/acs.jpclett.9b03500)

Purpose. The aldehyde, which has no C₂ᵥ symmetry, as the case where even a single oxygen hole might be enough.

Result. The five neutral C 1s → π* energies lie inside 0.7 eV. One oxygen hole spreads them over about 6 eV, with gaps of at least 0.4 eV. The double hole increases the minimum gap to 0.8 eV. The neutral LUMO is on the carbonyl with a small amplitude at C₂, and the pre-edge intensity is at about 287.7 eV. The double hole spreads that LUMO along the chain and leaves a node at C₃, so every carbon pre-edge except C₃ becomes strong.

## What this campaign takes from the list

The production initial condition is the pentanone one: a sudden double vacancy on one core MO, neutral orbitals kept, real-time propagation of the valence holes. It is not a calculated double ionization potential, and it is not a two-site hole. The two-site papers are the evidence that a second hole is a chemical probe. They are not the observable this series records.

Which core that vacancy sits on is not chosen. It waits on the molecule. A survey of trans-thioindigo, run before that choice was unset, is in `Step_2/thioindigo_trial/`: MO 0 and MO 1 are sulfur 1s, MO 2 and MO 3 are oxygen 1s. The sulfur comparison that exists in this note is S 2p⁻² of H₂S, SO₂, and CS₂, not a sulfur K⁻² hole. The oxygen comparison that matches the sudden-hole dynamics is 3-pentanone. Neither fact picks the production molecule. Do not copy the thioindigo indices forward unless that molecule is the one selected.

## Valence absorption against the plasmons

The NWChem folders have been run. trans-Thioindigo is the exception: `Molecules/transthioindigo/` is a PySCF linear-response spectrum on the existing PBE0/6-311G* geometry, not an NWChem optimization, and `LC-wPBE mu/transthioindigo/tune.json` has not been run. The geometry level is still PBE0/6-311G*. The roots are LC-ωPBE, μ = 0.34272. A line is counted as bright when its dipole oscillator strength is at least 0.01 and its energy is at most 10 eV. Degenerate π pairs are one transition. LiF and BeO were taken to 24 roots so the list passes 10 eV. Adenine stops at 9.94 eV and guanine at 9.97 eV with 30 roots; their bright lines are below that. trans-Thioindigo was 30 singlets and its last root is 7.28 eV.

The plasmons this screen was compared with are not selected. One is a 35 nm `Ag` sphere in water. Its Mie peak is 418.5 nm, 2.96 eV. Across diameters 10–60 nm that same dielectric stays between 406 and 442 nm. The other is the gold sphere the original dye search used, about 50 nm, near 540 nm.

Nothing in this set sits on either of those. The only visible lines are BeO, and both are weak (f = 0.02).

| Molecule | Lowest bright line | Wavelength | f |
| --- | --- | --- | --- |
| BeO | 1.959 eV | 633 nm | 0.019 |
| trans-thioindigo | 2.976 eV | 417 nm | 0.286 |
| BeO, next | 3.162 eV | 392 nm | 0.021 |
| para-aminophenol | 4.913 eV | 252 nm | 0.051 |
| ortho-aminophenol | 5.002 eV | 248 nm | 0.064 |
| meta-aminophenol | 5.106 eV | 243 nm | 0.038 |
| cytosine | 5.163 eV | 240 nm | 0.072 |
| guanine | 5.349 eV | 232 nm | 0.176 |
| purine | 5.358 eV | 231 nm | 0.109 |
| thymine | 5.491 eV | 226 nm | 0.200 |
| LiF | 5.495 eV | 226 nm | 0.035 |
| adenine | 5.614 eV | 221 nm | 0.013 |
| uracil | 5.643 eV | 220 nm | 0.189 |
| pyrimidine | 5.937 eV | 209 nm | 0.035 |
| BF | 6.324 eV | 196 nm | 0.221 |
| SO₂ | 6.504 eV | 191 nm | 0.067 |
| CS₂ | 6.875 eV | 180 nm | 1.223 |
| H₂S | 6.915 eV | 179 nm | 0.021 |
| NH₃ | 7.096 eV | 175 nm | 0.052 |
| benzene | 7.409 eV | 167 nm | 0.60 |
| H₂O | 7.499 eV | 165 nm | 0.025 |
| formamide | 7.948 eV | 156 nm | 0.011 |
| 2-pentanone | 8.116 eV | 153 nm | 0.030 |
| ethylene | 8.169 eV | 152 nm | 0.384 |
| formaldehyde | 8.308 eV | 149 nm | 0.112 |
| pentanal | 8.394 eV | 148 nm | 0.061 |
| CO | 8.661 eV | 143 nm | 0.084 |
| 3-pentanone | 8.995 eV | 138 nm | 0.036 |
| O₂ | 9.550 eV | 130 nm | 0.189 |

N₂, CO₂, N₂O, acetylene, and ethane have roots below 10 eV and none of them are bright. The lowest are N₂O at 6.65 eV, acetylene at 6.93 eV, CO₂ at 8.86 eV, and N₂ at 9.52 eV, all with zero dipole strength in this calculation. Methane's first root is 11.56 eV (f = 0.22). Ethane's first root is 10.22 eV and dark.

trans-Thioindigo at 417 nm is 1.5 nm short of the 35 nm silver peak. That is the one bright line in this set on that plasmon. The benzene maximum is 543 nm and the vapor maximum is 508 nm, so the solvent is about 0.16 eV of the gap to this root. The rest is LC-ωPBE on the charge-transfer excitation. If this dye is selected, match 417 nm, not 543 nm. The 418.5 nm silver number is the Rakic `Ag` dielectric, not `Ag_visible`. Meep marks that Palik fit unstable. BeO at 392 nm is 26 nm short of the same peak and 14 nm short of the 10 nm silver peak, with f = 0.02. BeO at 633 nm is 93 nm long of the 540 nm gold target. The nearest other organic line is para-aminophenol at 252 nm, 166 nm short of the silver peak. No literature molecule here is a bright line on either plasmon. The production molecule is still not chosen. trans-Thioindigo is in the set and is not selected.
