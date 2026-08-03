# Theory and Methodology of PlasMol

PlasMol performs **self-consistent hybrid FDTD–RT-TDDFT** simulations of plasmon–molecule systems. Classical electromagnetic fields (Meep) drive quantum time propagation (PySCF), and the induced molecular dipole is fed back into the classical simulation as a polarization source. This page presents the theoretical framework in detail. Specialized workflows (Fourier spectra, the quasistatic Gersten–Nitzan model, and sudden core-hole dynamics) are expanded on linked pages.

---

## Background and significance

Plasmon–molecule interactions are a cornerstone of modern nanophotonics: they enable manipulation of light–matter phenomena at the nanoscale with large local-field enhancements. When a metallic nanoparticle (NP) is illuminated by electromagnetic radiation, the collective oscillation of its conduction electrons—the **localized surface plasmon resonance (LSPR)**—generates intense electromagnetic fields near the particle surface. Those fields couple strongly to nearby molecules. A prime example is **surface-enhanced Raman spectroscopy (SERS)**, where the Raman scattering cross-section of molecules on or near plasmonic surfaces can be amplified by orders of magnitude, enabling detection down to the single-molecule level. Related applications include ultrasensitive biosensing and plasmon-driven chemistry such as hot-electron photocatalysis. The **Purcell effect** further shows that a surrounding NP can enhance the spontaneous emission rate of a quantum emitter (atom, molecule, or quantum dot).

While many such interactions primarily affect the molecule’s electronic *excited* states, they may also extend to deeper electronic processes involving **core electrons**. A molecule with a single core-hole (SCH), typically created through X-ray absorption, is expected to experience markedly different electronic dynamics in the presence of a plasmon. Double core-hole (DCH) spectroscopy—where two core electrons are promoted via X-ray absorption—offers even more chemically identifiable spectral features among similar species. Given that intense NP fields can modify lifetimes of molecular excited states, plasmon-modulated S/DCH spectroscopy and dynamics are a central scientific motivation for PlasMol.

As theoreticians, we construct and probe such systems computationally. Care is needed in choosing the method: NPs are typically large enough that their plasmonic response is well described by **classical** Maxwell solvers, while the molecule requires an **electronic-structure** treatment. Bulk-level approaches that replace the molecular layer by an additional dielectric material can capture qualitative bulk NP–molecule trends but sacrifice molecular shape and electronic structure. Parameterized two- or four-level systems likewise lack flexibility for chemically detailed applications.

PlasMol therefore couples **finite-difference time-domain (FDTD)** electromagnetics for the NP and fields to **real-time time-dependent density functional theory (RT-TDDFT)** for the molecule, so that plasmonic and electronic subsystems evolve simultaneously under mutual influence. The software is open-source and intended for the computational chemistry and nanophotonics communities.

---

## Classical and quantum hybrid computational algorithm

### Real-time time-dependent density functional theory

In RT-TDDFT, electronic dynamics are described by the time-dependent Kohn–Sham (KS) equations

$$
\mathrm{i}\,\frac{\partial}{\partial t}\,\phi_i(\mathbf{r},t)
=
\left[
  -\frac{\nabla^2}{2m}
  +
  v_{\mathrm{KS}}(\mathbf{r},t)
\right]
\phi_i(\mathbf{r},t),
$$

where $\phi_i(\mathbf{r},t)$ are single-particle orbitals. The KS potential decomposes as

$$
v_{\mathrm{KS}}(\mathbf{r},t)
=
v_{\mathrm{H}}[\rho](\mathbf{r},t)
+
v_{\mathrm{xc}}[\rho](\mathbf{r},t)
+
v_{\mathrm{ext}}(\mathbf{r},t),
$$

with $v_{\mathrm{H}}$ the Hartree potential, $v_{\mathrm{xc}}$ the exchange–correlation (xc) potential, $v_{\mathrm{ext}}$ the external potential, and

$$
\rho(\mathbf{r},t)
=
\sum_{i=1}^{N}
\bigl|\phi_i(\mathbf{r},t)\bigr|^2
$$

the electron density for $N$ electrons. The choice of xc functional is particularly important in RT-TDDFT, as it governs the accuracy of the electronic dynamics and response properties.

The Hamiltonian from the KS equations above can be rewritten as the time-dependent KS Hamiltonian

$$
\hat{H}_{\mathrm{KS}}(t)
=
-\frac{\nabla^2}{2m}
+
v_{\mathrm{H}}[\rho](\mathbf{r},t)
+
v_{\mathrm{xc}}[\rho](\mathbf{r},t)
+
v_{\mathrm{ext}}(\mathbf{r},t),
$$

often represented in a basis set as the **Fock matrix**, $\hat{H}_{\mathrm{KS}}(t)\equiv F(t)$. This framework is especially useful when the molecule responds to external perturbations such as the electric field of an optically excited NP.

Under an electric field, the molecule’s primary response is the physical displacement of its electrons, measured by the **induced dipole** $\boldsymbol{\mu}_{\mathrm{ind}}$. By the **Runge–Gross theorem**, observables such as $\boldsymbol{\mu}_{\mathrm{ind}}$ are functionals of $\rho(\mathbf{r},t)$ and the initial state $\{\phi_i(\mathbf{r},0)\}$. In practice we work with the **matrix representation** of the density rather than $\rho(\mathbf{r},t)$ on a real-space grid.

The time-dependent KS orbitals are expanded in a finite atomic-orbital basis $\{\chi_\mu(\mathbf{r})\}$:

$$
\phi_i(\mathbf{r},t)
=
\sum_{\mu}
C_{\mu i}(t)\,\chi_\mu(\mathbf{r}),
$$

where $\{C_{\mu i}(t)\}$ are molecular orbital (MO) coefficients. The one-particle reduced density matrix $\mathbf{D}(t)$ then has elements

$$
D_{\mu\nu}(t)
=
\sum_{i=1}^{N}
C_{\mu i}(t)\,C_{\nu i}^{*}(t).
$$

The electron density can be reconstructed as

$$
\rho(\mathbf{r},t)
=
\sum_{\mu\nu}
D_{\mu\nu}(t)\,\chi_{\mu}^{*}(\mathbf{r})\,\chi_{\nu}(\mathbf{r}),
$$

so that one propagates a matrix of coefficients rather than a volumetric density field while still satisfying the conditions of the Runge–Gross theorem.

Using these matrices, the induced dipole moment is computed as

$$
\boldsymbol{\mu}_{\mathrm{ind}}(t)
=
-\Bigl(
  \operatorname{Tr}\!\bigl[\mathbf{D}(t)\cdot\boldsymbol{\mu}\bigr]
  -
  \operatorname{Tr}\!\bigl[\mathbf{D}(0)\cdot\boldsymbol{\mu}\bigr]
\Bigr),
$$

where $\boldsymbol{\mu}$ is the dipole integral matrix with elements $\mu_{\mu\nu}=\langle\chi_\mu|\mathbf{r}|\chi_\nu\rangle$.

PlasMol propagates the RT-TDDFT system with a **second-order Magnus** operator and a self-consistent **predictor–corrector** scheme (detailed in [RT-TDDFT propagation](#appendix-a1-rt-tddft-propagation) below). From the time series of $\boldsymbol{\mu}_{\mathrm{ind}}(t)$ one can produce an absorption spectrum and compare it to experiment.

An absorption spectrum from *unmodified* RT-TDDFT yields discrete lines. While peak *energies* at low energy often align reasonably with experiment, the stick spectrum does not resemble experimental line shapes. At high enough energies a broad shoulder appears as electrons ionize from the finite basis. Experimental spectra are broadened by vibrational effects, solvent interactions, thermal broadening, lifetime effects, and instrumental resolution. Researchers therefore often apply artificial post-processing broadening. More detail on spectrum construction and artificial broadening is given in [Producing an absorption spectrum](#appendix-a2-producing-an-absorption-spectrum-in-rt-tddft).

```{figure} assets/methodology/fig2_water_absorption.png
:alt: Water absorption spectrum RT-TDDFT vs EELS
:width: 90%

Absorption spectrum generated by RT-TDDFT without (pink) and with (green) artificial broadening applied globally. EELS data (orange) from Brion (as cited in Lopata & Govind, *J. Chem. Theory Comput.* **9**, 4939 (2013)).
```

### Finite-difference time domain

The FDTD method solves the time-dependent Maxwell equations on a spatial grid. In macroscopic form,

$$
\nabla\times\mathbf{E}(t)
=
-\mu\,\frac{\partial\mathbf{H}(t)}{\partial t},
$$

$$
\nabla\times\mathbf{H}(t)
=
\epsilon\,\frac{\partial\mathbf{E}(t)}{\partial t}
+
\frac{\partial\mathbf{P}(t)}{\partial t},
$$

where $\mathbf{E}(t)$, $\mathbf{H}(t)$, and $\mathbf{P}(t)$ are the electric, magnetic, and polarization fields, respectively. PlasMol uses [Meep](https://meep.readthedocs.io/) for this classical subsystem: spherical metallic NPs (e.g. Au/Ag with dispersive materials), custom sources, symmetries, PML boundaries, optional field imaging/GIFs, and flux-based absorption/scattering cross sections.

### Hybrid theory

The hybrid workflow begins with an FDTD simulation that contains the NP of interest and a molecular site represented, at the classical resolution, as a **singular point / voxel dipole**. Concurrently, the RT-TDDFT side is initialized by performing all ground-state calculations for the molecule.

At each time step, the electric field $\mathbf{E}(t)$ at the dipole position in FDTD is inserted as an external potential into the $v_{\mathrm{ext}}(\mathbf{r},t)$ term of the KS Hamiltonian in a single RT-TDDFT step:

$$
v_{\mathrm{ext}}(\mathbf{r},t)
=
-\mathbf{r}\cdot\mathbf{E}(t)
=
-\sum_{i\in\{x,y,z\}}
r_i\,E_i(t).
$$

Propagation then evolves the density matrix to $\mathbf{D}(t+\Delta t)$, from which the induced dipole is obtained from the induced-dipole trace formula above. Because molecules are typically much smaller than the optical wavelength, we assume that $\boldsymbol{\mu}_{\mathrm{ind}}$ sufficiently captures the molecular response (electric-dipole approximation).

After unit conversion and scaling, this quantity becomes a polarization field emitted by the molecule,

$$
\mathbf{P}(\mathbf{r},t)
=
ea_0\,\frac{\boldsymbol{\mu}_{\mathrm{ind}}}{\Delta V},
$$

where $\Delta V$ is the volume occupied by the molecule (voxel) in SI units. $\mathbf{P}$ is added to the FDTD simulation for the next time step. Through this loop, the NP influences the molecule and the molecule influences the NP **in real time**.

```{figure} assets/Method_Schema.png
:alt: Hybrid FDTD and RT-TDDFT coupling schematic
:width: 95%

Schematic of the hybrid algorithm. Gray arrows are handled by the FDTD implementation (Meep). Formation of the $\mathbf{D}$ and $\mathbf{F}$ matrices uses PySCF’s DFT infrastructure. The quantum side uses a second-order Magnus propagator of the Liouville–von Neumann equation $\mathrm{i}\,\partial_t\mathbf{D}=[\mathbf{F}(t),\mathbf{D}(t)]$.
```

#### High-level workflow (per time step)

1. **Classical advance (Meep)** — Update $\mathbf{E}$ and $\mathbf{H}$ according to Maxwell’s equations with sources (external drive + molecular polarization).
2. **Field extraction** — If a molecule is present and $|\mathbf{E}|$ at its location exceeds `tolerance_field_e`, extract $\mathbf{E}(t)$ at that point.
3. **Quantum propagation (RT-TDDFT)** — Build the time-dependent Fock matrix in the orthogonal basis, optionally with non-Hermitian CAP broadening. $ F_{\mathrm{orth}}(t) = F_0 + V_{\mathrm{ext}}\!\bigl(\mathbf{E}(t)\bigr) - \mathrm{i}\,\Gamma(t). $
Propagate orbitals / density with the chosen propagator (default: 2nd-order Magnus + predictor–corrector). Compute $\boldsymbol{\mu}_{\mathrm{ind}}(t)$
4. **Back-coupling** — Inject $\boldsymbol{\mu}_{\mathrm{ind}}$ into Meep as a `CustomSource` / polarization at the molecule’s position (via the polarization injection above).
5. **Repeat** until $t_{\mathrm{end}}$.

---

## Core-hole dynamics and dissipative channels

Incident electric fields from a plasmon can influence lifetimes of electronic excited states. Core-hole dynamics should not be exempt from this physics, yet few studies have confirmed it. Simulating plasmon influence on SCH and DCH dynamics and spectra is therefore a primary application of PlasMol. Standard TDDFT cannot fully capture some dissipative pathways in S/DCH; improved treatments of **state-specific broadening** are needed for faithful ultrafast simulations.

One promising strategy is to incorporate dissipative channels **directly** into the TDDFT Hamiltonian. Lopata and Govind demonstrated—and subsequent studies confirmed—that a non-Hermitian dissipative term yields remarkably accurate state-specific broadenings. Although the approach primarily accounts for autoionization decay, it produces realistic peak widths that align well with experiment. Inclusion of such a modification is essential for evaluating lifetimes in S/DCH relaxations, particularly under plasmon coupling.

Operational SCH/DCH initial conditions, MO tracking, and JSON parameters are documented on the dedicated page [Core-Hole Dynamics](core_hole.md).

### The non-Hermitian (Lopata) approach

The continuum cannot be described by the finite Gaussian basis sets used for molecular TDDFT. Lopata and Govind introduced an ansatz that replicates ionization in RT-TDDFT by adding a **non-Hermitian absorbing potential** to the Fock matrix.

First, construct a diagonal **damping matrix**

$$
\boldsymbol{\Lambda}
=
\begin{pmatrix}
\gamma_1 & 0 & \cdots & 0 \\
0 & \gamma_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \gamma_M
\end{pmatrix},
$$

with $M$ the number of MOs and $\gamma_i$ the damping parameter for the $i$th MO (energy $\varepsilon_i$), chosen to damp all MOs above a cutoff energy $\varepsilon_0$:

$$
\gamma_i
=
\begin{cases}
0, & \varepsilon_i \le \varepsilon_0, \\[0.4em]
\gamma_0\bigl[\exp\!\bigl(\xi(\varepsilon_i-\varepsilon_0)\bigr)-1\bigr],
& \varepsilon_i > \varepsilon_0.
\end{cases}
$$

Here $\xi$ controls the intensity of the damping and $\gamma_0$ sets the energy scale (both chosen phenomenologically). After projection onto the time-dependent MO eigenvectors,

$$
\boldsymbol{\Gamma}^{\mathrm{MO}}(t)
=
\mathbf{C}(t)\,\boldsymbol{\Lambda}\,\mathbf{C}^{\dagger}(t),
$$

the damping matrix is added to the unaltered Fock matrix throughout the RT-TDDFT propagation:

$$
\mathbf{F}^{\mathrm{MO}}(t)
=
\mathbf{F}_0^{\mathrm{MO}}(t)
+
\mathrm{i}\,\boldsymbol{\Gamma}^{\mathrm{MO}}(t).
$$

This method closely replicates absorption spectra of water and acetylene relative to unmodified TDDFT. Even though many dissipative channels exist, an absorptive potential based solely on ionization remarkably recovers much of the experimental broadening—including the prominent continuum shoulder above the ionization threshold.

```{figure} assets/methodology/fig5_lopata_absorption.png
:alt: Water absorption with non-Hermitian CAP vs experiment
:width: 90%

Absorption spectrum generated by RT-TDDFT without (pink) and with (blue) the non-Hermitian absorption term $\mathrm{i}\Gamma$. EELS data (orange) from Brion (as cited in Lopata & Govind).
```

The ansatz is **highly dependent on the exchange–correlation functional and basis set**, because the MO energies that enter $\mathbf{\Lambda}$ shift with both. Lopata and Govind’s preferred choice is a Koopmans IP-tuned LC-PBE\*. PlasMol implements the CAP pathway in the quantum stack and exposes tuning of LRC / vacuum parameters; users should verify that the LUMO (or chosen continuum onset) sits above the damping threshold for their functional and basis.

```{figure} assets/methodology/fig3_mo_energies.png
:alt: MO energy grids for water across basis and functional
:width: 95%

Basis set and functional affect MO energies for the Lopata ansatz. Rows: 6-31G\*, aug-cc-pVTZ, def2-TZVPPD. Columns: PBE0, B3LYP, CAM-B3LYP, LC-PBE. Graphs show the 3rd–8th MO energies for H$_2$O. Green: 6th MO is the first with positive energy; yellow: 6th MO positive but very near zero ($<10^{-2}$); red: 6th MO still negative.
```

Moreover, $\boldsymbol{\Lambda}$ is typically built from the **time-independent ground-state** MO spectrum. A natural extension is to carry the same dissipation idea into **frequency-domain** linear-response TDDFT, so that state-specific widths appear without long real-time propagations.

---

## Validation against an analytically soluble model

The hybrid methodology above—classical Maxwell evolution, molecular RT-TDDFT driven by the local field, optional dipole back-action, and Fourier spectra built from $\mu(\omega)/E_{\mathrm{inc}}(\omega)$—is **not only formally consistent; it can be checked against a model that is analytically soluble** in the same physical regime.

### Why a quasistatic sphere + point molecule is a benchmark

In the quasistatic limit ($d\ll\lambda$), a point polarizable molecule at distance $d$ from a sphere of radius $a$ admits a closed Gersten–Nitzan construction: the molecule responds with an **effective polarizability** $\alpha_{\mathrm{eff}}(\omega)$ relative to the incident field $\mathbf{E}_{\mathrm{inc}}$, obtained by eliminating the induced multipoles of the sphere. Absorption is then

$$
A(\omega)
\propto
-\omega\,\operatorname{Im}\,\alpha_{\mathrm{eff}}(\omega),
$$

with distinct parallel (radial) and perpendicular (tangential) channels. The full algebra—Lorentz $\alpha_m$, Mie multipoles $\alpha_{\mathrm{NP}}^{(\ell)}$, local-field factors $G_{\parallel,\perp}$, image propagators $S_{\parallel,\perp}$, and

$$
\alpha_{\mathrm{eff}}
=
\frac{\alpha_m G}{1-\alpha_m S}
$$

—is developed on the dedicated page [Quasistatic Model](quasistatic_model.md).

That model is analytically soluble once geometry, host index, metal dielectric $\varepsilon_m(\omega)$, and oscillator parameters are fixed. PlasMol’s hybrid Fourier workflow targets the **same observables**:

| Quasistatic model | PlasMol hybrid Fourier |
| ------------------- | ------------------------ |
| Incident field $\mathbf{E}_{\mathrm{inc}}$ | Vacuum reference $E_{\mathrm{inc}}(t)$ (no NP, no molecule) |
| $\alpha_{\mathrm{eff}}(\omega)=\mathbf{p}/E_{\mathrm{inc}}$ | $\mathcal{R}(\omega)=\mu(\omega)/E_{\mathrm{inc}}(\omega)$ |
| $A\propto -\omega\,\operatorname{Im}\alpha_{\mathrm{eff}}$ | $A\propto -\omega\,\operatorname{Im}\mathcal{R}$ |
| Parallel channel $G_\parallel$, $S_\parallel$ | Polarization mode `parallel` (drive along NP→molecule) |
| Perpendicular channel $G_\perp$, $S_\perp$ | Polarization mode `perpendicular` (drive $\perp$ NP→molecule) |
| Classical back-action $\alpha_m S$ | Optional molecular dipole back-propagation into Meep |

Agreement between the two therefore tests the **end-to-end hybrid pipeline**: transverse Meep drive, field sampling at the molecule, RT-TDDFT induced dipole, vacuum deconvolution, and orientation-resolved spectra—not merely a single isolated subroutine.

### Observed alignments

Hybrid Fourier spectra for a molecule near a gold sphere were computed in **parallel** and **perpendicular** polarizations and fit jointly to the quasistatic $\alpha_{\mathrm{eff}}$ model (geometry fixed to the numerical cell; free parameters mainly the Lorentz oscillator $\omega_m$, $\gamma_m$, $\alpha_0$). The classical curves track the PlasMol lineshapes on both orientations, including the relative strength and slight peak shifts between the radial and tangential channels.

```{figure} assets/spectrum_model2_joint_fit.png
:alt: Joint quasistatic fit to PlasMol parallel and perpendicular hybrid spectra (Au_JC_visible)
:width: 95%

**Alignment (recommended dielectric).** Peak-normalized PlasMol hybrid Fourier spectra (parallel and perpendicular) versus the quasistatic Gersten–Nitzan model with $\mathrm{Au\_JC\_visible}$ permittivity, host index $n_{\mathrm{host}}=1.33$, multipoles through $\ell_{\max}=25$, and joint oscillator parameters $\omega_m\approx 2.055\,\mathrm{eV}$, $\gamma_m\approx 0.076\,\mathrm{eV}$, $\alpha_0\approx 0.43\,\mathrm{nm}^3$. Parallel and perpendicular peak positions agree to within a few tens of meV; average mean-squared error over both orientations is $\sim 10^{-3}$ on the normalized spectra.
```

```{figure} assets/spectrum_model2_joint_fit_Au.png
:alt: Joint quasistatic fit using Rakić Au dielectric
:width: 95%

**Alignment (alternate dielectric).** Same geometry and joint-fit protocol with Rakić bulk $\mathrm{Au}$. The model again reproduces the PlasMol parallel and perpendicular hybrid spectra, confirming that the correspondence is not an artifact of a single tabulated $\varepsilon_m(\omega)$.
```

These comparisons support several methodological claims:

1. **Vacuum deconvolution is the correct hybrid counterpart of $\mathbf{p}/E_{\mathrm{inc}}$.** Using bare $E_{\mathrm{inc}}$ (rather than the local hybrid field) in the denominator matches the definition of $\alpha_{\mathrm{eff}}$ relative to the incident wave.
2. **Orientation-resolved Fourier modes are necessary.** A single isotropic $x{+}y{+}z$ average would mix the inequivalent $G_\parallel,S_\parallel$ and $G_\perp,S_\perp$ channels that the analytic model treats separately—and that PlasMol resolves with `parallel` / `perpendicular` polarization.
3. **Local-field enhancement and (weak) back-action are captured.** Fits improve when the classical $G$ factors and the $1-\alpha_m S$ denominator are retained, consistent with PlasMol recording a fully hybrid $\mu(t)$ while still normalizing to vacuum $E_{\mathrm{inc}}$.
4. **The hybrid time loop is spectrally consistent** with a known continuum electrodynamics limit in the quasistatic regime, giving an independent check before applying PlasMol to regimes (core holes, strong molecular structure, etc.) where no closed analytic solution exists.

Full formulas, multipole series, and the domain of validity of the analytic model are in [Quasistatic Model](quasistatic_model.md). Fourier methodology (including ∥/⊥ construction) is in [Fourier Spectra](fourier.md).

---

## Appendix A1: RT-TDDFT propagation

The density matrix obeys the Liouville–von Neumann equation adapted to the non-interacting KS system,

$$
\mathrm{i}\,\frac{\mathrm{d}\mathbf{D}(t)}{\mathrm{d}t}
=
\bigl[\mathbf{F}(t),\,\mathbf{D}(t)\bigr],
$$

where $\mathbf{F}(t)$ is the time-dependent Fock matrix embodying $\hat{H}_{\mathrm{KS}}(t)$. Time is discretized in steps $\Delta t$, and the change in $\mathbf{D}$ over each step is obtained by a numerical propagator. Many techniques exist; the **second-order Magnus** propagator is a standard balance of accuracy and cost, and is PlasMol’s default.

The exact time-evolution operator for the TDKS equations over a step from $t_n$ to $t_{n+1}=t_n+\Delta t$ is the time-ordered exponential

$$
U(t_{n+1},t_n)
=
\hat{T}\,
\exp\!\left(
  -\mathrm{i}
  \int_{t_n}^{t_{n+1}}
  F(t')\,\mathrm{d}t'
\right),
$$

with orbital propagation $\phi_i(t_{n+1})=U(t_{n+1},t_n)\,\phi_i(t_n)$. Because $F(t)$ is not known continuously between steps, the time-ordered exponential is approximated by an ordinary exponential of an effective Magnus operator $\Omega$:

$$
U(t_{n+1},t_n)
\approx
\exp\!\bigl(\Omega(t_{n+1},t_n)\bigr).
$$

The second-order Magnus approximation truncates the series and evaluates the integral by the midpoint rule:

$$
\Omega^{[2]}
=
-\mathrm{i}\,\Delta t\;
F\!\left(t_n+\frac{\Delta t}{2}\right).
$$

The propagation step becomes $\phi_i(t_{n+1})=\exp(\Omega^{[2]})\,\phi_i(t_n)$. The matrix exponential may be evaluated by diagonalization (small systems), Padé approximants, or scaling-and-squaring.

Because part of $F\!\left(t_n+\frac{\Delta t}{2}\right)$ depends on the unknown density $\rho\!\left(t_n+\frac{\Delta t}{2}\right)$, a **predictor–corrector** scheme is required for self-consistency. A common implementation is:

1. **Initial extrapolation.** Predict the midpoint Hamiltonian by linear extrapolation from the previous step:

   $$
   F^{(0)}\!\left(t_n+\frac{\Delta t}{2}\right)
   =
   2F(t_n)
   -
   F\!\left(t_{n-1}+\frac{\Delta t}{2}\right),
   
   $$

   where $F\!\left(t_{n-1}+\frac{\Delta t}{2}\right)$ is the converged midpoint Hamiltonian from the previous step.

2. **Iterative predictor–corrector.**

   (a) **Predictor.** Propagate the orbitals with the $k$th midpoint estimate:

   $$
   \phi_i^{(k+1)}(t_{n+1})
   =
   \exp\!\left(
     -\mathrm{i}\,\Delta t\;
     F^{(k)}\!\left(t_n+\frac{\Delta t}{2}\right)
   \right)
   \phi_i(t_n).
   $$

   (b) **Predicted density:**

   $$
   \rho^{(k+1)}(t_{n+1})
   =
   \sum_{i=1}^{N}
   \bigl|\phi_i^{(k+1)}(t_{n+1})\bigr|^2.
   $$

   \(c) **Predicted Fock matrix at $t_{n+1}$:**

   $$
   F^{(k+1)}(t_{n+1})
   =
   -\frac{\nabla^2}{2m}
   +
   v_{\mathrm{H}}[\rho^{(k+1)}](\mathbf{r},t_{n+1})
   +
   v_{\mathrm{xc}}[\rho^{(k+1)}](\mathbf{r},t_{n+1})
   +
   v_{\mathrm{ext}}(\mathbf{r},t_{n+1}).
   $$

   (d) **Corrector.** Update the midpoint Hamiltonian as the average

   $$
   F^{(k+1)}\!\left(t_n+\frac{\Delta t}{2}\right)
   \approx
   \frac{1}{2}
   \Bigl[
     F(t_n)
     +
     F^{(k+1)}(t_{n+1})
   \Bigr].
   $$

   (e) **Convergence.** If
   $\bigl\|\phi^{(k+1)}(t_{n+1})-\phi^{(k)}(t_{n+1})\bigr\|<\epsilon$
   (user tolerance), accept
   $F\!\left(t_n+\frac{\Delta t}{2}\right)=F^{(k+1)}\!\left(t_n+\frac{\Delta t}{2}\right)$;
   otherwise continue iterating.

Through this procedure the orbitals advance one time step. For larger systems, a larger $\Delta t$ saves cost; a fourth-order Magnus propagator can allow larger steps without sacrificing accuracy, but second-order Magnus is sufficient for systems typically studied with PlasMol. PlasMol also exposes **RK4** and simple **step** propagators as alternatives.

---

## Appendix A2: Producing an absorption spectrum in RT-TDDFT

RT-TDDFT propagates the density matrix under a time-dependent perturbation. To survey many excited states without probing each resonance individually, one excites the molecule with a **broadband $\delta$-like kick**. In the time domain the ideal kick is a Dirac delta $\delta(t)$; its Fourier transform is flat in frequency, so all frequencies $\omega$ are excited with equal amplitude.

In practice, three simulations are often run (or one combined workflow), each with a kick along a Cartesian direction $\{x,y,z\}$, and allowed to propagate for a long total time $T$. Longer $T$ improves frequency resolution ($\Delta\omega\sim 2\pi/T$). Because of numerical stability, the kick is limited to a small intensity (e.g. $\kappa\sim 10^{-4}\,\mathrm{a.u.}$) and a single time step width $\Delta t$:

$$
v_{\mathrm{ext},i}
=
\begin{cases}
-\kappa\,r_i, & \text{at the kick step, } i\in\{x,y,z\}, \\
0, & \text{otherwise}
\end{cases}
$$

(same dipole coupling as in the hybrid section).

At each time step the induced dipole $\mu_{\mathrm{ind},i}$ is recorded via the induced-dipole formula above. After propagation, each Cartesian component is Fourier-transformed:

$$
\tilde{\mu}_{\mathrm{ind},i}(\omega)
=
\int_{-\infty}^{\infty}
\mu_{\mathrm{ind},i}(t)\,
\mathrm{e}^{-\mathrm{i}\omega t}\,
\mathrm{d}t.
$$

When the kick is applied along direction $i$, only the induced dipole along $i$ is needed for that run. Once all three components are available, the absorption cross-section (isotropic average) is

$$
S(\omega)
=
\frac{4\pi\omega}{3c\kappa}\,
\operatorname{Im}\!
\Bigl[
  \tilde{\mu}_{\mathrm{ind},x}(\omega)
  +
  \tilde{\mu}_{\mathrm{ind},y}(\omega)
  +
  \tilde{\mu}_{\mathrm{ind},z}(\omega)
\Bigr],
$$

where $c$ is the speed of light. $S(\omega)$ yields a stick-like spectrum of discrete peaks.

**Artificial (post-processing) broadening** damps the time signal before the transform:

$$
\tilde{\mu}_{\mathrm{ind},i}(\omega)
=
\int_{-\infty}^{\infty}
\mu_{\mathrm{ind},i}(t)\,
\mathrm{e}^{-\mathrm{i}(\omega+\mathrm{i}\gamma)t}\,
\mathrm{d}t,
$$

with a global damping rate $\gamma$. This is distinct from the **state-specific** non-Hermitian CAP described above, which acts *during* propagation and can reproduce continuum shoulders more physically.

PlasMol’s production Fourier workflow implements these ideas (with optional vacuum $E_{\mathrm{inc}}$ deconvolution for hybrid NP–molecule runs, and parallel/perpendicular polarization modes). See [Fourier Spectra](fourier.md) for the methodology and [Quasistatic Model](quasistatic_model.md) for the analytic $\alpha_{\mathrm{eff}}$ benchmark (with alignments discussed in [Validation](#validation-against-an-analytically-soluble-model)).

---

## Selected references

The following are key sources underlying the theory above.

1. K. Lopata and N. Govind, “Near and above ionization electronic excitations with non-hermitian real-time time-dependent density functional theory,” *J. Chem. Theory Comput.* **9**, 4939–4946 (2013).
2. E. Runge and E. K. U. Gross, “Density-functional theory for time-dependent systems,” *Phys. Rev. Lett.* **52**, 997–1000 (1984).
3. R. G. Fernando, M. C. Balhoff, and K. Lopata, “X-ray absorption in insulators with non-hermitian real-time time-dependent density functional theory,” *J. Chem. Theory Comput.* **11**, 646–654 (2015).
4. S. Hirata and M. Head-Gordon, “Time-dependent density functional theory within the Tamm–Dancoff approximation,” *Chem. Phys. Lett.* **314**, 291–299 (1999).
5. A. Castro, M. A. L. Marques, and A. Rubio, “Propagators for the time-dependent Kohn–Sham equations,” *J. Chem. Phys.* **121**, 3425–3433 (2004).
6. S. Blanes, F. Casas, J. Oteo, and J. Ros, “The Magnus expansion and some of its applications,” *Phys. Rep.* **470**, 151–238 (2009).
