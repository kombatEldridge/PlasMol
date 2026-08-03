# Fourier Absorption Spectra

This page describes the **mathematics and methodology** of PlasMol’s Fourier absorption workflow: how an induced dipole is transformed into a spectrum, how quantum-only and hybrid drives differ, and why parallel / perpendicular polarizations are treated separately near a nanoparticle.

For runnable inputs, CLI usage, and parameter tables, see [Simulations: Fourier](../simulations/fourier.md), [Usage](../usage.md), and [Tutorials](../tutorials.md).

---

## Physical overview

### Induced dipole

The Cartesian induced dipole is the change in the expectation value of the dipole operator relative to the initial (usually ground-state) density:

$$
\mu_i(t)
=
\operatorname{Tr}\!\big[
  \hat{\mu}_i \big(\mathbf{D}(t) - \mathbf{D}_0\big)
\big],
\qquad i \in \{x,y,z\}.
$$

Open-shell (UKS) runs sum α and β density blocks before the trace. In hybrid FDTD–RT-TDDFT, $\boldsymbol{\mu}(t)$ is recorded while the molecule is driven by the local Meep field at the molecular site (see [Theory & Methodology](../methodology.md)).

### Response function and absorption

A broadband excitation is applied; the time series $\mu_i(t)$ is Fourier-transformed to $\mu_i(\omega)$. The spectral response used in the absorption formula depends on the drive:

| Regime | Drive | Response $\mathcal{R}_i(\omega)$ |
| -------- | ------- | ---------------------------------- |
| Quantum-only | Ideal $\delta$-kick on the molecule | $\mathcal{R}_i(\omega)=\mu_i(\omega)$ |
| Hybrid (Meep) | Broadband Gaussian pulse through the cell | $\mathcal{R}_i(\omega)=\mu_i(\omega)/E_{\mathrm{inc},i}(\omega)$ |

Here $E_{\mathrm{inc},i}(\omega)$ is the Fourier transform of the **vacuum** incident field at the molecular site (same source and cell geometry, no NP and no molecule). Dividing by $E_{\mathrm{inc}}$ removes the source spectral envelope. Local-field and NP enhancement remain inside $\mu$, so $\mathcal{R}$ is an *effective*, polarizability-like response—not the bare molecular $\alpha_m$. See [Quasistatic Model](quasistatic_model.md).

The isotropic (three-direction) absorption-like spectrum is

$$
A(\omega)
=
-\frac{4\pi\omega}{3c}
\sum_{i\in\{x,y,z\}}
\operatorname{Im}\!\big[\mathcal{R}_i(\omega)\big],
$$

with $c$ the speed of light in atomic units. For a single Cartesian polarization $i$,

$$
A_i(\omega)
=
-\frac{4\pi\omega}{c}\,
\operatorname{Im}\!\big[\mathcal{R}_i(\omega)\big].
$$

Frequencies come from the real FFT grid. For display they are converted to electronvolts via $E[\mathrm{eV}]=\omega[\mathrm{a.u.}]\times 27.211386$ and restricted to a chosen energy window. Spectra are typically peak-normalized before plotting.

---

## Common spectral post-processing

Regardless of quantum-only vs hybrid, the discrete path is:

1. **Optional damping of $\mu(t)$** (and of $E_{\mathrm{inc}}(t)$ when deconvolving), e.g.
$$
w(t)=e^{-\gamma t}
\quad\text{and/or}\quad
e^{-t/\tau},
$$
   which implements artificial (global) Lorentzian-like broadening, distinct from the non-Hermitian CAP of the main methodology page.
2. **Discrete Fourier transform** (up to normalization conventions in the code),
$$
S_{\mu,i}(\omega)
=
\Delta t\sum_n
\mu_i(t_n)\,w(t_n)\,e^{-\mathrm{i}\omega t_n},
$$
   and likewise for $E_{\mathrm{inc},i}$ when needed.
3. **Assemble** $\mathcal{R}_i$ and $A(\omega)$ or $A_i(\omega)$ as above.
4. Optionally apply a spectral floor on $|E_{\mathrm{inc}}|$ before division to suppress numerical noise in the hybrid path.

---

## Quantum-only methodology ($\delta$-kicks)

With only a molecular electronic-structure problem (no classical nanoparticle / FDTD cell), three independent RT-TDDFT trajectories are driven by a short kick along $x$, $y$, and $z$. An ideal Dirac kick has a flat spectrum in frequency, so the response collapses to the dipole itself:

$$
\mathcal{R}_i(\omega)=\mu_i(\omega).
$$

The isotropic spectrum is then the three-direction average of $\operatorname{Im}\mu_i(\omega)$ in the formula for $A(\omega)$ above. This is the same conceptual route as [Appendix A2 of the main methodology page](../methodology.md#appendix-a2-producing-an-absorption-spectrum-in-rt-tddft), with PlasMol’s discrete FFT and optional $\gamma,\tau$ windows applied in post-processing.

---

## Hybrid methodology — full (isotropic) polarization

When both a classical electromagnetic cell and a molecule are present, a true $\delta$-kick is not natural for the FDTD source. PlasMol instead uses a **broadband Gaussian** (or similar) Meep source and recovers a comparable response by **vacuum deconvolution**.

### Production and reference fields

For each Cartesian direction $i\in\{x,y,z\}$:

1. **Production trajectory** — Meep cell with nanoparticle (optional) and molecule; record $\mu_i(t)$ (and typically the local $\mathbf{E}$ at the molecule). The local field that drives RT-TDDFT is the full hybrid field at the molecular site.
2. **Vacuum reference trajectory** — same source and cell geometry, **no nanoparticle and no molecule**; record the incident field $E_{\mathrm{inc},i}(t)$ at the molecular site only.
3. **Deconvolution**
$$
\mathcal{R}_i(\omega)
=
\frac{\mu_i(\omega)}{E_{\mathrm{inc},i}(\omega)}.
$$

The isotropic hybrid spectrum uses the three-direction sum in $A(\omega)$. Physically:

- The **denominator** normalizes out the temporal/spectral shape of the classical source.
- The **numerator** still contains molecule–plasmon coupling, local-field enhancement, and (if enabled) dipole back-action into the FDTD cell.

### Hybrid field path during production

At each Meep step for which the field at the molecule exceeds the sampling threshold:

1. Sample $\mathbf{E}(t)$ at the molecular position.
2. Advance RT-TDDFT one step under $v_{\mathrm{ext}}=-\mathbf{r}\cdot\mathbf{E}(t)$.
3. Optionally inject $\boldsymbol{\mu}$ back into Meep as a polarization source (back-propagation).

Only the recorded $\mu_i(t)$ and vacuum $E_{\mathrm{inc},i}(t)$ enter the spectral deconvolution. Local-field physics is therefore encoded in $\mu$, not in the denominator.

---

## Parallel and perpendicular methodology

Near a spherical nanoparticle the hybrid response is **anisotropic**: radial (parallel, ∥) and tangential (perpendicular, ⊥) polarizations couple differently to the plasmon. Averaging $x+y+z$ mixes inequivalent axes once an NP–molecule geometry is fixed. Orientation-resolved hybrid spectra therefore restrict the drive and analysis to a single Cartesian lab component $c$ aligned with either $\mathbf{r}$ or a direction orthogonal to $\mathbf{r}$.

### Geometry and axes

Define nanoparticle center $\mathbf{r}_{\mathrm{NP}}$ and molecular sample point $\mathbf{r}_{\mathrm{mol}}$. The connecting axis is

$$
\mathbf{r}
=
\mathbf{r}_{\mathrm{mol}}-\mathbf{r}_{\mathrm{NP}},
\qquad
\hat{\mathbf{r}}
=
\mathbf{r}/|\mathbf{r}|.
$$

| Mode | Field orientation | Role |
| ------ | ------------------- | ------ |
| Full | Three Cartesian drives, isotropic average | Orientation-averaged hybrid response |
| Parallel | $\mathbf{E}$ along $\hat{\mathbf{r}}$ (radial) | Couples strongly to the plasmon gap/radial mode |
| Perpendicular | $\mathbf{E}\perp\hat{\mathbf{r}}$ (tangential) | Orthogonal Gersten–Nitzan channel |

In practice the continuous direction $\hat{\mathbf{r}}$ is mapped onto the nearest lab axis $\hat{\mathbf{e}}_c\in\{\hat{\mathbf{x}},\hat{\mathbf{y}},\hat{\mathbf{z}}\}$ for parallel mode, and onto a lab axis most orthogonal to $\hat{\mathbf{r}}$ (or a user-chosen axis) for perpendicular mode. Geometry is preferred when the molecule lies on a Cartesian axis relative to the NP so that $|\hat{\mathbf{r}}\cdot\hat{\mathbf{e}}_c|\approx 1$.

### Transverse plane-wave drive

Hybrid Fourier sources are planar faces representing a plane wave with propagation $\mathbf{k}$ **perpendicular** to the polarization $\mathbf{E}$. If the face normal would be parallel to $\mathbf{E}$ (longitudinal injection), the source face is rearranged so that $\mathbf{k}\perp\mathbf{E}$. That keeps the classical drive consistent with a transverse plane wave for the chosen polarization component $c$.

### Single-polarization spectrum

Only one production trajectory and one vacuum reference are required. With component $c$,

$$
\mathcal{R}_c(\omega)
=
\frac{\mu_c(\omega)}{E_{\mathrm{inc},c}(\omega)},
\qquad
A_c(\omega)
=
-\frac{4\pi\omega}{c}\,
\operatorname{Im}\,\mathcal{R}_c(\omega).
$$

(Here the second $c$ in the prefactor is the speed of light; the subscript $c$ labels the Cartesian component.)

### Relation to Gersten–Nitzan effective polarizability

Parallel and perpendicular hybrid Fourier spectra map onto the classical channels of the Gersten–Nitzan model:

| Hybrid spectrum | Classical factors |
| ----------------- | ------------------- |
| Parallel $A_\parallel(\omega)$ | $G_\parallel$, $S_\parallel$ |
| Perpendicular $A_\perp(\omega)$ | $G_\perp$, $S_\perp$ |

The quasistatic $\alpha_{\mathrm{eff}}$ construction and its relation to these spectra are in [Quasistatic Model](quasistatic_model.md); numerical alignments with PlasMol are summarized in [Theory & Methodology — Validation](../methodology.md#validation-against-an-analytically-soluble-model).

---

## Summary of the three regimes

| Regime | What is driven | What is divided out | Spectrum |
| -------- | ---------------- | --------------------- | ---------- |
| Quantum-only | $\delta$-kick on molecule | — (flat source) | $A(\omega)$ from $\mu_x,\mu_y,\mu_z$ |
| Hybrid full | Broadband Meep, 3 axes | Vacuum $E_{\mathrm{inc},i}$ each axis | Isotropic $A(\omega)$ |
| Hybrid ∥ or ⊥ | Broadband Meep, one axis $c$ | Vacuum $E_{\mathrm{inc},c}$ | $A_c(\omega)$ |

All three share the same underlying objects—$\boldsymbol{\mu}(t)$, optional damping windows, discrete Fourier transforms, and absorption from $\operatorname{Im}\mathcal{R}(\omega)$—and differ only in the definition of $\mathcal{R}$ and in which polarizations are retained.

---

## See also

- [Theory & Methodology](../methodology.md) — hybrid time loop, RT-TDDFT, CAP, and spectrum appendix
- [Quasistatic Model](quasistatic_model.md) — Gersten–Nitzan $G,S$ and $\alpha_{\mathrm{eff}}$
- [Core-Hole Dynamics](core_hole.md) — sudden SCH/DCH (separate from Fourier linear absorption)
- [Simulations: Fourier](../simulations/fourier.md) — driver usage and inputs
- [Usage](../usage.md) — full JSON schema
