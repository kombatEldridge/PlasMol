# Quasistatic Model (Gersten–Nitzan)

These notes summarize the **analytically soluble quasistatic model** used to interpret PlasMol hybrid absorption spectra: a point-like molecule next to a dielectric (or metallic) sphere acquires an *effective* polarizability relative to the incident field $\mathbf{E}_{\mathrm{inc}}$.

$$
\mathbf{p}
=
\alpha_{\mathrm{eff}}(\omega)\,\mathbf{E}_{\mathrm{inc}}.
$$

The construction is the classical Gersten–Nitzan / image-multipole framework. Full derivation notes live in the repository under `jobs/model2_quasistatic_sphere/effective_polarizability_derivation.pdf` (source: `.tex`). This page is **math and methodology only**; comparison of the model to PlasMol hybrid Fourier spectra is discussed in [Theory & Methodology](../methodology.md#validation-against-an-analytically-soluble-model).

---

## Geometry and notation

| Symbol | Meaning |
| -------- | --------- |
| $a$ | Sphere radius |
| $d$ | Distance from sphere center to the molecular point dipole ($d > a$) |
| $\varepsilon_m(\omega)$ | Relative permittivity of the nanoparticle |
| $\varepsilon_b$ | Host relative permittivity (real; e.g. water $\varepsilon_b = n^2$) |
| $\alpha_m(\omega)$ | Bare molecular polarizability **volume** |
| $\parallel$ | Radial: $\mathbf{p} \parallel \hat{\mathbf{r}}$ (NP $\to$ molecule) |
| $\perp$ | Tangential: $\mathbf{p} \perp \hat{\mathbf{r}}$ |

Polarizabilities are written as **volumes** (cgs-like convention), so $\mathbf{p}=\alpha\mathbf{E}$ without a $4\pi\varepsilon_0$ factor. The quasistatic assumption is $d\ll\lambda$ (no retardation).

---

## Bare molecular polarizability (Lorentz oscillator)

A single classical resonance for the molecule is taken as

$$
\alpha_m(\omega)
=
\alpha_0
\frac{\omega_m^2}{\omega_m^2 - \omega^2 - \mathrm{i}\gamma_m\omega},
$$

with transition frequency $\omega_m$, linewidth $\gamma_m$, and static polarizability volume $\alpha_0$.

---

## Sphere multipoles

The $\ell$-pole polarizability of a sphere of radius $a$ in a host of permittivity $\varepsilon_b$ is

$$
\alpha_{\mathrm{NP}}^{(\ell)}(\omega)
=
a^{2\ell+1}
\frac{\varepsilon_m(\omega)-\varepsilon_b}
     {\varepsilon_m(\omega)+(\ell+1)\varepsilon_b/\ell},
\qquad \ell = 1,2,\ldots
$$

For $\ell=1$ this is the usual Mie dipole polarizability of a sphere in a host (quasistatic limit).

---

## Dipole–dipole interaction

In the quasistatic limit the field of a point dipole at distance $d$ has principal values along the NP–molecule axis:

$$
M_\parallel = \frac{2}{d^3},
\qquad
M_\perp = -\frac{1}{d^3}.
$$

These couple the molecular dipole to the induced multipoles on the sphere (and vice versa).

---

## Gersten–Nitzan elimination

Coupled equations for the molecule and NP dipoles, after eliminating the NP response in favor of $\mathbf{E}_{\mathrm{inc}}$, yield the working formula

$$
\alpha_{\mathrm{eff}}
=
\frac{\alpha_m G}{1 - \alpha_m S},
$$

with **local-field factor** $G$ and **image / self-interaction propagator** $S$:

$$
G \equiv 1 + M\alpha_2,
\qquad
S \equiv M^2\alpha_2.
$$

Here $\alpha_2$ stands for the relevant NP multipole response projected onto the interaction channel (dipole-only or multipolar series below). The denominator $1-\alpha_m S$ encodes classical **back-action**: the field scattered by the sphere acts back on the molecule.

### Point-dipole sphere ($\alpha_2=\alpha_{\mathrm{NP}}^{(1)}$)

**Parallel (radial)**

$$
G_\parallel
=
1 + 2\frac{\alpha_{\mathrm{NP}}^{(1)}}{d^3},
\qquad
S_\parallel^{(\ell=1)}
=
4\frac{\alpha_{\mathrm{NP}}^{(1)}}{d^6}.
$$

**Perpendicular (tangential)**

$$
G_\perp
=
1 - \frac{\alpha_{\mathrm{NP}}^{(1)}}{d^3},
\qquad
S_\perp^{(\ell=1)}
=
\frac{\alpha_{\mathrm{NP}}^{(1)}}{d^6}.
$$

### Multipolar image series

For small molecules often $\alpha_m S\ll 1$ and even $S\approx 0$ is a useful first approximation. When higher multipoles of the sphere are retained,

$$
S_\parallel
=
\sum_{\ell=1}^{\ell_{\max}}
(\ell+1)^2
\frac{\alpha_{\mathrm{NP}}^{(\ell)}}{d^{2\ell+4}},
\qquad
S_\perp
=
\sum_{\ell=1}^{\ell_{\max}}
\frac{\ell(\ell+1)}{2}
\frac{\alpha_{\mathrm{NP}}^{(\ell)}}{d^{2\ell+4}}.
$$

Local-field factors $G_{\parallel,\perp}$ may likewise be extended beyond the pure dipole ($\ell=1$) truncation when consistent with the multipolar image series.

---

## Absorption proxy

Relative to the incident field $\mathbf{E}_{\mathrm{inc}}$, the classical absorption proxy is

$$
A(\omega)
\propto
-\omega\,\operatorname{Im}\,\alpha_{\mathrm{eff}}(\omega).
$$

Orientation-resolved channels use $\alpha_{\mathrm{eff},\parallel}$ and $\alpha_{\mathrm{eff},\perp}$ separately. This is the analytical counterpart of PlasMol’s hybrid Fourier spectrum based on

$$
\mathcal{R}(\omega)
=
\frac{\mu(\omega)}{E_{\mathrm{inc}}(\omega)},
\qquad
A(\omega)
\propto
-\omega\,\operatorname{Im}\,\mathcal{R}(\omega)
$$

in the [Fourier Spectra](fourier.md) methodology (parallel and perpendicular modes map onto $G_\parallel,S_\parallel$ and $G_\perp,S_\perp$).

---

## Domain of validity

- **Quasistatic:** $d\ll\lambda$; no retardation or radiation reaction beyond the local model.
- **Local bulk dielectric** $\varepsilon_m(\omega)$ for the NP; no nonlocal or quantum-size corrections to the metal.
- **Point molecule** with a single Lorentz oscillator; no spatial extent or molecular multipoles.
- **Linear classical response**; host $\varepsilon_b$ real and frequency-independent in the usual application.

Within that domain the model is **analytically soluble** once $\varepsilon_m(\omega)$, geometry $(a,d)$, and oscillator parameters $(\omega_m,\gamma_m,\alpha_0)$ are fixed—making it a controlled benchmark for hybrid FDTD–RT-TDDFT spectra.

---

## See also

- [Theory & Methodology](../methodology.md#validation-against-an-analytically-soluble-model) — alignment of PlasMol hybrid spectra with this model
- [Fourier Spectra](fourier.md) — hybrid $\mu/E_{\mathrm{inc}}$ and parallel / perpendicular modes
- [Theory & Methodology](../methodology.md) — self-consistent FDTD–RT-TDDFT loop
