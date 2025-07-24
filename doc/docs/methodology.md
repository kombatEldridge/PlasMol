# Theory and Methodology of PlasMol

PlasMol is a hybrid simulation tool designed to model plasmon-molecule interactions by coupling classical Finite-Difference Time-Domain (FDTD) electromagnetics with quantum Real-Time Time-Dependent Density Functional Theory (RT-TDDFT). This document outlines the theoretical foundations and methodological workflow of PlasMol, drawing from its codebase and the provided schematic. The schematic illustrates the integration between quantum mechanical (QM) software, a handler for coupling, and classical FDTD (e.g., MEEP for electromagnetic propagation).

## Introduction

PlasMol enables three simulation modes:

- **Classical Only**: FDTD simulation of nanoparticles (NPs) using MEEP.
- **Quantum Only**: RT-TDDFT simulation of molecules.
- **Full PlasMol**: Coupled FDTD-RT-TDDFT for NP-molecule systems.

The core innovation lies in the bidirectional coupling: the electric field from the classical simulation drives quantum propagation, while the induced molecular dipole feeds back into the classical field as a point source. This allows modeling phenomena like plasmon-enhanced spectroscopy or Surface-Enhanced Raman Scattering (SERS).

The schematic highlights:

- QM side: Time-dependent Fock matrix and integrals.
- Handler: Density matrix propagation and dipole calculation.
- MEEP side: Maxwell's equations with polarization source.

## Classical Component: FDTD with MEEP

The classical part simulates electromagnetic fields around NPs using MEEP's FDTD solver. Key equations from the schematic (right side) describe field propagation:

\[
\nabla \times \tilde{\mathbf{E}}(t) = -\mu \frac{\partial \tilde{\mathbf{H}}(t)}{\partial t}
\]

\[
\nabla \times \tilde{\mathbf{H}}(t) = \epsilon \frac{\partial \tilde{\mathbf{E}}(t)}{\partial t} + \frac{\partial \tilde{\mathbf{P}}(t)}{\partial t}
\]

Here, \(\tilde{\mathbf{E}}\) and \(\tilde{\mathbf{H}}\) are the electric and magnetic fields, \(\epsilon\) and \(\mu\) are permittivity and permeability, and \(\tilde{\mathbf{P}}(t)\) is the polarization (from quantum feedback in the full PlasMol mode).

### Supported Features
- **Sources**: Continuous, Gaussian, chirped, or pulsed waves (see `src/classical/sources.py`).
- **Geometry**: Spheres (Au/Ag materials); extensible to other shapes.
- **Boundaries**: Perfectly Matched Layers (PML).
- **Outputs**: HDF5 images, GIFs, field CSVs; custom tracking (e.g., extinction spectra) via injections in `src/classical/simulation.py`.

In the full PlasMol mode, the induced dipole \(\tilde{\mathbf{P}}(t)\) is injected as a custom source at the molecule's position.

## Quantum Component: RT-TDDFT with PySCF

The quantum part computes molecular responses using RT-TDDFT in PySCF. The time-dependent Fock matrix (from the schematic, left side) is:

\[
F_{\mu\nu}(t) = T_{\mu\nu} + V_{\mu\nu} + J_{\mu\nu}(t) - K_{\mu\nu}(t) - \sum_{a \in x,y,z} \mu_a \tilde{E}_a(t)
\]

Where:

- \(T_{\mu\nu}\): Kinetic energy integral:

\[
T_{\mu\nu} = \int \phi_\mu(\mathbf{r}) \left( -\frac{1}{2} \nabla^2 \right) \phi_\nu(\mathbf{r}) \, d\mathbf{r}
\]

- \(V_{\mu\nu}\): Nuclear attraction integral:

\[
V_{\mu\nu} = \int \phi_\mu(\mathbf{r}) \left( -\sum_A \frac{Z_A}{|\mathbf{R}_A - \mathbf{r}|} \right) \phi_\nu(\mathbf{r}) \, d\mathbf{r}
\]

- \(J_{\mu\nu}(t)\): Coulomb integral (time-dependent via density):

\[
J_{\mu\nu} = \sum_{\lambda\sigma} D_{\lambda\sigma} \iint \phi^*_\mu(\mathbf{r}_1) \phi_\nu(\mathbf{r}_1) \left( \frac{1}{r_{12}} \right) \phi^*_\sigma(\mathbf{r}_2) \phi_\sigma(\mathbf{r}_2) \, d\mathbf{r}_1 d\mathbf{r}_2
\]

- \(K_{\mu\nu}(t)\): Exchange integral (similarly time-dependent):

\[
K_{\mu\nu} = \frac{1}{2} \sum_{\lambda\sigma} D_{\lambda\sigma} \iint \phi^*_\mu(\mathbf{r}_1) \phi_\lambda(\mathbf{r}_1) \left( \frac{1}{r_{12}} \right) \phi^*_\sigma(\mathbf{r}_2) \phi_\nu(\mathbf{r}_2) \, d\mathbf{r}_1 d\mathbf{r}_2
\]

The external field term \(-\sum \mu_a \tilde{E}_a(t)\) couples to the classical field.

### Density Matrix Evolution
The density matrix \(\mathbf{D}(t)\) evolves under the Liouville-von Neumann equation (schematic, handler):

\[
\frac{\partial \mathbf{D}(t)}{\partial t} = [ \mathbf{F}(t), \mathbf{D}(t) ]
\]

In practice, PlasMol propagates molecular orbitals \(\mathbf{C}(t)\) in an orthogonal basis and reconstructs \(\mathbf{D}(t)\).

### Supported Propagators
PlasMol offers three methods (see [Propagators](api-reference.md#quantumpropagators) for more details):

1. **Step (Modified Midpoint Unitary Transformation - MMUT)**:

\[
\mathbf{C}_{\text{ortho}}(t + \Delta t) = \exp(-i \cdot 2\Delta t \cdot \mathbf{F}_{\text{ortho}}(t)) \cdot \mathbf{C}_{\text{ortho}}(t - \Delta t)
\]

2. **Runge-Kutta 4 (RK4)**:

\[
\mathbf{C}_{\text{ortho}}(t + \Delta t) = \mathbf{C}_{\text{ortho}}(t) + \frac{k_1 + 2k_2 + 2k_3 + k_4}{6}
\]

With intermediate \(k_i\) terms computed via Fock matrix multiplications.

3. **2nd-Order Magnus with Predictor-Corrector** (Recommended):

Initial extrapolation:

\[
\mathbf{F}_{\text{ortho}}^{(0)}(t + \Delta t / 2) = 2 \mathbf{F}_{\text{ortho}}(t) - \mathbf{F}_{\text{ortho}}(t - \Delta t / 2)
\]

Iterative update until convergence:

\[
\mathbf{C}_{\text{ortho}}^{(k+1)}(t + \Delta t) = \exp(-i \Delta t \cdot \mathbf{F}_{\text{ortho}}^{(k)}(t + \Delta t / 2)) \cdot \mathbf{C}_{\text{ortho}}(t)
\]

From the schematic (handler, propagation method):

\[
\tilde{\mathbf{P}}(t + dt) = \frac{3}{4 \tau a^3} \operatorname{Tr}[\mathbf{D}(t + dt) \cdot \mu] - \operatorname{Tr}[\mathbf{D}_0 \cdot \mu]
\]

This induced polarization \(\tilde{\mathbf{P}}(t)\) is fed back to MEEP.

### Absorption Spectra
With the `transform` flag, PlasMol runs three directional simulations and applies a Fourier transform (see `src/utils/fourier.py`).

## Coupling Mechanism: Handler

The "Handler" bridges QM and classical parts (schematic, center):

- **QM to Classical**: Induced dipole from RT-TDDFT is injected as a point source in MEEP.
- **Classical to QM**: Electric field at the molecule's position drives Fock matrix updates.
- Workflow (per time step in `src/classical/simulation.py` and `src/quantum/propagation.py`):
    1. (Before simulation starts to loop) Generate ground state matrices using PySCF.
    2. Extract \(\tilde{\mathbf{E}}(t)\) from MEEP at molecule position.
    3. Propagate \(\mathbf{D}(t)\) using chosen method.
    4. Compute \(\tilde{\mathbf{P}}(t)\) via dipole contraction.
    5. Update MEEP source with \(\partial \tilde{\mathbf{P}} / \partial t\).

This loop enables self-consistent hybrid dynamics.

## Methodology Workflow

1. **Input Parsing**: `src/input/` processes blocks for classical/quantum parameters.
2. **Initialization**:
    - Classical: Build MEEP simulation (`src/classical/simulation.py`).
    - Quantum: Build molecule and initial state (`src/quantum/molecule.py`).
3. **Simulation Loop**:
    - Advance FDTD step in MEEP.
    - If molecule present and field exceeds cutoff: Call RT-TDDFT propagation.
    - Inject dipole back as source.
4. **Outputs**: CSVs, plots, checkpoints via `src/utils/`.
5. **Extensions**: Custom injections for tracking (e.g., SERS) in commented sections.

For code details, see [API Reference](api-reference.md).