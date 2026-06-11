# Theory and Methodology of PlasMol

PlasMol performs **self-consistent hybrid FDTD–RT-TDDFT** simulations of plasmon-molecule systems. The classical electromagnetic fields (Meep) drive the quantum time propagation (PySCF), and the resulting induced molecular dipole is fed back into the classical simulation as a point source.

## High-Level Workflow (per time step)

1. **Classical advance** (Meep) 
      - Update E and H fields according to Maxwell's equations with the source.

2. **Field extraction**
      - If a molecule is present and |E| at its location exceeds `tolerance_field_e`, extract the electric field vector at that point E(t).

3. **Quantum propagation** (RT-TDDFT)
      - Build time-dependent Fock matrix in the orthogonal basis:
      ```
      F_orth(t) = F_0 + V_ext(E(t)) - i Γ(t)   (optional Lopata CAP broadening)
      ```
      - Propagate molecular orbitals / density matrix using chosen propagator (default: 2nd-order Magnus + predictor-corrector).
      - Compute induced dipole:
      ```
      μ_ind(t) = Tr[μ (D(t) - D_0)]
      ```

4. **Back-coupling**
      - The induced dipole is injected back into Meep as a `CustomSource` at the molecule's position.

5. **Repeat** until `t_end`.
