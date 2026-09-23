# Core-Hole Dynamics (SCH / DCH)

Sudden single and double core-hole initial conditions are a **molecule** feature: put a `molecule.core_hole` block on a `quantum`, `absorption`, or `plasmol` run. PlasMol ionizes the listed MOs without re-SCF, propagates with the usual RT-TDDFT stack, and tracks hole occupations on the **neutral** MO basis.

The dedicated **`core_hole` driver** does **not** propagate. It only surveys per-atom contributions for the listed MOs (`check_mo_contrib_by_atom`) so you can choose which orbital to ionize, then exits.

The name is **`core_hole`**, not “DCH”: the same subsection covers **SCH** (single core-hole), **DCH** (double core-hole on one MO), and two simultaneous single holes on different MOs.

## Motivation

Core-ionized states are the starting point for X-ray spectroscopies and many pump–probe scenarios. PlasMol builds a **sudden** core hole from a neutral ground-state SCF, freezes the neutral MO basis for analysis, and propagates the open-shell density in time—optionally under an external field—while logging which orbitals carry the hole.

## Sudden approximation

1. **Neutral SCF** — Closed- or open-shell DFT on the user geometry (charge/spin as given).
2. **Freeze neutral MOs** — Store neutral coefficients and occupations as the projection basis for later analysis.
3. **Remove electrons without re-SCF** — For each entry in `mo_removal_index_dict`, zero the requested number of electrons (1 or 2) on that MO index (0-based) using a maximum-overlap / `mom_occ` style occupation set.
4. **Adjust charge and spin** on the PySCF molecule object and rebuild. A singlet DCH stays restricted (RKS); SCH and two-site holes switch to UKS.
5. **Propagate** with the usual RT-TDDFT stack (default Magnus2). The density is **not** re-optimized after ionization; the initial condition is intentionally non-stationary.

Charge/spin rules (neutral parent with spin \(S_0\)):

| Mode | `mo_removal_index_dict` | Charge | Spin |
| ------ | ------------------------- | -------- | ------ |
| SCH | `{i: 1}` | \(+1\) | \(S_0 + 1\) |
| DCH (one MO) | `{i: 2}` | \(+2\) | \(S_0\) (closed double hole) |
| Two SCH | `{i: 1, j: 1}` | \(+2\) | \(S_0 + 2\) |

A singlet DCH (`{i: 2}` on a spin-0 parent) stays **closed-shell (RKS)**, which is what stock NWChem DFT / RT-TDDFT does for `mult 1` without `odft`. SCH and two-site holes **force UKS** so α/β channels can describe the unpaired hole.

## JSON: `molecule.core_hole`

Production SCH/DCH is enabled by the subsection, not by the driver name. Use `"driver": "quantum"` (field-free or with a molecule source), `"absorption"` (spectra, including hybrid NP+molecule), or omit the driver on a molecule-only / hybrid input.

```json
{
  "settings": {
    "dt": 0.05,
    "t_end": 200,
    "driver": "quantum"
  },
  "molecule": {
    "geometry": [{"atom": "C", "coord": [0.0, 0.0, 0.0]}, {"atom": "O", "coord": [0.0, 0.0, 1.13]}],
    "geometry_units": "angstrom",
    "charge": 0,
    "spin": 0,
    "basis": "6-311G*",
    "xc": "PBE0",
    "propagator": {
      "type": "magnus2"
    },
    "core_hole": {
      "mo_removal_index_dict": {"0": 2},
      "mo_occ_filepath": "mo_occ.csv",
      "watch_indices": [21, 22, 23, 24],
      "filter_by_amplitude": false,
      "amplitude_threshold": 0.2
    }
  },
  "files": {
    "field_e_filepath": "field_e.csv",
    "field_p_filepath": "field_p.csv",
    "spectra_e_vs_p_filepath": "output.png"
  }
}
```

| Key | Description |
| ----- | ------------- |
| `mo_removal_index_dict` | **Required.** Map of 0-based MO index → electrons to remove (1 or 2). JSON keys may be strings. |
| `mo_occ_filepath` | **Required** for propagation. CSV of time-dependent hole occupations. Absorption copies write this file next to that direction's field CSVs (`x_dir/`, `fields/`, …). |
| `watch_indices` | Optional list of MO indices to include in the final plot (logging always covers 0 … LUMO+1). |
| `filter_by_amplitude` | If true, plot only MOs whose peak-to-peak hole amplitude exceeds the threshold. |
| `amplitude_threshold` | Amplitude cutoff (default 0.2). |

A molecule source is optional for field-free DCH: omitted `molecule.source` is a zero field. Hybrid runs still use the Meep field at the molecule; RT-TDDFT is called even when \(|\mathbf{E}|\) is below `tolerance_field_e` so the hole can evolve with no plasmon drive.

Older inputs that put these keys on `settings.driver` (including `core_hole_mo_occ_filepath`) or under `additional_parameters` are rewritten onto `molecule.core_hole` with a warning.

## Hole occupation logging

After each RT-TDDFT step the driver projects the current density onto the **neutral** MOs and records the hole occupation \(h_k(t)\).

**Closed-shell (singlet DCH).** PySCF’s RKS density is the total density (occupations 0 or 2). The logged number is the Nascimento / NWChem closed-shell \(P\) (occupations 0 or 1):

\[
n_k(t)=\tfrac12\bigl[C_n^\dagger S\,D_{\mathrm{AO}}(t)\,S\,C_n\bigr]_{kk},\qquad
h_k(t)=\tfrac12 n_k^{(0)}-n_k(t).
\]

A double hole on MO 0 starts near \(h_0\approx 1\).

**Open-shell (SCH / two-site holes).** Each spin is already 0 or 1; the log is the α+β sum, so a single hole starts near \(h\approx 1\).

The final PNG is written next to `mo_occ_filepath` (same basename).

## Survey driver (`settings.driver: "core_hole"`)

The `core_hole` driver always runs the MO-contribution survey and **exits** (no ionization, no time loop). List candidate MOs in `molecule.core_hole.mo_removal_index_dict` (electron counts are ignored; more than two MOs are allowed):

```json
{
  "settings": {
    "dt": 0.05,
    "t_end": 1.0,
    "driver": "core_hole"
  },
  "molecule": {
    "geometry": [{"atom": "C", "coord": [0.0, 0.0, 0.0]}, {"atom": "O", "coord": [0.0, 0.0, 1.13]}],
    "geometry_units": "angstrom",
    "charge": 0,
    "spin": 0,
    "basis": "sto3g",
    "xc": "pbe0",
    "core_hole": {
      "mo_removal_index_dict": {"0": 2, "1": 2, "2": 2}
    }
  }
}
```

PlasMol builds the **neutral** molecule, prints AO-projected contributions above a small threshold for each listed MO, and exits. Then switch `driver` to `quantum` or `absorption` and add `mo_occ_filepath` for the production run.

## Checkpointing

Molecule-only quantum (and molecule-only absorption) core-hole runs support the usual [checkpoint](checkpointing.md) machinery. The MO-occupation CSV is embedded in the checkpoint NPZ so restarts restore both the electronic state and the occupation history. Hybrid plasmon runs cannot be checkpointed.

## Implementation map

| Piece | Location |
| ------- | ---------- |
| Survey driver | `plasmol/drivers/custom_drivers/core_hole.py` |
| Sudden ionization | `MOLECULE.remove_core_electrons` (when `molecule.core_hole` is set) |
| MO logging | `MOLECULE.get_mo_occupations`, `_setup_core_hole_mo_logging` |
| Validation | `PARAMS` / `has_core_hole` when `molecule.core_hole` is present |
| Plotting | `maybe_plot_core_hole_occupations` from quantum / plasmol / absorption workers |

## See also

- [Usage](../usage.md) — full parameter tables
- [Checkpointing](checkpointing.md) — snapshots and resume
- [Tutorials](../tutorials.md) — walkthroughs
- [Theory & Methodology](../methodology.md) — hybrid FDTD–RT-TDDFT loop
