# Core-Hole Dynamics (SCH / DCH)

PlasMol’s **core-hole** driver (`settings.driver: "core_hole"`) implements sudden single and double core-hole initial conditions for real-time TDDFT and tracks hole occupation dynamics projected onto the neutral molecular orbitals.

The driver name is intentionally **`core_hole`**, not “DCH”: the same workflow covers **SCH** (single core-hole), **DCH** (double core-hole on one MO), and two simultaneous single holes on different MOs.

## Motivation

Core-ionized states are the starting point for X-ray spectroscopies and many pump–probe scenarios. PlasMol builds a **sudden** core hole from a neutral ground-state SCF, freezes the neutral MO basis for analysis, and propagates the open-shell density in time—optionally under an external field—while logging which orbitals carry the hole.

## Sudden approximation

1. **Neutral SCF** — Closed- or open-shell DFT on the user geometry (charge/spin as given).
2. **Freeze neutral MOs** — Store neutral coefficients and occupations as the projection basis for later analysis.
3. **Remove electrons without re-SCF** — For each entry in `mo_removal_index_dict`, zero the requested number of electrons (1 or 2) on that MO index (0-based) using a maximum-overlap / `mom_occ` style occupation set.
4. **Adjust charge and spin** on the PySCF molecule object and rebuild so UKS can carry the non-stationary density.
5. **Propagate** with the usual RT-TDDFT stack (default Magnus2). The density is **not** re-optimized after ionization; the initial condition is intentionally non-stationary.

Charge/spin rules (neutral parent with spin \(S_0\)):

| Mode | `mo_removal_index_dict` | Charge | Spin |
| ------ | ------------------------- | -------- | ------ |
| SCH | `{i: 1}` | \(+1\) | \(S_0 + 1\) |
| DCH (one MO) | `{i: 2}` | \(+2\) | \(S_0\) (closed double hole) |
| Two SCH | `{i: 1, j: 1}` | \(+2\) | \(S_0 + 2\) |

Core-hole runs **always force open-shell (UKS)** so α/β channels can describe the hole.

## Driver and JSON parameters

```json
{
  "settings": {
    "dt": 0.05,
    "t_end": 200,
    "driver": "core_hole"
  },
  "molecule": {
    "geometry": "3p.xyz",
    "geometry_units": "angstrom",
    "charge": 0,
    "spin": 0,
    "basis": "6-311G*",
    "xc": "PBE0",
    "propagator": { "type": "magnus2" }
  },
  "files": {
    "field_e_filepath": "field_e.csv",
    "field_p_filepath": "field_p.csv",
    "spectra_e_vs_p_filepath": "output.png"
  },
  "additional_parameters": {
    "mo_removal_index_dict": {"0": 2},
    "core_hole_mo_occ_filepath": "mo_occ.csv",
    "core_hole_watch_indices": [21, 22, 23, 24],
    "core_hole_filter_by_amplitude": false,
    "core_hole_amplitude_threshold": 0.2
  }
}
```

| Key | Description |
| ----- | ------------- |
| `mo_removal_index_dict` | **Required.** Map of 0-based MO index → electrons to remove (1 or 2). JSON keys may be strings. |
| `core_hole_mo_occ_filepath` | **Required** for propagation. CSV of time-dependent hole occupations. |
| `core_hole_watch_indices` | Optional list of MO indices to include in the final plot (logging always covers 0 … LUMO+1). |
| `core_hole_filter_by_amplitude` | If true, plot only MOs whose peak-to-peak hole amplitude exceeds the threshold. |
| `core_hole_amplitude_threshold` | Amplitude cutoff (default 0.2). |
| `check_mo_contrib_by_atom` | Survey mode: print per-atom contributions for listed MOs and **exit** before propagation. |

## Hole occupation logging

After each RT-TDDFT step the driver projects the current density onto the **neutral** MOs and records

\[
n_{\mathrm{hole},k}(t) = n_k^{(0)} - n_k(t)
\]

for each logged MO index \(k\). A double hole on MO 0 starts near \(n_{\mathrm{hole},0}\approx 2\); dynamics may redistribute the hole among valence/virtuals.

The final PNG is written next to `core_hole_mo_occ_filepath` (same basename).

## Survey mode (choosing the core MO)

Before a production run it is often useful to inspect which atoms dominate a candidate MO:

```json
"additional_parameters": {
  "check_mo_contrib_by_atom": true,
  "mo_removal_index_dict": {"0": 2, "1": 2, "2": 2},
  "core_hole_mo_occ_filepath": "mo_occ.csv"
}
```

PlasMol builds the neutral molecule, prints AO-projected contributions above a small threshold for each listed MO, and exits (no time propagation).

## Checkpointing

Core-hole runs are pure quantum simulations and support the usual checkpoint machinery. The MO-occupation CSV content is embedded in the checkpoint NPZ so restarts can restore both the electronic state and the occupation history.

## Implementation map

| Piece | Location |
| ------- | ---------- |
| Driver orchestration | `plasmol/drivers/custom_drivers/core_hole.py` |
| Sudden ionization | `MOLECULE.remove_core_electrons` |
| MO logging | `MOLECULE.get_mo_occupations`, `_setup_core_hole_mo_logging` |
| Validation | `PARAMS` when `driver_str == "core_hole"` |
| Plotting | `plot_core_hole_mo_occupations` |

## See also

- [Usage](../usage.md) — full parameter tables
- [Tutorials](../tutorials.md) — walkthroughs
- [Theory & Methodology](../methodology.md) — hybrid FDTD–RT-TDDFT loop
