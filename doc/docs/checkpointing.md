# Checkpointing

PlasMol can snapshot a **pure-quantum** time propagation so a long RT-TDDFT run can be stopped and continued later. Checkpoints are NumPy `.npz` archives: they hold the electronic state, a copy of the input, and the field / occupation CSVs written so far.

```{warning}
Checkpointing is **molecule-only**. If the input contains a `plasmon` section (Meep cell, nanoparticle, hybrid absorption, …), PlasMol **disables** checkpointing at parse time and logs a warning. There is no FDTD / plasmon restart.
```

## What is supported

| Run | Checkpointing |
| ------ | ---------------- |
| [`quantum`](simulations/quantum.md) | Yes — mid-run snapshots plus a final archive |
| [`absorption`](simulations/absorption.md) **without** `plasmon` | Yes — one snapshot set per kick direction (`x`/`y`/`z`), then merged |
| [`core_hole`](core_hole.md) | Yes — same quantum path, plus the MO-occupation CSV |
| Hybrid `absorption` (`parallel` / `perpendicular` / full with a plasmon) | **No** |
| [`plasmol`](simulations/plasmol.md), [`classical`](simulations/classical.md), NP cross-section, scatter, verify-source | **No** |
| [`comparison`](simulations/comparison.md), [`tune`](simulations/tune.md) | Not used (no time loop writes snapshots) |

The gate is simple: a `files.checkpoint` block **and** a `molecule` section **and** no `plasmon` section. The quantum time loop (`plasmol/drivers/quantum.py`) is what actually writes and reloads state.

## How it works

Two files are maintained during a run:

| File | Role |
| ------ | ------ |
| `checkpoint.npz` (name from `files.checkpoint.filepath`) | Periodic snapshot every `frequency_steps` or `frequency_time` |
| `final-checkpoint.npz` | Written at the end of a successful run |

Hidden working copies (`.checkpoint.npz`, `.checkpoint_x.npz`, …) are used while workers write. They are removed on a clean exit. Resume from the **visible** `checkpoint.npz` or `final-checkpoint.npz`.

On each snapshot PlasMol embeds:

- **Propagator state** — `D_ao_0`, `mo_coeff`, and either Magnus2 `F_orth_n12dt` or step `C_orth_ndt` (UKS keeps a leading spin axis)
- **Clock** — `checkpoint_time` (a.u.), or `checkpoint_time_x/y/z` for three-direction absorption
- **Input copy** — raw JSON bytes plus the geometry file, if any
- **CSVs** — `field_e` / `field_p` (per direction when needed) as raw file bytes
- **Core-hole** — `core_hole_mo_occ_content` when that driver is active
- **Flags** — `is_absorption`, `is_open_shell`, `updated_after_init`

The log prints a short NPZ header (keys + unpack line) when an archive is written. To inspect a file later:

```bash
python -m plasmol.utils.npz checkpoint.npz
```

```{note}
`params_dict` is a pickled snapshot of the `PARAMS` object at save time. Resume still **re-parses the restored JSON**; the live run is not a raw unpickle of that dict.
```

## Enable it

```json
"files": {
  "checkpoint": {
    "filepath": "checkpoint.npz",
    "frequency_steps": 100
  }
}
```

Use **either** `frequency_steps` **or** `frequency_time` (a.u.), not both. `frequency_time` must be an integer multiple of `settings.dt`. See [Usage](usage.md) for the full key table. The quantum template (`templates/template-quantum.json`) already includes a checkpoint block.

## Resume (two steps)

Restore is intentionally **not** “pass `-c` and keep going.” It is two launches:

**1. Restore files and exit**

```bash
python -m plasmol.main -c checkpoint.npz
# or: python -m plasmol.main checkpoint.npz
```

This writes a restored input (`…_restored.json`), restored geometry if needed (`…_restored.xyz`), and the field CSVs, then **stops**. The restored JSON sets `additional_parameters.checkpoint_filename_used` to the archive you named.

**2. Continue from the restored input**

```bash
python -m plasmol.main -f my_run_restored.json -vv -l resume.log
```

New snapshots go to `checkpoint_new.npz` so the archive you resumed from is not overwritten.

Before step 2 you may edit the restored JSON. Parameters treated as safe include `dt`, `t_end`, checkpoint frequency / filepath, absorption spectrum window / `npz_filepath`, `spectra_e_vs_p_filepath`, and verbosity.

```{warning}
`t_end` must be **strictly after** the saved `checkpoint_time`. If it is equal, that direction is treated as already finished; if it is earlier, the run errors.
```

```{caution}
Changing `dt` on resume is allowed by the parser but reticks the time grid. The code finds the first new `times` entry at or after `checkpoint_time`. Do not change geometry, basis, `xc`, charge, spin, or propagator type and expect a consistent restart.
```

## What checkpointing will not do

```{warning}
**No plasmon / Meep restart.** Hybrid and classical jobs cannot be continued from a quantum snapshot. If you put `files.checkpoint` on a plasmon input, the block is stripped and the run proceeds without snapshots.
```

```{warning}
**Not a bit-identical replay.** Resume rebuilds the molecule from the restored JSON, then overwrites density / MO / propagator arrays from the NPZ. SCF is performed again on the way in; the saved density is what continues, not the SCF result.
```

```{caution}
**Absorption + plasmon is never checkpointed**, including `polarization: parallel` or `perpendicular`. Only the molecule-only (three-kick) absorption path writes per-direction archives.
```

Other limits:

- Checkpoints do **not** store PNG/GIF frames, the absorption spectrum plot, or `absorption.npz`.
- They do **not** store a live Meep simulation, PML fields, or nanoparticle polarization.
- `comparison` / `tune` do not write these snapshots even if a checkpoint block parses.
- Older archives may still use the key `is_fourier` instead of `is_absorption`; load accepts both.

## See also

- [Usage](usage.md) — `files.checkpoint` keys
- [Quantum driver](simulations/quantum.md)
- [Core-hole](core_hole.md) — occupation CSV embed
- [Absorption](simulations/absorption.md) — molecule-only vs hybrid
