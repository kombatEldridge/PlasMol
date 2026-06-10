# Bug-Diagnosis Strategies for `mcp/h2o.json` Fourier Runs

This document describes how to identify each injected fault from `randomizer.py` by working **backwards from the Fourier absorption spectrum**, using only the PlasMol MCP tools (plus targeted, hypothesis-driven code search). The goal is to localize the culprit without reading the entire codebase.

## Baseline workflow (all bugs)

Use this pipeline for every run. Subsequent sections explain what to look for at each step.

```
1. lab_manual()
      → skim installation, usage, and methodology; note the Fourier driver runs
        three directional kick simulations (x/y/z), folds dipole CSVs, FFTs, normalizes.

2. read_json("mcp/h2o.json")
      → confirm driver=fourier, dt=0.01, t_end=2000, kick intensity=5e-5,
        lrc_parameter=0.581692, broadening gam0/xi/eps0.

3. parse_params("mcp/h2o.json")
      → record effective dt, t_end, fourier_gamma, broadening settings.
        Compare these to the JSON — mismatches here still flag input-parsing bugs,
        but `dt_wrong_unit` / `t_end_wrong_unit` now live in `classical/simulation.py`
        and do not change parsed params.

4. setup_conda_environment() → list_jobs() → confirm_conda_environment_setup()
      → environment must be ready before run_simulation().

5. run_simulation("mcp/h2o.json") → job_id

6. list_jobs() → read_log(job_id)
      → check status (completed / failed), runtime, logged dt/t_end in au and fs,
        Fourier gamma, damping messages, errors in stderr.

7. read_fourier_spectrum(job_id)
      → primary observable; note peak positions (eV) and relative heights.
        For h2o, expect prominent features roughly near 7–10 eV and ~12 eV
        (compare against a known-good run if available).

8. read_field_csv(job_id, "p")  and  read_field_csv(job_id, "e")
      → inspect time-domain dipole (polarization) and driving field.
        Count rows, infer timestep from timestamps, check kick magnitude and decay.

9. reference_spectrum(molecule, condition)
      → placeholder reference only; useful for workflow practice, not for water.

10. Hypothesis → search_code("<pattern>", "plasmol/")
      → one targeted grep, not a full-tree read.

11. read_file("<suspect>", start, end)
      → confirm the single offending line.

12. submit_diagnosis(soln_desc, strategy_desc, uncertainty)
```

### Known-good anchors for `h2o.json`

Keep a spectrum from a clean tree (no injection) as a mental reference. The MCP
`read_fourier_spectrum` output is **max-normalized** (`abs / max(abs)`), so bugs
that only scale absorption uniformly can be invisible in step 7 alone.

---

## 1. `electric_field_intensity_au`

**Injection:** `plasmol/quantum/sources.py` — kick amplitude multiplied by `1e2` at the peak-time mask.

**Physical effect:** Driving field is 100× stronger; induced dipole and raw absorption scale ~100× (linear response). Input JSON still says `intensity: 5e-5`.

### Expected symptoms

| Stage | Signal |
|-------|--------|
| `parse_params` | `molecule_source_intensity` unchanged (5e-5) — bug is downstream |
| `read_field_csv(job_id, "e")` | Kick column magnitude ~100× larger than a good run at `peak_time ≈ 0.05` au |
| `read_field_csv(job_id, "p")` | Dipole amplitude ~100× larger throughout |
| `read_fourier_spectrum` | **Normalized shape may match a good run** (scaling cancels in `abs/max(abs)`) |
| `read_log` | Simulation completes normally |

### Diagnosis strategy

1. Run baseline workflow through step 8.
2. Compare `read_field_csv(..., "e")` kick value against `parse_params` intensity — a 100× gap between configured intensity and realized field implicates the source constructor.
3. If only the spectrum is available, compare unnormalized absorption magnitudes (see **Recommended additional tools** — this bug is hard to catch from normalized spectrum alone).
4. Confirm:
   ```
   search_code("intensity_au", "plasmol/quantum/sources.py")
   read_file("plasmol/quantum/sources.py", 54, 62)
   ```
   Look for `1e2 * self.intensity_au` on the kick branch.

### MCP tool sequence (concise)

`parse_params` → `run_simulation` → `read_field_csv(e)` → `read_field_csv(p)` → `search_code` → `read_file` → `submit_diagnosis`

---

## 2. `dt_wrong_unit`

**Injection:** `plasmol/classical/simulation.py` — Meep timestep uses the wrong unit conversion:

```python
# correct
self.dt_meep = self.dt / constants.convertTimeMeep2Atomic
# injected
self.dt_meep = self.dt * constants.convertTimeMeep2Atomic
```

**Scope:** Only affects runs that instantiate `SIMULATION` (classical FDTD or full hybrid PlasMol). **`mcp/h2o.json` is quantum-only Fourier and never touches this file** — if this bug is injected, a water Fourier run will look like `no_fault`.

**Physical effect (hybrid/classical):** Meep advances ~(convertTimeMeep2Atomic)² too slowly per atomic timestep. FDTD–quantum coupling fires at the wrong rate; hybrid field CSV timestamps and induced-dipole feedback are misaligned with the quantum sub-steps.

### Expected symptoms (hybrid/classical input)

| Stage | Signal |
|-------|--------|
| `parse_params` | `dt` and `t_end` match JSON — **no smoking gun here** |
| `read_log` | Quantum sub-log may look normal; Meep section shows odd coupling frequency or premature termination |
| `read_field_csv(..., "e")` / `"p"` | Timestamp spacing inconsistent with `parse_params` `dt`; hybrid dipole rows sparse or clustered |
| `read_fourier_spectrum` | Distorted spectrum from mis-sampled dipole trace (hybrid absorption workflows) |

### Diagnosis strategy

1. Confirm the driver is **not** pure quantum Fourier — check `read_json` for both `plasmon` and `molecule` blocks (hybrid) or `plasmon` only (classical).
2. `parse_params` matches JSON for `dt` — rules out the old params-level bug pattern.
3. Compare inferred CSV timestep against `parse_params` `dt`; a large discrepancy without a params mismatch points to Meep conversion code.
4. Confirm:
   ```
   search_code("dt_meep", "plasmol/classical/simulation.py")
   read_file("plasmol/classical/simulation.py", 17, 20)
   ```
   Look for `*` instead of `/` in the `dt_meep` assignment.

### MCP tool sequence

`read_json` → `parse_params` → `run_simulation` → `read_field_csv(e/p)` → `read_fourier_spectrum` → `search_code("dt_meep")` → `read_file` → `submit_diagnosis`

---

## 3. `t_end_wrong_unit`

**Injection:** `plasmol/classical/simulation.py` — Meep end time uses the wrong unit conversion:

```python
# correct
self.t_end_meep = self.t_end / constants.convertTimeMeep2Atomic
# injected
self.t_end_meep = self.t_end * constants.convertTimeMeep2Atomic
```

**Scope:** Same as `dt_wrong_unit` — classical/hybrid only; **inert for `mcp/h2o.json`**.

**Physical effect (hybrid/classical):** Meep runs far too long or far too short relative to the configured atomic `t_end`. Hybrid simulations may terminate before the plasmon field fully interacts with the molecule, or run far past the intended end.

### Expected symptoms (hybrid/classical input)

| Stage | Signal |
|-------|--------|
| `parse_params` | `t_end` matches JSON |
| `read_log` | Wall time wildly off vs a good run; Meep "completed" message may arrive too early/late |
| `read_field_csv(..., "p")` | Final timestamp does not approach `parse_params` `t_end` (in au) |
| `read_fourier_spectrum` | Truncated or over-extended dipole data alters FFT peaks |

### Diagnosis strategy

1. `parse_params` — `t_end` matches JSON, so the bug is downstream in Meep setup.
2. `read_field_csv(..., "p")` — inspect the last timestamp vs expected `t_end`.
3. `read_log` — compare runtime and Meep completion against a known-good hybrid run.
4. Confirm:
   ```
   search_code("t_end_meep", "plasmol/classical/simulation.py")
   read_file("plasmol/classical/simulation.py", 17, 20)
   ```
   Look for `*` instead of `/` in the `t_end_meep` assignment.

### MCP tool sequence

`parse_params` → `run_simulation` → `read_field_csv(p)` → `read_log` → `read_fourier_spectrum` → `search_code("t_end_meep")` → `read_file` → `submit_diagnosis`

---

## 4. `fourier_field_p_damping_gamma_flipped`

**Injection:** `plasmol/utils/input/params.py` — swaps the `fourier_damp` True/False assignments tied to `fourier_field_p_damping_gamma`.

**Physical effect (with `mcp/h2o.json`):** This input does **not** set `field_p_damping_gamma`, so the correct value is `fourier_damp = False`. The bug forces `fourier_damp = True`, and the Fourier driver then calls `apply_damping` using a missing `fourier_field_p_damping_gamma` attribute.

### Expected symptoms

| Stage | Signal |
|-------|--------|
| `read_log` | Job **failed** during Fourier post-processing; stderr shows `AttributeError` for `fourier_field_p_damping_gamma` |
| `read_fourier_spectrum` | Unavailable (job not completed) |
| `read_log` stdout | May show x/y/z quantum runs completed, then failure at damping step |

If the input **did** include `field_p_damping_gamma`, the symptom would instead be: missing log line *"Damped polarizability field written to …_damped.csv"* and a spectrum computed from undamped (noisier) dipole data.

### Diagnosis strategy

1. `read_log` — failure after directional runs implicates Fourier post-processing, not propagation.
2. Search for damping logic:
   ```
   search_code("fourier_damp", "plasmol/")
   read_file("plasmol/utils/input/params.py", 355, 365)
   read_file("plasmol/drivers/custom_drivers/fourier.py", 170, 180)
   ```
3. Cross-check `read_json` — if `field_p_damping_gamma` is absent but damping code ran, the flip is confirmed.

### MCP tool sequence

`run_simulation` → `read_log` → `search_code("fourier_damp")` → `read_file` → `submit_diagnosis`

---

## 5. `get_gamma_ao_wrong`

**Injection:** `plasmol/quantum/molecule.py` — replaces Lopata damping `gam0 * (exp(xi * e_tilde) - 1)` with `gam0 * exp(xi * e_tilde)`.

**Physical effect:** Artificial broadening in the Fock matrix is too large (missing the `-1` offset). Static broadening (`h2o.json` default) injects excess imaginary damping during propagation.

### Expected symptoms

| Stage | Signal |
|-------|--------|
| `parse_params` | Broadening block looks normal (gam0, xi, eps0) |
| `read_log` | "Broadening modifier selected" — completes without error |
| `read_fourier_spectrum` | **Peak widths differ** from good run; peaks may be overly broad or suppressed |
| `read_field_csv(..., "p")` | Faster decay / stronger damping envelope in dipole trace |

### Diagnosis strategy

1. `read_fourier_spectrum` — compare linewidths and relative peak heights to a good run (not just positions).
2. `read_field_csv(..., "p")` — inspect post-kick dipole decay; excessive decay suggests broadening/Gamma error.
3. `lab_manual()` — read methodology on Lopata broadening to know the expected `(exp(ξε̃) − 1)` form.
4. Confirm:
   ```
   search_code("exp\\(xi \\* e_tilde\\)", "plasmol/quantum/molecule.py")
   read_file("plasmol/quantum/molecule.py", 400, 410)
   ```

### MCP tool sequence

`lab_manual` → `run_simulation` → `read_fourier_spectrum` → `read_field_csv(p)` → `search_code` → `read_file` → `submit_diagnosis`

---

## 6. `absorption_missing_factor_4`

**Injection:** `plasmol/drivers/custom_drivers/fourier.py` — removes the factor of `4` in `return - 4 * np.pi * freqs / 3 / C_AU * fullsum`.

**Physical effect:** Raw absorption is exactly 4× too small. **Max-normalization cancels this completely** in the saved spectrum CSV and plot.

### Expected symptoms

| Stage | Signal |
|-------|--------|
| `read_fourier_spectrum` | **Identical normalized shape** to a good run — this bug is invisible here |
| `read_field_csv(..., "p")` | Normal — dipole data unaffected |
| `read_log` | Completes normally |
| Intermediate data | Raw `abs` array in memory is 4× smaller before normalization |

### Diagnosis strategy

This is the hardest bug to catch from spectrum alone. Work backwards:

1. If dipole traces (`read_field_csv`) match a good run but you still suspect a post-processing error, narrow search to the Fourier driver:
   ```
   search_code("absorption|4 \\* np.pi", "plasmol/drivers/custom_drivers/fourier.py")
   read_file("plasmol/drivers/custom_drivers/fourier.py", 128, 190)
   ```
2. Inspect the `absorption()` formula for the missing factor of 4.
3. Ideally compare **unnormalized** absorption integrals or `fourier.npz` contents (see additional tools).

**Practical workaround without new tools:** Run two simulations (injected vs known-good tree) and diff intermediate outputs on disk if accessible; or reason that only a formula constant in `fourier.py` can change spectrum amplitude without changing dipole CSVs.

### MCP tool sequence

`read_field_csv(p)` (matches good) → `read_fourier_spectrum` (matches good) → `search_code("4 \\* np.pi", "plasmol/drivers")` → `read_file` → `submit_diagnosis`

---

## 7. `mf_omega_zero`

**Injection:** `plasmol/quantum/molecule.py` — `self.mf.omega = 0` instead of `self.molecule_lrc_parameter`.

**Physical effect:** Range-separated functional (`HYB_GGA_XC_LC_PBEOP`) uses wrong ω. Ground-state electronic structure and excitation energies shift; absorption peak **positions** move.

### Expected symptoms

| Stage | Signal |
|-------|--------|
| `parse_params` | `molecule_lrc_parameter: 0.581692` — input is correct |
| `read_log` | SCF completes; no obvious error |
| `read_fourier_spectrum` | Peak **locations** (eV) shifted vs good run; overall shape wrong |
| `read_field_csv` | Dipole qualitatively similar but dynamics differ |

### Diagnosis strategy

1. `read_fourier_spectrum` — focus on peak **positions**, not just amplitude (contrast with `electric_field_intensity_au` and `absorption_missing_factor_4`).
2. `parse_params` confirms LRC parameter in JSON is fine → bug is in how it is applied to PySCF.
3. Confirm:
   ```
   search_code("mf\\.omega", "plasmol/quantum/molecule.py")
   read_file("plasmol/quantum/molecule.py", 56, 62)
   ```

### MCP tool sequence

`parse_params` → `run_simulation` → `read_fourier_spectrum` → `search_code("mf.omega")` → `read_file` → `submit_diagnosis`

---

## 8. `no_fault`

**Injection:** none.

### Expected symptoms

All workflow steps succeed. `read_fourier_spectrum` matches a known-good reference. `parse_params` matches `read_json` for dt/t_end. Field CSV row counts and kick magnitudes are consistent.

### Diagnosis strategy

1. Run full baseline workflow.
2. Compare spectrum peaks and `read_field_csv` statistics to a stored good run.
3. If everything aligns, `submit_diagnosis(..., uncertainty=0.0–0.1)`.

---

## Decision tree (quick reference)

```
Job failed?
  └─ stderr mentions fourier_field_p_damping_gamma → fourier_field_p_damping_gamma_flipped

Hybrid/classical run, parse_params dt/t_end match JSON, but CSV timestamps disagree?
  └─ dt_wrong_unit or t_end_wrong_unit (check classical/simulation.py Meep conversions)

Spectrum peak positions wrong, params look fine?
  └─ mf_omega_zero

Spectrum peak widths wrong, dipole decay too fast?
  └─ get_gamma_ao_wrong

field_e kick ~100× parse_params intensity?
  └─ electric_field_intensity_au

Dipole CSV matches good run, spectrum normalized shape matches good run?
  └─ absorption_missing_factor_4 (check fourier.py formula)
     OR no_fault (if all intermediate checks pass)
```

---

## Recommended additional MCP tools

These are not essential for every bug, but they close blind spots in the current toolset.

| Proposed tool | Why it helps |
|---------------|--------------|
| `field_csv_summary(job_id, field)` | Return row count, inferred `dt`, max/mean dipole or field amplitude without transferring full CSVs. Speeds detection of Meep conversion bugs (`dt_wrong_unit`, `t_end_wrong_unit`) and `electric_field_intensity_au`. |
| `read_spectrum_raw(job_id)` | Return unnormalized absorption (`abs` before `max(abs)`). **Essential** for reliably catching `absorption_missing_factor_4` and `electric_field_intensity_au`. |
| `read_fourier_npz(job_id)` | Load `fourier.npz` (`abs_imag`, `freqs`) for per-axis FFT debugging without re-running. |
| `grep_log(job_id, pattern)` | Search stdout/stderr for timestep lines, damping messages, errors — faster than reading full logs. |
| `list_run_outputs(job_id)` | List files in the run directory (`x_dir/`, `spectrum.csv`, `fourier.npz`, etc.) to guide which reader to call next. |
| `compare_spectrum(job_id, reference_path)` | Peak-pick and report position/width/height deltas vs a known-good `spectrum.csv`. Automates the manual comparison steps for `get_gamma_ao_wrong` and `mf_omega_zero`. |

### Priority additions

1. **`read_spectrum_raw`** — without it, `absorption_missing_factor_4` is theoretically undetectable from `read_fourier_spectrum` alone.
2. **`field_csv_summary`** — cheap confirmation of timestep and duration bugs before reading megabyte CSVs.

---

## Notes on `edit_file` and remediation

Once a bug is identified, `edit_file(path, old_line, new_line)` can revert the single injected line. Always re-run:

```
run_simulation("mcp/h2o.json") → read_fourier_spectrum(job_id)
```

to verify the fix restores expected peak structure before calling `submit_diagnosis`.