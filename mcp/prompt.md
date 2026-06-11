You are a computational chemistry software developer who has cloned a fork of an internal software suite **PlasMol** that was modified by a colleague. The fork is downloaded to `/workspace/PlasMol`. The tool in PlasMol used to compute the absorption spectrum of a given molecule has started returning wrong results but the test suite passes superficially on the small cases. 

Within `/workspace/PlasMol/jobs`, there is the input file used for creating an absorption spectrum for water called `water.json`. It shouldn’t take that long to run given that the molecule is small and we are using a limited basis set and xc functional.

**At most one** of the following bug categories may be active:

* `bug_intensity_au`
* `bug_dt_meep`
* `bug_t_end_meep`
* `bug_fourier_damp`
* `bug_gamma`
* `bug_absorption`
* `bug_mf_omega`
* `no_fault`

Use `list_bug_categories()` for the exact strings accepted by `submit_diagnosis`.

A single absorption spectrum won’t be able to help tell a difference between any of these bugs, so you’ll need to construct a strategy to verify which bug is active. Identify the bug and explain why it produces the observed failure mode. Work only through the tools provided to you (described below). Do not read private codebase internals directly.

**Tools:**

* `get_cwd()` → returns the current working directory  
* `setup_conda_environment()` → one-time setup for the conda environment containing Meep; returns `job_id` and `conda_env`  
* `confirm_conda_environment_setup()` → verifies the conda environment is ready (imports Meep); required before `run_simulation()`  
* `lab_manual()` → returns compiled PlasMol documentation (index, about, installation, usage, methodology)  
* `list_dir(path)` → lists files and subdirectories at a given path in the repository  
* `read_file(path, start_line, end_line)` → reads a range of lines from a source file (`start_line` is 1-indexed, `end_line` is exclusive)  
* `search_code(pattern, path)` → grep-style search over `.py` source files for a string or regex  
* `edit_file(path, old_line, new_line)` → replaces exactly one line in a source file; fails if the line does not match uniquely  
* `read_json(input_json_path)` → returns the raw input text  
* `parse_params(input_json_path)` → returns the parsed input params  
* `run_simulation(input_json_path)` → submits a full PlasMol run; returns `job_id` and `run_directory`  
* `list_jobs()` → returns list of jobs and their statuses (most recent first)  
* `read_log(job_id)` → returns stdout and stderr for a job  
* `kill_job(job_id)` → terminates a running job  
* `read_field_csv(job_id, field)` → returns electric field or polarization CSV data (`field` is `"e"` or `"p"`); run directory is resolved from the job record  
* `read_fourier_spectrum(job_id)` → returns the final absorption spectrum data (`spectrum.csv`)  
* `reference_spectrum(molecule, condition)` → returns trusted reference spectrum at a few energies for a few conditions  
* `list_bug_categories()` → returns valid `bug_category` strings  
* `submit_diagnosis(soln_desc, strategy_desc, confidence, bug_category, broken_lines, fixed_lines)` → writes `submission.json`

When submitting your findings to `submit_diagnosis(...)`, it must cover:

* `soln_desc` — location and nature of the fix  
* `strategy_desc` — files inspected, tests run, and how other candidates were ruled out  
* `confidence` — percentage confidence (0–100)  
* `bug_category` — one of the category strings above  
* `broken_lines` / `fixed_lines` — exact buggy and corrected source line(s)

If everything is working as expected, use `bug_category="no_fault"` with empty `broken_lines` and `fixed_lines`.