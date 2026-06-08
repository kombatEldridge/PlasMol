import uuid
import threading
import subprocess
import os
import shutil
import json
import atexit
import re
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from mcp.server.fastmcp import FastMCP

from plasmol.utils.input.params import PARAMS

try:
    import meep as mp
    atexit.unregister(mp.report_elapsed_time)
except Exception:
    pass

mcp = FastMCP("agent-server")
jobs = {}
BASE_DIR = Path.cwd().resolve()
BASE_RUNS_DIR = BASE_DIR / "mcp_runs"
CONDA_ENV = ""
SETUP_JOB_ID = 0

def _create_run_directory(job_id: str) -> Path:
    """Create a fresh directory for this run."""
    run_dir = BASE_RUNS_DIR / f"run_{job_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def _load_json_strip_comments(path: Path) -> dict:
    with open(path, "r") as f:
        content = "".join(
            re.sub(r"(#|--|%|//)(.*)$", "", line)
            for line in f
            if not line.strip().startswith(("#", "--", "%", "//"))
        )
    return json.loads(content)

def _stage_input_into_run_dir(src: Path, run_dir: Path) -> Path:
    """Copy the input JSON and any referenced files into the run directory."""
    dst = run_dir / src.name
    shutil.copy2(src, dst)

    data = _load_json_strip_comments(dst)
    changed = False

    molecule = data.get("molecule") or {}
    geometry = molecule.get("geometry")
    if isinstance(geometry, str):
        geom_src = Path(geometry)
        if not geom_src.is_absolute():
            geom_src = (src.parent / geometry).resolve()
        if geom_src.exists():
            geom_dst = run_dir / geom_src.name
            if geom_src != geom_dst:
                shutil.copy2(geom_src, geom_dst)
            molecule["geometry"] = geom_dst.name
            changed = True

    if changed:
        with open(dst, "w") as f:
            json.dump(data, f, indent=2)

    return dst

def _run_job(job_id: str, conda_env: str, input_file: str, run_dir: Path):
    """Background worker that does the actual work."""
    jobs[job_id]["status"] = "running"
    jobs[job_id]["started_at"] = datetime.now().isoformat()
    jobs[job_id]["run_dir"] = str(run_dir)

    stdout_file = run_dir / "stdout.log"
    stderr_file = run_dir / "stderr.log"

    try:
        src = Path(input_file).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        dst = _stage_input_into_run_dir(src, run_dir)
        jobs[job_id]["copied_file"] = str(dst)

        if not conda_env:
            raise RuntimeError(
                "No conda environment configured. Run setup_conda_environment() first "
                "or set CONDA_ENV to an existing environment with pymeep installed."
            )

        run_dir = run_dir.resolve()
        cmd = [
            "conda", "run", "-n", conda_env, "--cwd", str(run_dir),
            "python", "-m", "plasmol.main",
            dst.name, "-vv", "-l", "log.out",
        ]

        with open(stdout_file, "w") as out, open(stderr_file, "w") as err:
            proc = subprocess.Popen(
                cmd,
                cwd=str(run_dir),
                stdout=out,
                stderr=err,
                text=True,
            )
            jobs[job_id]["pid"] = proc.pid
            jobs[job_id]["proc"] = proc
            exit_code = proc.wait()
            jobs[job_id]["exit_code"] = exit_code
            jobs[job_id]["status"] = "completed" if exit_code == 0 else "failed"
            jobs[job_id]["finished_at"] = datetime.now().isoformat()

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["finished_at"] = datetime.now().isoformat()
        with open(stderr_file, "a") as f:
            f.write(f"\n[ERROR] {str(e)}\n")

def _run_conda_setup_job(job_id: str, log_dir: Path, env_name: str):
    """Background worker that runs your exact conda commands."""
    global CONDA_ENV
    jobs[job_id]["status"] = "running"
    jobs[job_id]["started_at"] = datetime.now().isoformat()
    stdout_file = log_dir / "stdout.log"
    stderr_file = log_dir / "stderr.log"
    try:
        setup_cmds = [
            ["conda", "create", "-n", env_name, "-y"],
            [
                "conda", "install", "-n", env_name, "-c", "conda-forge",
                "pymeep", "pyscf", "pip", "-y",
            ],
            ["conda", "run", "-n", env_name, "pip", "install", "-e", str(BASE_DIR)],
        ]
        exit_code = 0
        with open(stdout_file, "w") as out, open(stderr_file, "w") as err:
            for cmd in setup_cmds:
                if exit_code != 0:
                    break
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(BASE_DIR),
                    stdout=out,
                    stderr=err,
                    text=True,
                )
                exit_code = proc.wait()
        jobs[job_id]["exit_code"] = exit_code
        if exit_code == 0:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["message"] = f"✅ '{env_name}' conda environment is ready."
        else:
            if CONDA_ENV == env_name:
                CONDA_ENV = ""
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = f"❌ '{env_name}' conda environment setup failed."
        jobs[job_id]["finished_at"] = datetime.now().isoformat()
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["finished_at"] = datetime.now().isoformat()

def _job_log_dir(job_id: str) -> Path:
    job = jobs[job_id]
    job_type = job.get("type")
    if job_type == "conda_setup":
        return BASE_RUNS_DIR / f"setup_conda_{job_id}"
    if job_type == "confirm_conda":
        return BASE_RUNS_DIR / f"confirm_conda_{job_id}"
    return Path(job.get("run_dir", BASE_RUNS_DIR / f"run_{job_id}"))

@mcp.tool()
def get_cwd() -> str:
    """Get the current working directory."""
    return str(Path.cwd())

@mcp.tool()
def setup_conda_environment() -> dict:
    """One-time setup for the conda environment containing meep."""
    global CONDA_ENV
    global SETUP_JOB_ID
    job_id = str(uuid.uuid4())[:8]
    env_name = f"plasmol_env_{job_id}"
    CONDA_ENV = env_name
    SETUP_JOB_ID = job_id

    jobs[job_id] = {
        "job_id": job_id,
        "type": "conda_setup",
        "status": "pending",
        "env_name": env_name,
        "created_at": datetime.now().isoformat(),
    }

    # Create log folder for this setup
    log_dir = BASE_RUNS_DIR / f"setup_conda_{job_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    thread = threading.Thread(
        target=_run_conda_setup_job,
        args=(job_id, log_dir, env_name),
        daemon=True
    )
    thread.start()

    return {
        "job_id": job_id,
        "status": "running",
        "conda_env": env_name,
        "message": f"Setting up '{env_name}' conda environment...",
        "log_directory": str(log_dir),
        "note": "CONDA_ENV is set. Wait for setup to complete before run_simulation(). "
               "Check progress with list_jobs() or confirm_conda_environment_setup().",
    }

@mcp.tool()
def confirm_conda_environment_setup() -> dict:
    """Confirm that the conda environment is set up."""
    global CONDA_ENV
    global SETUP_JOB_ID
    if not CONDA_ENV:
        return {
            "error": "No conda environment configured. Run setup_conda_environment() first.",
        }

    if not SETUP_JOB_ID:
        return {
            "error": "No conda setup job found. Run setup_conda_environment() first.",
        }

    if SETUP_JOB_ID not in jobs:
        return {
            "error": f"Conda setup job '{SETUP_JOB_ID}' not found.",
            "setup_job_id": SETUP_JOB_ID,
        }

    setup_status = jobs[SETUP_JOB_ID].get("status")
    if setup_status in {"pending", "running"}:
        return {
            "error": "Conda setup job is not finished.",
            "setup_job_id": SETUP_JOB_ID,
            "status": setup_status,
            "message": (
                f"Setup job '{SETUP_JOB_ID}' is still {setup_status}. "
                "Check progress with list_jobs() or read_log()."
            ),
        }

    if setup_status != "completed":
        return {
            "error": "Conda setup job did not complete successfully.",
            "setup_job_id": SETUP_JOB_ID,
            "status": setup_status,
            "message": jobs[SETUP_JOB_ID].get("error") or jobs[SETUP_JOB_ID].get("message", ""),
        }

    env_name = CONDA_ENV
    job_id = str(uuid.uuid4())[:8]

    jobs[job_id] = {
        "job_id": job_id,
        "type": "confirm_conda",
        "status": "pending",
        "env_name": env_name,
        "created_at": datetime.now().isoformat(),
    }

    # Create log folder for this setup
    log_dir = BASE_RUNS_DIR / f"confirm_conda_{job_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs[job_id]["status"] = "running"
    jobs[job_id]["started_at"] = datetime.now().isoformat()
    stdout_file = log_dir / "stdout.log"
    stderr_file = log_dir / "stderr.log"
    try:
        check_cmd = [
            "conda", "run", "-n", env_name,
            "python", "-c", "import meep as mp; print(mp.__version__)"
        ]
        with open(stdout_file, "w") as out, open(stderr_file, "w") as err:
            proc = subprocess.Popen(
                check_cmd,
                cwd=Path.cwd(),
                stdout=out,
                stderr=err,
                text=True,
            )
            exit_code = proc.wait()
        
        with open(stdout_file, "r") as f:
            stdout_content = f.read().strip()

        version_line = stdout_content.splitlines()[0].strip() if stdout_content else ""
        env_ready = exit_code == 0 and bool(version_line) and version_line[0].isdigit()
        jobs[job_id]["exit_code"] = exit_code
        jobs[job_id]["finished_at"] = datetime.now().isoformat()
        if env_ready:
            CONDA_ENV = env_name
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["message"] = f"✅ '{env_name}' conda environment is ready (meep {version_line})."
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = f"meep not found or import failed. Got: {stdout_content}"
            jobs[job_id]["message"] = f"❌ '{env_name}' conda environment setup failed."

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["finished_at"] = datetime.now().isoformat()

    return jobs[job_id]

@mcp.tool()
def lab_manual() -> str:
    """Return the manual for the lab."""
    # gather certain .md files from doc/docs/. and comppile them into one string block to return.
    md_files = ["doc/docs/index.md", "doc/docs/about.md", "doc/docs/installation.md", "doc/docs/usage.md", "doc/docs/methodology.md"]
    manual_content = ""
    for file in md_files:
        manual_content += "# File: " + file + "\n"
        with open(file, "r") as f:
            manual_content += f.read()
    return manual_content

@mcp.tool()
def list_dir(path: str) -> str:
    """List files and subdirectories at a given path."""
    try:
        abs_path = Path(path).resolve()
        if not abs_path.is_relative_to(BASE_DIR):
            raise ValueError("Path is outside the allowed directory")
        return "\n".join(os.listdir(abs_path))
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def read_file(path: str, start_line: int, end_line: int) -> str:
    """Read a range of lines from a source file. `start_line` is 1-indexed, `end_line` is not included."""
    with open(path, "r") as f:
        lines = f.readlines()
    return "".join(lines[start_line-1:end_line-1])

@mcp.tool()
def search_code(pattern: str, path: str) -> str:
    """Grep-style search over source files for a string or regex."""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"

    results = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith(".py"):
                filepath = os.path.join(root, filename)
                with open(filepath, "r") as f:
                    for i, line in enumerate(f, start=1):
                        if regex.search(line):
                            results.append(f"{filepath}:{i}: {line.strip()}")
    return "\n".join(results)

@mcp.tool()
def edit_file(path: str, old_line: str, new_line: str) -> str:
    """Replace exactly one line in a source file."""
    with open(path, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line == old_line:
            lines[i] = new_line
            with open(path, "w") as f:
                f.writelines(lines)
            return f"Updated line {i + 1} in {path}"
    return f"Error: line not found in {path}"

@mcp.tool()
def read_json(input_json_path: str) -> str:
    """Return the raw input text."""
    with open(input_json_path, "r") as f:
        return f.read()

@mcp.tool()
def parse_params(input_json_path: str) -> str:
    """Return the parsed input params."""
    args = SimpleNamespace(input=input_json_path)
    params = PARAMS(args)
    lines = []
    for key in sorted(vars(params).keys()):
        value = getattr(params, key)
        lines.append(f"{key}: {value}")
    return "\n".join(lines)

@mcp.tool()
def run_simulation(input_json_path: str) -> dict:
    """Submit a full PlasMol run."""
    global CONDA_ENV
    job_id = str(uuid.uuid4())[:8]   # short readable ID

    confirm = confirm_conda_environment_setup()
    if confirm.get("status") != "completed":
        return {
            "error": confirm.get("error")
            or confirm.get("message")
            or "Failed to confirm conda environment is set up.",
        }

    # Initialize job record
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "conda_env": CONDA_ENV,
        "input_file_path": input_json_path,
        "created_at": datetime.now().isoformat(),
    }

    # Create isolated directory
    run_dir = _create_run_directory(job_id)

    # Start background execution
    thread = threading.Thread(
        target=_run_job,
        args=(job_id, CONDA_ENV, input_json_path, run_dir),
        daemon=True
    )
    thread.start()

    return {
        "job_id": job_id,
        "status": "running",
        "conda_env": CONDA_ENV,
        "message": f"Job started in background. Use list_jobs() to check progress.",
        "run_directory": str(run_dir),
    }

@mcp.tool()
def read_log(job_id: str) -> dict:
    """Read the stdout and stderr."""
    if job_id not in jobs:
        return {"error": "Job not found"}

    log_dir = _job_log_dir(job_id)
    stdout_file = log_dir / "stdout.log"
    stderr_file = log_dir / "stderr.log"

    stdout = stdout_file.read_text() if stdout_file.exists() else ""
    stderr = stderr_file.read_text() if stderr_file.exists() else ""

    return {
        "job_id": job_id,
        "status": jobs[job_id].get("status"),
        "stdout": stdout,
        "stderr": stderr,
    }

@mcp.tool()
def list_jobs() -> list:
    """List all jobs (most recent first)."""
    return sorted(jobs.values(), key=lambda x: x.get("created_at", ""), reverse=True)

@mcp.tool()
def kill_job(job_id: str) -> dict:
    """Kill a running job."""
    if job_id not in jobs:
        return {"error": "Job not found"}

    proc = jobs[job_id].get("proc")
    if proc and proc.poll() is None:
        proc.terminate()
        jobs[job_id]["status"] = "cancelled"
        return {"message": f"Job {job_id} terminated."}
    else:
        return {"message": f"Job {job_id} is not running."}

@mcp.tool()
def read_field_csv(job_id: str, field: str) -> str:
    """Return the electric field or polarization CSV data for a given job."""
    if job_id not in jobs:
        return f"Error: Job '{job_id}' not found"

    job = jobs[job_id]
    if job.get("status") != "completed":
        return "Job not completed. Field data not available."

    run_dir = Path(job.get("run_dir", BASE_RUNS_DIR / f"run_{job_id}"))
    input_file_path = run_dir / Path(job["input_file_path"]).name
    if not input_file_path.exists():
        input_file_path = Path(job["input_file_path"]).resolve()

    with open(input_file_path, "r") as f:
        content = "".join(
            re.sub(r"(#|--|%|//)(.*)$", "", line)
            for line in f
            if not line.strip().startswith(("#", "--", "%", "//"))
        )
        data = json.loads(content)

    field_key = f"field_{field}_filepath"
    if "files" not in data or field_key not in data["files"]:
        return f"Error: '{field_key}' not found in input JSON"

    field_csv = run_dir / data["files"][field_key]
    if not field_csv.exists():
        return f"Error: {field_csv} not found"

    return field_csv.read_text()

@mcp.tool()
def read_fourier_spectrum(job_id: str) -> str:
    """Return the final absorption spectrum data for a given job."""
    if job_id not in jobs:
        return f"Error: Job '{job_id}' not found"

    if jobs[job_id].get("status") != "completed":
        return "Job not completed. Spectrum not available."
    
    jobs[job_id][""]

    run_dir = Path(jobs[job_id].get("run_dir", BASE_RUNS_DIR / f"run_{job_id}"))
    spectrum_file = run_dir / "spectrum.csv"
    if not spectrum_file.exists():
        return f"Error: spectrum.csv not found in {run_dir}"

    return spectrum_file.read_text()

@mcp.tool()
def reference_spectrum(molecule: str, condition: str) -> str:
    """Return the trusted reference spectrum at a few energies for a few conditions."""
    reference_data = {
        ("molecule1", "conditionA"): "energy (eV),absorption peaks (arb. units)\n1.0,0.5\n2.0,0.8\n",
        ("molecule1", "conditionB"): "energy (eV),absorption peaks (arb. units)\n1.0,0.6\n2.0,0.9\n",
        ("molecule2", "conditionA"): "energy (eV),absorption peaks (arb. units)\n1.0,0.4\n2.0,0.7\n",
        ("molecule2", "conditionB"): "energy (eV),absorption peaks (arb. units)\n1.0,0.3\n2.0,0.6\n",
    }
    failed_statement = "Reference data not available. The molecules and conditions in the reference dataset are {}.".format(", ".join([f"{m[0]} under {m[1]}" for m in reference_data.keys()]))
    return reference_data.get((molecule, condition), failed_statement)

@mcp.tool()
def submit_diagnosis(soln_desc: str, strategy_desc: str, uncertainty: float) -> str:
    """Return the preformatted RESULTS.md."""
    # Form a preformatted markdown string that includes the diagnosis, strategy, and uncertainty.
    diagnosis_md = f"""# Diagnosis
**Solution Description:** {soln_desc}
**Proposed Strategy:** {strategy_desc}
**Uncertainty Level:** {uncertainty:.2f}
"""
    with open("RESULTS.md", "w") as f:
        f.write(diagnosis_md)
    return "RESULTS.md updated."

if __name__ == "__main__":
    BASE_RUNS_DIR.mkdir(exist_ok=True)
    mcp.run()
