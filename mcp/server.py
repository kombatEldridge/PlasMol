import uuid
import threading
import subprocess
import atexit
import sys
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from mcp.server.fastmcp import FastMCP

MCP_DIR = Path(__file__).resolve().parent
BASE_DIR = MCP_DIR.parent
for path in (BASE_DIR, MCP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from plasmol.utils.params import PARAMS

from utils import (
    CONDA_ENV,
    cleanup_empty_stderr,
    require_job_dir,
    resolve_job_path,
    resolve_job_path_in_dir,
    stage_input,
)

try:
    import meep as mp
    atexit.unregister(mp.report_elapsed_time)
except Exception:
    pass

mcp = FastMCP("agent-server")
jobs = {}


def _run_job(job_id: str, input_file: str, run_dir: Path):
    jobs[job_id]["status"] = "running"
    jobs[job_id]["started_at"] = datetime.now().isoformat()
    jobs[job_id]["run_dir"] = str(run_dir)

    stderr_file = run_dir / "stderr.log"

    try:
        src = Path(input_file).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        dst = stage_input(src, run_dir)
        jobs[job_id]["input_file"] = str(dst)

        run_dir = run_dir.resolve()
        cmd = [
            "conda", "run", "-n", CONDA_ENV, "--cwd", str(run_dir),
            "python", "-m", "plasmol.main",
            dst.name, "-l", "log.out",
        ]

        with open(stderr_file, "w") as err:
            proc = subprocess.Popen(
                cmd,
                cwd=str(run_dir),
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
    finally:
        cleanup_empty_stderr(stderr_file)


@mcp.tool()
def parse_params(input_json_path: str) -> str:
    """Return the parsed input params."""
    try:
        abs_path = resolve_job_path(input_json_path)
        args = SimpleNamespace(input=str(abs_path))
        params = PARAMS(args)
        lines = []
        for key in sorted(vars(params).keys()):
            value = getattr(params, key)
            lines.append(f"{key}: {value}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def run_simulation(job_dir: str, input_json_path: str) -> dict:
    """Submit a full PlasMol run in the given job directory."""
    try:
        run_dir = require_job_dir(job_dir)
        abs_path = resolve_job_path_in_dir(input_json_path, run_dir)
    except Exception as e:
        return {"error": str(e)}

    job_id = str(uuid.uuid4())[:8]

    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "conda_env": CONDA_ENV,
        "input_file_path": str(abs_path),
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(),
    }

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, str(abs_path), run_dir),
        daemon=True,
    )
    thread.start()

    return {
        "job_id": job_id,
        "status": "running",
        "conda_env": CONDA_ENV,
        "message": "Job started in background. Use list_jobs() to check progress.",
        "run_directory": str(run_dir),
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
    return {"message": f"Job {job_id} is not running."}


if __name__ == "__main__":
    mcp.run()