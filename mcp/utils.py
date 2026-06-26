import json
import os
import re
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
JOBS_DIR = BASE_DIR / "jobs"
CONDA_ENV = "meep"

def require_job_cwd() -> Path:
    """Return the active job directory from PLASMOL_JOB_DIR or process cwd."""
    job_dir = os.environ.get("PLASMOL_JOB_DIR", "").strip()
    if job_dir:
        cwd = Path(job_dir).resolve()
    else:
        cwd = Path.cwd().resolve()
    if not cwd.is_relative_to(JOBS_DIR):
        raise ValueError(
            f"MCP tools must run from a directory under {JOBS_DIR}. "
            f"Set PLASMOL_JOB_DIR or start the server from a job folder. "
            f"Current directory: {cwd}"
        )
    return cwd


def resolve_job_path(path: str) -> Path:
    """Resolve an input path under jobs/, using PLASMOL_JOB_DIR/cwd for relative paths."""
    return resolve_job_path_in_dir(path, require_job_cwd())


def require_job_dir(job_dir: str) -> Path:
    """Validate and resolve a job directory path under jobs/."""
    resolved = Path(job_dir).resolve()
    if not resolved.is_relative_to(JOBS_DIR):
        raise ValueError(f"Job directory must be under {JOBS_DIR}. Got: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"Job directory does not exist: {resolved}")
    return resolved


def resolve_job_path_in_dir(path: str, job_dir: Path) -> Path:
    """Resolve an input path under jobs/, relative to the given job directory."""
    abs_path = Path(path)
    if not abs_path.is_absolute():
        abs_path = (job_dir / path).resolve()
    else:
        abs_path = abs_path.resolve()
    if not abs_path.is_relative_to(JOBS_DIR):
        raise ValueError("Path is outside the jobs directory")
    return abs_path


def load_json_strip_comments(path: Path) -> dict:
    with open(path, "r") as f:
        content = "".join(
            re.sub(r"(#|--|%|//)(.*)$", "", line)
            for line in f
            if not line.strip().startswith(("#", "--", "%", "//"))
        )
    return json.loads(content)


def cleanup_empty_stderr(stderr_file: Path) -> None:
    if stderr_file.exists() and not stderr_file.read_text().strip():
        stderr_file.unlink()


def stage_input(src: Path, run_dir: Path) -> Path:
    """Ensure the input JSON and any referenced geometry files are usable from run_dir."""
    if src.parent.resolve() != run_dir.resolve():
        dst = run_dir / src.name
        shutil.copy2(src, dst)
    else:
        dst = src

    data = load_json_strip_comments(dst)
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