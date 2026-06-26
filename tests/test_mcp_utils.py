import sys
from pathlib import Path

import pytest

MCP_DIR = Path(__file__).resolve().parent.parent / "mcp"
sys.path.insert(0, str(MCP_DIR))

from utils import require_job_dir, resolve_job_path_in_dir


@pytest.fixture
def job_tree(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "Na" / "COM"
    job_dir.mkdir(parents=True)
    input_file = job_dir / "input.json"
    input_file.write_text("{}")

    import utils

    monkeypatch.setattr(utils, "JOBS_DIR", jobs_dir)
    return job_dir, input_file


def test_require_job_dir_resolves_valid_directory(job_tree):
    job_dir, _ = job_tree
    assert require_job_dir(str(job_dir)) == job_dir.resolve()


def test_require_job_dir_rejects_path_outside_jobs(job_tree):
    job_dir, _ = job_tree
    with pytest.raises(ValueError, match="Job directory must be under"):
        require_job_dir(str(job_dir.parent.parent.parent))


def test_require_job_dir_rejects_missing_directory(job_tree):
    job_dir, _ = job_tree
    missing = job_dir / "missing"
    with pytest.raises(ValueError, match="Job directory does not exist"):
        require_job_dir(str(missing))


def test_resolve_job_path_in_dir_relative(job_tree):
    job_dir, input_file = job_tree
    assert resolve_job_path_in_dir("input.json", job_dir) == input_file.resolve()


def test_resolve_job_path_in_dir_absolute(job_tree):
    job_dir, input_file = job_tree
    assert resolve_job_path_in_dir(str(input_file), job_dir) == input_file.resolve()


def test_resolve_job_path_in_dir_rejects_path_outside_jobs(job_tree):
    job_dir, _ = job_tree
    outside = job_dir.parent.parent.parent / "outside.json"
    with pytest.raises(ValueError, match="Path is outside the jobs directory"):
        resolve_job_path_in_dir(str(outside), job_dir)