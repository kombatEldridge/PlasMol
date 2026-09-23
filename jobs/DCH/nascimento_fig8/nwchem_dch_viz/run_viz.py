#!/usr/bin/env python3
"""Phase A (0-40 au dumps) then Phase B (to 206.7 au) if the projector is green."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NWCHEM = "/opt/homebrew/bin/nwchem"
PY = "/Users/brinton/miniconda/envs/meep/bin/python"
STATUS = HERE / "viz_status.json"


def write_status(payload: dict) -> None:
    STATUS.write_text(json.dumps(payload, indent=2) + "\n")


def run_nwchem(inp: str, log_name: str) -> int:
    log = HERE / log_name
    print(f"START nwchem {inp} -> {log_name}", flush=True)
    with log.open("a") as fh:
        proc = subprocess.Popen(
            [NWCHEM, inp],
            cwd=str(HERE),
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
        write_status({"state": "running", "phase": inp, "pid": proc.pid})
        return proc.wait()


def project() -> dict:
    print("PROJECT dumps", flush=True)
    proc = subprocess.run(
        [PY, str(HERE / "project_ptot.py"), str(HERE)],
        cwd=str(HERE),
        capture_output=True,
        text=True,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    verdict_path = HERE / "phase_verdict.json"
    if verdict_path.exists():
        return json.loads(verdict_path.read_text())
    return {"green": False, "error": proc.stderr[-500:], "rc": proc.returncode}


def main() -> int:
    os.chdir(HERE)
    write_status({"state": "running", "phase": "A"})
    rc = run_nwchem("3p_dch_viz_A.nw", "phase_A.out")
    if rc != 0:
        write_status({"state": "failed", "phase": "A", "rc": rc})
        print(f"FAIL phase A rc={rc}", flush=True)
        return 1

    verdict = project()
    write_status({"state": "phase_A_done", "verdict": verdict})
    print("PHASE_A_VERDICT", json.dumps(verdict), flush=True)
    if not verdict.get("green"):
        write_status({"state": "red", "verdict": verdict})
        print("RED: Phase A did not pass t=0 / period checks. Stopping.", flush=True)
        return 2

    print("GREEN: starting Phase B", flush=True)
    rc = run_nwchem("3p_dch_viz_B.nw", "phase_B.out")
    if rc != 0:
        write_status({"state": "failed", "phase": "B", "rc": rc})
        print(f"FAIL phase B rc={rc}", flush=True)
        return 1

    verdict = project()
    write_status({"state": "done", "verdict": verdict})
    print("PHASE_B_VERDICT", json.dumps(verdict), flush=True)
    print("ALL_DONE", flush=True)
    return 0 if verdict.get("green") else 2


if __name__ == "__main__":
    sys.exit(main())
