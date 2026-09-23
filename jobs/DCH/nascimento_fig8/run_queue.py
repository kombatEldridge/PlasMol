#!/usr/bin/env python3
"""Launch NWChem DCH + the PySCF one-factor sweep with a limited process pool."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PY = "/Users/brinton/miniconda/envs/meep/bin/python"
NWCHEM = "/opt/homebrew/bin/nwchem"
STATUS = HERE / "queue_status.json"
PLASMOL_SLOTS = 4

NWCHEM_JOB = {
    "name": "nwchem",
    "cwd": HERE / "nwchem_dch",
    "cmd": [NWCHEM, "3p_dch.nw"],
    "log": HERE / "nwchem_dch" / "3p_dch.out",
    "env": {},
}

PLASMOL_QUEUE = [
    "cart",
    "grid5",
    "hf20",
    "rk4",
    "grid7",
    "hf30",
    "dt025",
]


def plasmol_job(name: str) -> dict:
    cwd = HERE / "pyscf_sweep" / name
    env = {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "PYTHONPATH": str(REPO),
    }
    return {
        "name": name,
        "cwd": cwd,
        "cmd": [PY, "-m", "plasmol.main", "3p.json", "-l", "log.out"],
        "log": cwd / "launch.out",
        "env": env,
    }


def write_status(payload: dict) -> None:
    STATUS.write_text(json.dumps(payload, indent=2) + "\n")


def start(job: dict) -> subprocess.Popen:
    job["log"].parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(job["env"])
    logf = open(job["log"], "a")
    proc = subprocess.Popen(
        job["cmd"],
        cwd=str(job["cwd"]),
        stdout=logf,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    job["logf"] = logf
    job["proc"] = proc
    job["pid"] = proc.pid
    job["started"] = time.time()
    print(f"START {job['name']} pid={proc.pid}", flush=True)
    return proc


def finished_ok(job: dict) -> bool:
    if job["name"] == "nwchem":
        text = job["log"].read_text(errors="replace") if job["log"].exists() else ""
        return "Total times" in text and "Propagation started" in text and "failed" not in text.lower()[-400:]
    csv = job["cwd"] / "mo_occ.csv"
    if not csv.exists():
        return False
    last = ""
    for line in csv.read_text().splitlines()[::-1]:
        if line and not line.startswith("#") and not line.startswith("Timestamps"):
            last = line
            break
    if not last:
        return False
    try:
        t = float(last.split(",")[0])
    except ValueError:
        return False
    return t >= 206.0


def main() -> int:
    pending = [plasmol_job(n) for n in PLASMOL_QUEUE]
    running: list[dict] = []
    done: list[dict] = []
    failed: list[dict] = []

    def snapshot(state: str) -> None:
        write_status({
            "state": state,
            "running": [j["name"] for j in running],
            "pending": [j["name"] for j in pending],
            "done": [j["name"] for j in done],
            "failed": [j["name"] for j in failed],
            "pids": {j["name"]: j.get("pid") for j in running},
        })

    def handle_sig(signum, frame):
        for j in running:
            try:
                os.killpg(j["pid"], signal.SIGTERM)
            except OSError:
                pass
        snapshot("killed")
        sys.exit(1)

    signal.signal(signal.SIGTERM, handle_sig)
    signal.signal(signal.SIGINT, handle_sig)

    start(NWCHEM_JOB)
    running.append(NWCHEM_JOB)
    while pending and sum(1 for j in running if j["name"] != "nwchem") < PLASMOL_SLOTS:
        job = pending.pop(0)
        start(job)
        running.append(job)
    snapshot("running")

    while running:
        time.sleep(30)
        still = []
        for job in running:
            rc = job["proc"].poll()
            if rc is None:
                still.append(job)
                continue
            job["logf"].close()
            job["rc"] = rc
            if rc == 0 and finished_ok(job):
                print(f"DONE {job['name']} rc=0", flush=True)
                done.append(job)
            else:
                print(f"FAIL {job['name']} rc={rc}", flush=True)
                failed.append(job)
        running = still
        while pending and sum(1 for j in running if j["name"] != "nwchem") < PLASMOL_SLOTS:
            job = pending.pop(0)
            start(job)
            running.append(job)
        snapshot("failed" if failed else "running")
        if failed:
            for j in running:
                try:
                    os.killpg(j["pid"], signal.SIGTERM)
                except OSError:
                    pass
            snapshot("failed")
            return 1

    snapshot("done")
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
