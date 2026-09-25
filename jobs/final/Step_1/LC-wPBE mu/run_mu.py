#!/usr/bin/env python3
"""IP-tune LC-ωPBE μ for every Step 1 molecule with the plasmol tune driver.

One folder per molecule. The geometry is the PBE0/6-311G* structure in
../Molecules/<name>/opt.xyz. The driver minimizes |E(cation) - E(neutral) + ε_HOMO|
and does not propagate. O2 is the triplet (spin = 2). Nothing else is open shell.
eps0 is not tuned.

    python jobs/final/Step_1/mu/run_mu.py prepare
    python jobs/final/Step_1/mu/run_mu.py run
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
STEP = HERE.parent
PYTHON = "/Users/bldrdge1/miniconda3/envs/plasmol_env_1eb6e532/bin/python"
SLOTS = 4

# folder, pyscf spin (multiplicity - 1)
MOLECULES = [
    ("n2", 0),
    ("co", 0),
    ("co2", 0),
    ("n2o", 0),
    ("o2", 2),
    ("h2o", 0),
    ("nh3", 0),
    ("ch4", 0),
    ("h2co", 0),
    ("c2h2", 0),
    ("c2h4", 0),
    ("c2h6", 0),
    ("c6h6", 0),
    ("para_aminophenol", 0),
    ("ortho_aminophenol", 0),
    ("meta_aminophenol", 0),
    ("h2s", 0),
    ("so2", 0),
    ("cs2", 0),
    ("lif", 0),
    ("beo", 0),
    ("bf", 0),
    ("pyrimidine", 0),
    ("purine", 0),
    ("uracil", 0),
    ("cytosine", 0),
    ("thymine", 0),
    ("adenine", 0),
    ("guanine", 0),
    ("formamide", 0),
    ("pentan3one", 0),
    ("pentan2one", 0),
    ("pentanal", 0),
    ("transthioindigo", 0),
]

MU_RE = re.compile(r"Optimal μ \(lrc_parameter\) = ([0-9.eE+-]+)")


def job_dir(name: str) -> Path:
    return HERE / name


def write_input(name: str, spin: int) -> None:
    geometry = STEP / "Molecules" / name / "opt.xyz"
    if not geometry.exists():
        raise FileNotFoundError(geometry)
    folder = job_dir(name)
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": {
            "dt": 0.05,
            "t_end": 1.0,
            "driver": "tune",
        },
        "molecule": {
            "geometry": os.path.relpath(geometry, folder),
            "geometry_units": "angstrom",
            "charge": 0,
            "spin": spin,
            "basis": "6-311G*",
            "xc": "HYB_GGA_XC_LC_WPBE",
            "lrc_parameter": "tune",
        },
        "files": {
            "field_e_filepath": "field_e.csv",
            "field_p_filepath": "field_p.csv",
            "spectra_e_vs_p_filepath": "output.png",
        },
    }
    (folder / "tune.json").write_text(json.dumps(payload, indent=2) + "\n")


def read_mu(log_path: Path):
    if not log_path.exists():
        return None
    found = MU_RE.findall(log_path.read_text(errors="replace"))
    if not found:
        return None
    return float(found[-1])


def write_table() -> None:
    rows = ["molecule\tmu"]
    for name, _spin in MOLECULES:
        mu = read_mu(job_dir(name) / "log.out")
        rows.append(f"{name}\t{'' if mu is None else f'{mu:.6f}'}")
    (HERE / "mu.txt").write_text("\n".join(rows) + "\n")


def prepare(only: list[str] | None = None) -> None:
    for name, spin in MOLECULES:
        if only and name not in only:
            continue
        write_input(name, spin)
        print(name)


def run(only: list[str] | None = None) -> int:
    selected = [name for name, _spin in MOLECULES if only is None or name in only]
    log_path = HERE / "screen.log"
    log_path.write_text("")

    def log(message: str) -> None:
        with log_path.open("a") as handle:
            handle.write(message + "\n")
        print(message, flush=True)

    pending = list(selected)
    running = {}
    failed = []
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "2",
    })

    def start(name: str) -> None:
        folder = job_dir(name)
        handle = (folder / "launch.out").open("w")
        proc = subprocess.Popen(
            [PYTHON, "-m", "plasmol.main", str(folder / "tune.json"), "--log", "log.out"],
            cwd=REPO,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        running[name] = {"proc": proc, "handle": handle, "t0": time.time()}
        log(f"START {name} pid={proc.pid}")

    while pending or running:
        while pending and len(running) < SLOTS:
            name = pending.pop(0)
            if read_mu(job_dir(name) / "log.out") is not None:
                log(f"SKIP {name} mu={read_mu(job_dir(name) / 'log.out'):.6f}")
                continue
            start(name)
        if not running:
            break
        time.sleep(5)
        done = [name for name, job in running.items() if job["proc"].poll() is not None]
        for name in done:
            job = running.pop(name)
            job["handle"].close()
            mu = read_mu(job_dir(name) / "log.out")
            elapsed = time.time() - job["t0"]
            if job["proc"].returncode == 0 and mu is not None:
                log(f"OK {name} {elapsed:.0f}s mu={mu:.6f}")
            else:
                failed.append(name)
                log(f"FAIL {name} exit={job['proc'].returncode}")
        write_table()
    write_table()
    if failed:
        log("ALL_DONE with failures: " + " ".join(failed))
        return 1
    log("ALL_DONE")
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    names = sys.argv[2:] or None
    if command == "prepare":
        prepare(names)
    elif command == "run":
        prepare(names)
        raise SystemExit(run(names))
    else:
        raise SystemExit(f"usage: {sys.argv[0]} prepare|run [names...]")
