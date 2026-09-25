#!/usr/bin/env python3
"""Rerun Step 1 linear-response TDDFT with each molecule's tuned μ.

Geometries are the existing PBE0 opt.xyz files. Nothing is reoptimized.
The old fixed-μ spectra are kept as absorption_mu0.34272.txt.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

import run_screen as rs

HERE = Path(__file__).resolve().parent
SLOTS = 3


def load_mu() -> dict[str, float]:
    mus = {}
    for line in (HERE / "mu" / "mu.txt").read_text().splitlines()[1:]:
        if not line.strip():
            continue
        name, value = line.split("\t")
        mus[name] = float(value)
    missing = [name for name, *_ in rs.MOLECULES if name not in mus]
    if missing:
        raise SystemExit("missing μ for " + ", ".join(missing))
    return mus


def load_xyz(path: Path):
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    count = int(lines[0].split()[0])
    atoms = []
    for line in lines[2:2 + count]:
        symbol, x, y, z = line.split()[:4]
        atoms.append((symbol, (float(x), float(y), float(z))))
    if len(atoms) != count:
        raise ValueError(f"short xyz: {path}")
    return atoms


def nroots_for(folder: Path, default: int) -> int:
    deck = folder / "tddft.nw"
    if not deck.exists():
        return default
    match = re.search(r"nroots\s+(\d+)", deck.read_text())
    return int(match.group(1)) if match else default


def prepare() -> None:
    mus = load_mu()
    for name, label, _smiles, mult, nroots in rs.MOLECULES:
        folder = HERE / name
        old = folder / "absorption.txt"
        backup = folder / "absorption_mu0.34272.txt"
        if old.exists() and not backup.exists():
            backup.write_text(old.read_text())
        atoms = load_xyz(folder / "opt.xyz")
        roots = nroots_for(folder, nroots)
        rs.write_tddft(folder, label, mult, roots, atoms, mus[name])
        out = folder / "tddft.out"
        if out.exists() and f"mu={mus[name]:.6f}" not in out.read_text(errors="replace"):
            out.unlink()
        print(f"{name:20} mu={mus[name]:.6f}  roots={roots}")


def finished(folder: Path, mu: float) -> bool:
    out = folder / "tddft.out"
    if not out.exists():
        return False
    text = out.read_text(errors="replace")
    return f"mu={mu:.6f}" in text and "Total times" in text and bool(rs.parse_roots(out))


def run() -> int:
    mus = load_mu()
    log_path = HERE / "lr_rerun.log"
    log_path.write_text("")

    def log(message: str) -> None:
        with log_path.open("a") as handle:
            handle.write(message + "\n")
        print(message, flush=True)

    pending = [row[0] for row in rs.MOLECULES]
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
        folder = HERE / name
        for junk in list(folder.glob("*.db")) + list(folder.glob("*.movecs")):
            junk.unlink()
        handle = (folder / "tddft.out").open("w")
        proc = subprocess.Popen(
            [rs.NWCHEM, "tddft.nw"],
            cwd=folder,
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
            if finished(HERE / name, mus[name]):
                rs.write_absorption(HERE / name, rs.parse_roots(HERE / name / "tddft.out"))
                log(f"SKIP {name}")
                continue
            start(name)
        if not running:
            break
        time.sleep(5)
        done = [name for name, job in running.items() if job["proc"].poll() is not None]
        for name in done:
            job = running.pop(name)
            job["handle"].close()
            folder = HERE / name
            roots = rs.parse_roots(folder / "tddft.out")
            elapsed = time.time() - job["t0"]
            if job["proc"].returncode == 0 and roots:
                rs.write_absorption(folder, roots)
                log(f"OK {name} {elapsed:.0f}s roots={len(roots)}")
            else:
                failed.append(name)
                log(f"FAIL {name} exit={job['proc'].returncode} roots={len(roots)}")
    if failed:
        log("ALL_DONE with failures: " + " ".join(failed))
        return 1
    import plot_levels
    plot_levels.main()
    log("ALL_DONE")
    return 0


if __name__ == "__main__":
    prepare()
    raise SystemExit(run())
