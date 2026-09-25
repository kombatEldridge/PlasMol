#!/usr/bin/env python3
"""Valence absorption screen for the Step 1 double-core-hole molecules.

Starting coordinates are the NCI Chemical Identifier Resolver 3D models
(CACTUS). PubChem was returning HTTP 500 on 2026-09-23. Each molecule is
then optimized with NWChem PBE0/6-311G* (Cartesian), the same geometry
level as transthioindigo, and the singlet absorption below about 10 eV is
taken from NWChem linear-response TDDFT with LC-ωPBE, μ = 0.34272.

O2 is the triplet ground state. Every other molecule is a closed-shell singlet.
"""

from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
NWCHEM = "/opt/homebrew/bin/nwchem"
CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{}/sdf?get3d=true"
HA_TO_EV = 27.211386245988

# folder, label, SMILES, multiplicity, requested singlet roots
MOLECULES = [
    ("n2", "N2", "N#N", 1, 12),
    ("co", "CO", "[C-]#[O+]", 1, 12),
    ("co2", "CO2", "O=C=O", 1, 16),
    ("n2o", "N2O", "N#[N+][O-]", 1, 16),
    ("o2", "O2", "O=O", 3, 16),
    ("h2o", "H2O", "O", 1, 12),
    ("nh3", "NH3", "N", 1, 12),
    ("ch4", "CH4", "C", 1, 12),
    ("h2co", "H2CO", "C=O", 1, 16),
    ("c2h2", "C2H2", "C#C", 1, 16),
    ("c2h4", "C2H4", "C=C", 1, 16),
    ("c2h6", "C2H6", "CC", 1, 16),
    ("c6h6", "C6H6", "c1ccccc1", 1, 24),
    ("para_aminophenol", "para-aminophenol", "Nc1ccc(O)cc1", 1, 30),
    ("ortho_aminophenol", "ortho-aminophenol", "Nc1ccccc1O", 1, 30),
    ("meta_aminophenol", "meta-aminophenol", "Nc1cccc(O)c1", 1, 30),
    ("h2s", "H2S", "S", 1, 12),
    ("so2", "SO2", "O=S=O", 1, 16),
    ("cs2", "CS2", "S=C=S", 1, 16),
    ("lif", "LiF", "[Li]F", 1, 12),
    ("beo", "BeO", "O=[Be]", 1, 12),
    ("bf", "BF", "B#F", 1, 12),
    ("pyrimidine", "pyrimidine", "c1cncnc1", 1, 30),
    ("purine", "purine", "c1ncc2[nH]cnc2n1", 1, 30),
    ("uracil", "uracil", "O=c1cc[nH]c(=O)[nH]1", 1, 30),
    ("cytosine", "cytosine", "Nc1cc[nH]c(=O)n1", 1, 30),
    ("thymine", "thymine", "Cc1c[nH]c(=O)[nH]c1=O", 1, 30),
    ("adenine", "adenine", "Nc1ncnc2[nH]cnc12", 1, 30),
    ("guanine", "guanine", "Nc1nc2[nH]cnc2c(=O)[nH]1", 1, 30),
    ("formamide", "formamide", "NC=O", 1, 16),
    ("pentan3one", "3-pentanone", "CCC(=O)CC", 1, 30),
    ("pentan2one", "2-pentanone", "CCCC(C)=O", 1, 30),
    ("pentanal", "pentanal", "CCCCC=O", 1, 30),
]


def cactus_sdf(smiles: str) -> str:
    url = CACTUS.format(urllib.parse.quote(smiles))
    request = urllib.request.Request(url, headers={"User-Agent": "PlasMol/1.0"})
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 — retry network and HTTP errors
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"CACTUS failed for {smiles}: {last}")


def sdf_atoms(sdf: str) -> list[tuple[str, tuple[float, float, float]]]:
    lines = sdf.splitlines()
    for index, line in enumerate(lines):
        if "V2000" not in line:
            continue
        count = int(line[:3])
        atoms = []
        for raw in lines[index + 1:index + 1 + count]:
            x, y, z, symbol = raw.split()[:4]
            atoms.append((symbol, (float(x), float(y), float(z))))
        if len(atoms) != count:
            raise ValueError("SDF atom block is short")
        return atoms
    raise ValueError("No V2000 atom block")


def xyz_text(atoms, comment: str) -> str:
    rows = [str(len(atoms)), comment]
    for symbol, (x, y, z) in atoms:
        rows.append(f"{symbol:<2} {x:14.8f} {y:14.8f} {z:14.8f}")
    return "\n".join(rows) + "\n"


def geometry_block(atoms) -> str:
    lines = ["geometry units angstroms noautosym noautoz"]
    for symbol, (x, y, z) in atoms:
        lines.append(f"  {symbol:<2} {x:14.8f} {y:14.8f} {z:14.8f}")
    lines.append("end")
    return "\n".join(lines)


def write_tddft(folder: Path, label: str, mult: int, nroots: int, atoms, mu: float) -> None:
    geom = geometry_block(atoms)
    spin_line = "  notriplet\n" if mult == 1 else ""
    tddft = f"""echo
start {folder.name}_td
title "{label} LR-TDDFT, LC-wPBE mu={mu:.6f}, 6-311G* Cartesian, PBE0 geometry"
scratch_dir {folder}
permanent_dir {folder}
memory total 2500 mb
charge 0

{geom}

basis cartesian
  * library 6-311G*
end

dft
  xc xwpbe 1.00 cpbe96 1.0 hfexch 1.00
  cam {mu:.6f} cam_alpha 0.00 cam_beta 1.00
  mult {mult}
  grid fine
  iterations 100
  convergence energy 1.0e-7
end

tddft
  nroots {nroots}
{spin_line}end

task tddft energy
"""
    (folder / "tddft.nw").write_text(tddft)


def write_inputs(folder: Path, label: str, mult: int, nroots: int, atoms) -> None:
    geom = geometry_block(atoms)
    opt = f"""echo
start {folder.name}_opt
title "{label} geometry optimization, PBE0/6-311G* Cartesian"
scratch_dir {folder}
permanent_dir {folder}
memory total 2500 mb
charge 0

{geom}

basis cartesian
  * library 6-311G*
end

dft
  xc pbe0
  mult {mult}
  grid fine
  iterations 100
  convergence energy 1.0e-7
end

driver
  maxiter 120
end

task dft optimize
"""
    (folder / "opt.nw").write_text(opt)
    write_tddft(folder, label, mult, nroots, atoms, 0.34272)


def prepare(only: str | None = None) -> None:
    for folder_name, label, smiles, mult, nroots in MOLECULES:
        if only and folder_name != only:
            continue
        folder = HERE / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        url = CACTUS.format(urllib.parse.quote(smiles))
        sdf_path = folder / "cactus.sdf"
        if not sdf_path.exists():
            sdf_path.write_text(cactus_sdf(smiles))
            time.sleep(0.4)
        atoms = sdf_atoms(sdf_path.read_text())
        (folder / "start.xyz").write_text(xyz_text(atoms, f"CACTUS 3D  {url}"))
        (folder / "source.txt").write_text(
            f"label: {label}\n"
            f"smiles: {smiles}\n"
            f"multiplicity: {mult}\n"
            f"nroots: {nroots}\n"
            f"url: {url}\n"
            "fetched: 2026-09-23\n"
            "service: NCI Chemical Identifier Resolver (CACTUS), record 3d\n"
            "PubChem PUG returned HTTP 500 on this date and was not used.\n"
        )
        write_inputs(folder, label, mult, nroots, atoms)
        print(f"{folder_name:20} {len(atoms):3} atoms  mult {mult}  roots {nroots}")


def last_geometry(opt_out: Path):
    text = opt_out.read_text(errors="replace")
    if "Optimization converged" not in text:
        raise RuntimeError(f"optimization did not converge: {opt_out}")
    marker = "Output coordinates in angstroms"
    start = text.rfind(marker)
    if start < 0:
        raise RuntimeError(f"no coordinate block in {opt_out}")
    atoms = []
    for line in text[start:].splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 6 and parts[0].isdigit() and parts[1][:1].isalpha():
            atoms.append((parts[1], (float(parts[3]), float(parts[4]), float(parts[5]))))
            continue
        if atoms:
            break
    if not atoms:
        raise RuntimeError(f"empty coordinate block in {opt_out}")
    return atoms


def parse_roots(td_out: Path):
    """Singlet (or open-shell) roots with dipole oscillator strengths."""
    text = td_out.read_text(errors="replace")
    roots = []
    header = re.compile(
        r"Root\s+(\d+)\s+(?:(singlet|triplet)\s+)?(\S+)\s+([-\d.]+)\s+a\.u\.\s+([-\d.]+)\s+eV",
        re.I,
    )
    strength = re.compile(r"Dipole Oscillator Strength\s+([-\d.E]+)", re.I)
    matches = list(header.finditer(text))
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else match.end() + 800
        window = text[match.end():stop]
        found = strength.search(window)
        roots.append({
            "root": int(match.group(1)),
            "spin": match.group(2) or "open-shell",
            "au": float(match.group(4)),
            "eV": float(match.group(5)),
            "f": float(found.group(1)) if found else float("nan"),
        })
    return roots


def refresh_tddft_geometry(folder: Path, atoms) -> None:
    text = (folder / "tddft.nw").read_text()
    updated = re.sub(
        r"geometry units angstroms noautosym noautoz\n.*?\nend\n",
        geometry_block(atoms) + "\n",
        text,
        count=1,
        flags=re.S,
    )
    (folder / "tddft.nw").write_text(updated)


def write_absorption(folder: Path, roots) -> None:
    rows = ["root\tspin\tenergy_eV\twavelength_nm\toscillator_strength\tbelow_10eV"]
    for root in roots:
        energy = root["eV"]
        wavelength = 1239.84193 / energy if energy > 0.0 else float("nan")
        below = "yes" if energy <= 10.0 else "no"
        rows.append(
            f"{root['root']}\t{root['spin']}\t{energy:.6f}\t{wavelength:.4f}\t{root['f']:.8e}\t{below}"
        )
    (folder / "absorption.txt").write_text("\n".join(rows) + "\n")


def nwchem_ok(path: Path, token: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    text = path.read_text(errors="replace")
    if token not in text or "Total times" not in text:
        return False
    tail = text[-2500:].lower()
    return "there is an error in the input file" not in tail and "mpi_abort" not in tail


def screen(names: list[str] | None = None, slots: int = 3) -> int:
    import os
    import subprocess
    selected = [row for row in MOLECULES if names is None or row[0] in names]
    log_path = HERE / "screen.log"
    log_path.write_text("")

    def log(message: str) -> None:
        with log_path.open("a") as handle:
            handle.write(message + "\n")
        print(message, flush=True)

    pending = [row[0] for row in selected]
    running: dict[str, dict] = {}
    failed = []

    def start_one(folder_name: str, stage: str) -> None:
        folder = HERE / folder_name
        deck = "opt.nw" if stage == "opt" else "tddft.nw"
        log_name = "opt.out" if stage == "opt" else "tddft.out"
        for junk in list(folder.glob("*.db")) + list(folder.glob("*.movecs")):
            junk.unlink()
        env = os.environ.copy()
        env.update({
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "VECLIB_MAXIMUM_THREADS": "2",
        })
        handle = (folder / log_name).open("w")
        proc = subprocess.Popen(
            [NWCHEM, deck],
            cwd=folder,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        running[folder_name] = {"proc": proc, "handle": handle, "stage": stage, "t0": time.time()}
        log(f"START {folder_name} {stage} pid={proc.pid}")

    while pending or running:
        while pending and len(running) < slots:
            folder_name = pending.pop(0)
            folder = HERE / folder_name
            if nwchem_ok(folder / "opt.out", "Optimization converged"):
                if not (folder / "opt.xyz").exists():
                    atoms = last_geometry(folder / "opt.out")
                    (folder / "opt.xyz").write_text(xyz_text(atoms, "PBE0/6-311G* optimized"))
                    refresh_tddft_geometry(folder, atoms)
                if nwchem_ok(folder / "tddft.out", "Root") and (folder / "absorption.txt").exists():
                    log(f"SKIP {folder_name} already finished")
                    continue
                atoms = last_geometry(folder / "opt.out")
                refresh_tddft_geometry(folder, atoms)
                start_one(folder_name, "td")
            else:
                start_one(folder_name, "opt")
        if not running:
            break
        time.sleep(5)
        done = []
        for folder_name, job in running.items():
            if job["proc"].poll() is None:
                continue
            done.append(folder_name)
        for folder_name in done:
            job = running.pop(folder_name)
            job["handle"].close()
            folder = HERE / folder_name
            stage = job["stage"]
            code = job["proc"].returncode
            elapsed = time.time() - job["t0"]
            if stage == "opt":
                if code == 0 and nwchem_ok(folder / "opt.out", "Optimization converged"):
                    atoms = last_geometry(folder / "opt.out")
                    (folder / "opt.xyz").write_text(xyz_text(atoms, "PBE0/6-311G* optimized"))
                    refresh_tddft_geometry(folder, atoms)
                    log(f"OK {folder_name} opt {elapsed:.0f}s")
                    pending.append(folder_name)
                else:
                    failed.append(folder_name)
                    log(f"FAIL {folder_name} opt exit={code}")
            else:
                roots = parse_roots(folder / "tddft.out") if (folder / "tddft.out").exists() else []
                if code == 0 and roots and "Total times" in (folder / "tddft.out").read_text(errors="replace"):
                    write_absorption(folder, roots)
                    bright = [root for root in roots if root["eV"] <= 10.0 and root["f"] >= 0.01]
                    top = bright[0]["eV"] if bright else (roots[0]["eV"] if roots else float("nan"))
                    log(f"OK {folder_name} td {elapsed:.0f}s roots={len(roots)} first_bright_or_first={top:.3f}")
                else:
                    failed.append(folder_name)
                    log(f"FAIL {folder_name} td exit={code} roots={len(roots)}")
    if failed:
        log("ALL_DONE with failures: " + " ".join(failed))
        return 1
    log("ALL_DONE")
    return 0


if __name__ == "__main__":
    import sys
    command = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    if command == "prepare":
        prepare(sys.argv[2] if len(sys.argv) > 2 else None)
    elif command == "run":
        only = sys.argv[2:] or None
        raise SystemExit(screen(only))
    else:
        raise SystemExit(f"usage: {sys.argv[0]} prepare [name] | run [names...]")
