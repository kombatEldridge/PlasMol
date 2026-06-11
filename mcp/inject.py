#!/usr/bin/env python3
"""
Inject at most one bug into a clean PlasMol source tree.
Usage: randomizer.py <src_root> <seed>
   - src_root(str): Path to the root of the PlasMol source tree
   - seed(int): Seed for random number generation
Writes the ground truth to <src_root>/../injected.json (kept OUTSIDE agent view).
"""

import sys, json, random, os

BUGS = {
    "bug_intensity_au": [
        "plasmol/quantum/sources.py",
        [
            (
                "active_component[mask] = self.intensity_au",
                "active_component[mask] = 1e5 * self.intensity_au",
            ),
        ],
    ],
    "bug_dt_meep": [
        "plasmol/classical/simulation.py",
        [
            (
                "self.dt_meep = self.dt / constants.convertTimeMeep2Atomic",
                "self.dt_meep = self.dt * constants.convertTimeMeep2Atomic",
            ),
        ],
    ],
    "bug_t_end_meep": [
        "plasmol/classical/simulation.py",
        [
            (
                "self.t_end_meep = self.t_end / constants.convertTimeMeep2Atomic",
                "self.t_end_meep = self.t_end * constants.convertTimeMeep2Atomic",
            ),
        ],
    ],
    "bug_fourier_damp": [
        "plasmol/utils/input/params.py",
        [
            (
                "self.fourier_damp = True",
                "self.fourier_damp = placeholder",
            ),
            (
                "self.fourier_damp = False",
                "self.fourier_damp = True",
            ),
            (
                "self.fourier_damp = placeholder",
                "self.fourier_damp = False",
            ),

        ],
    ],
    "bug_gamma": [
        "plasmol/quantum/molecule.py",
        [
            (
                "gamma[i] = min(clamp, gam0 * (np.exp(xi * e_tilde) - 1.0))",
                "gamma[i] = min(clamp, gam0 * np.exp(xi * e_tilde))",
            ),
        ],
    ],
    "bug_absorption": [
        "plasmol/drivers/custom_drivers/fourier.py",
        [
            (
                "return - 4 * np.pi * freqs / 3 / constants.C_AU * fullsum",
                "return - np.pi * freqs / 3 / constants.C_AU * fullsum",
            ),
        ],
    ],
    "bug_mf_omega": [
        "plasmol/quantum/molecule.py",
        [
            (
                "self.mf.omega = self.molecule_lrc_parameter",
                "self.mf.omega = 0",
            ),
        ],
    ],
}

BUG_CATEGORIES = list(BUGS.keys()) + ["no_fault"]

def apply_edit(root, rel, old, new):
    path = os.path.join(root, rel)
    with open(path) as f:
        text = f.read()
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"FATAL: expected exactly 1 match for edit in {rel}, found {n}")
    with open(path, "w") as f:
        f.write(text.replace(old, new))

def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: inject.py <src_root> <seed>")
    root = sys.argv[1]
    seed = int(sys.argv[2])
    rng = random.Random(seed)
    choice = rng.choice(BUG_CATEGORIES)

    truth = {"category": choice, "edits": []}
    if choice != "no_fault":
        path, edits = BUGS[choice]
        for old, new in edits:
            apply_edit(root, path, old, new)
            truth["edits"].append({"file": path, "old": old, "new": new})
    
    out = os.path.join(os.path.dirname(root.rstrip("/")), "injected.json")
    with open(out, "w") as f:
        json.dump(truth, f, indent=2)

    print(f"seed={seed} -> injected: {choice}")

if __name__ == "__main__":
    main()