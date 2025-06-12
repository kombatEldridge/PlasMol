# input_parser.py
import constants
import numpy as np

def read_input(inputfile, options):
    with open(inputfile) as f:
        in_mol = False
        atoms, coords = [], {}
        for line in f:
            parts = line.split()
            if not parts: continue

            # geometry block
            if parts[0] == "start" and parts[1] == "molecule":
                in_mol = True
                continue
            if parts[0] == "end" and parts[1] == "molecule":
                in_mol = False
                continue
            if in_mol:
                symbol, x, y, z = parts
                atoms.append(symbol)
                coords[f"{symbol}{len(atoms)}"] = np.array([float(x),float(y),float(z)])
                continue

            # simple key = value lines
            key = parts[0].lower()
            val = parts[1]
            if key == "charge":
                options.charge = int(val)
            elif key == "spin":
                options.spin = int(val)
            elif key == "basis":
                options.basis = val
            elif key == "diis":
                options.diis = (val.lower() == "true")
            elif key in ("e_convergence","e_conv"):
                options.e_conv = float(val)
            elif key in ("d_convergence","d_conv"):
                options.d_conv = float(val)
            elif key == "maxiter":
                options.maxiter = int(val)
            elif key == "nroots":
                options.nroots = int(val)
            elif key == "xc":
                options.xc = val
            elif key == "resplimit":
                options.resplimit = float(val)
            elif key == "guess_mos":
                options.guess_mos = val
            elif key == "propagator":
                options.propagator = val.lower()
            elif key == "method":
                options.method = val.lower()
            elif key == "units":
                options.units = val

    options.molecule["atoms"] = atoms
    options.molecule["coords"] = convert_to_bohr(coords, options.units)
    return options.molecule, options


def convert_to_bohr(coords, units):
    """
    coords: dict mapping labels (e.g. "O1") → numpy.array([x,y,z])
    units: "angstrom" or "bohr"
    Returns a new dict with all coordinates in Bohr.
    """
    if units.lower().startswith("angstrom"):
        factor = constants.angs2bohr
    elif units.lower().startswith("bohr"):
        factor = 1.0
    else:
        raise ValueError(f"Unknown units '{units}', must be 'angstrom' or 'bohr'")
    return { label: xyz * factor for label, xyz in coords.items() }