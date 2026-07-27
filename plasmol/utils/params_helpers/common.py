"""Shared utilities for params_helpers (not a has_* section gate)."""
import math
import logging
from pathlib import Path

import numpy as np
from pyscf.dft import libxc

logger = logging.getLogger("main")


def get_nested_value(d, path):
    cur = d
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur


def check_xc(params, func_name: str, omega: float = None):
    try:
        func_name = func_name.upper()
        if "{TUNE}" in func_name:
            func_name = func_name.replace("{TUNE}", "0.4")
        derived_omega, _, _ = libxc.rsh_coeff(func_name)
        if omega == "tune":
            if derived_omega == 0:
                raise ValueError(f"Functional '{func_name}' is not a range-separated hybrid (RSH); cannot tune lrc_parameter.")
            return
        if omega is not None and derived_omega == 0:
            raise ValueError(f"Functional '{func_name}' is not a range-separated hybrid (RSH) so lrc_parameter will be ignored.")
        if omega is not None:
            if not math.isclose(omega, derived_omega, rel_tol=1e-9):
                logger.warning(f"Functional '{func_name}' has a default lrc_parameter of {derived_omega}, but {omega} was provided. Using the given value will override the default.")
        if omega is None and derived_omega > 0:
            logger.debug(f"Functional '{func_name}' is a range-separated hybrid (RSH) with default lrc_parameter = {derived_omega}.")
            params.molecule_lrc_parameter = derived_omega
    except Exception as e:
        raise ValueError(f"Error checking xc functional '{func_name}': {e}")


def load_meep_material(material_str):
    import importlib
    materials = importlib.import_module("meep.materials")
    try:
        return getattr(materials, material_str)
    except AttributeError as e:
        raise ImportError(
            f"Material '{material_str}' not found in meep.materials. "
            f"Check spelling/case or available materials."
        ) from e


def resolve_geometry_path(params, geometry: str) -> Path:
    path = Path(geometry)
    if not path.is_absolute():
        path = (Path(params.input_file_path).resolve().parent / path).resolve()
    return path


def construct_geometry(params, geometry, units):
    """
    Post-process molecule geometry:
    - Accepts either:
        1. List of dicts: [{"atom": "O", "coord": [x,y,z]}, ...]
        2. String path to a .xyz file
    - Validates input
    - Converts to Bohr units
    - Builds the exact coords string expected by the simulator
    """
    atoms = []
    coords_bohr = {}

    if isinstance(geometry, str):
        path = params._resolve_geometry_path(geometry)

        # Parse XYZ file
        with open(path) as f:
            lines = [line.strip() for line in f if line.strip()]

        # First line: total number of atoms (optional)
        # Second line: molecule name or comment (optional)
        # All other lines: element symbol or atomic number, x, y, and z coordinates, separated by spaces, tabs, or commas
        start_line = None
        num_atoms = None
        for current_line, line in enumerate(lines):
            items = line.split()
            if len(items) < 4:
                if len(items) == 1 and items[0].isdigit():
                    num_atoms = int(items[0])
                continue
            else:
                for item in items:
                    item = item.replace('.', '').replace(',', '')
                    if not item.isdigit():
                        continue
                start_line = current_line

        if start_line is None:
            raise ValueError("Invalid XYZ file format: no valid atom lines found.")
        if num_atoms is None:
            num_atoms = 0
            for i in range(start_line, len(lines)):
                items = lines[i].split()
                if len(items) == 4:
                    num_atoms += 1

        geometry = []
        for i in range(2, 2 + num_atoms):
            parts = lines[i].split()
            atom = parts[0]
            coord = [float(x) for x in parts[1:4]]
            geometry.append({"atom": atom, "coord": coord})

        params.geometry_xyz_filepath = path

    if not isinstance(geometry, list):
        raise ValueError("geometry must be a list of dicts or a path to a .xyz file.")

    for idx, entry in enumerate(geometry, start=1):
        if not isinstance(entry, dict) or 'atom' not in entry or 'coord' not in entry:
            raise ValueError("Each geometry entry must be a dict with 'atom' (str) and 'coord' (list of 3 floats).")

        atom = entry['atom']
        coord = entry['coord']

        if len(coord) != 3:
            raise ValueError(f"Coords for atom {atom} must have exactly 3 numbers.")

        atoms.append(atom)
        label = f"{atom}{idx}"
        coords_bohr[label] = np.array(coord, dtype=float)

    # Convert to Bohr if input was in Ångstroms
    if units.lower().startswith('angstrom'):
        factor = 1.8897259886
        coords_bohr = {label: xyz * factor for label, xyz in coords_bohr.items()}
        units = "bohr"

    # Build the exact string format PySCF wants
    coords_str = ""
    for i, atom in enumerate(atoms):
        x, y, z = coords_bohr[f"{atom}{i+1}"]
        coords_str += f" {atom} {x} {y} {z}"
        if i < len(atoms) - 1:
            coords_str += ";"

    return atoms, coords_str.strip(), units


