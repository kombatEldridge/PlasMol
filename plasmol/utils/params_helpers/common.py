"""Shared utilities for params_helpers (not a has_* section gate)."""
import math
import logging
from pathlib import Path

import numpy as np
from pyscf.dft import libxc

logger = logging.getLogger("main")


def driver_sets_plasmon_source_component(params):
    """True when the selected driver overwrites ``plasmon.source.component`` per run.

    Hybrid absorption ``full`` / ``parallel`` / ``perpendicular`` pick E themselves.
    ``polarization: single`` keeps the JSON source as given, so component is required.
    ``scatter_response_fxn`` rebuilds X- and Y-polarized sources itself.
    """
    name = getattr(params, 'driver_str', None)
    if name == 'absorption' and getattr(params, 'has_plasmon', False):
        pol = getattr(params, 'absorption_polarization', 'full') or 'full'
        if str(pol).lower().strip() == 'single':
            return False
        return True
    if name == 'scatter_response_fxn':
        return True
    return False


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
        raw = func_name
        # PySCF compound mixes (e.g. "0.20*HF + 0.80*PBE, PBE") are not Libxc names.
        if "*" in raw or ("," in raw and "HF" in raw.upper()):
            logger.debug(f"Using PySCF compound xc '{raw}' without Libxc RSH check.")
            return
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


def _builtin_media(materials):
    """Names on ``meep.materials`` that are already a ``Medium``."""
    import meep as mp
    return {
        name: value
        for name, value in vars(materials).items()
        if not name.startswith("_") and isinstance(value, mp.Medium)
    }


def _rollback_material_module(materials, media_snapshot, names_before):
    """Undo a custom-material file that failed the name check."""
    for name in list(vars(materials)):
        if name.startswith("_") or name in names_before:
            continue
        delattr(materials, name)
    for name, old in media_snapshot.items():
        setattr(materials, name, old)


def _resolve_material_file(material_file, input_path):
    path = Path(material_file)
    if not path.is_absolute():
        base = Path(input_path).resolve().parent if input_path else Path.cwd()
        path = (base / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Custom material file not found: {path}")
    return path


def _exec_material_file(path):
    import importlib.util
    module_name = "plasmol_user_material_" + str(abs(hash(str(path))))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load custom material file '{path}'.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def load_meep_material(material_str, material_file=None, input_path=None):
    """Return a Meep ``Medium``.

    With no ``material_file``, ``material_str`` is a name in ``meep.materials``.
    With a file, that file is executed first and must assign
    ``meep.materials.<material_str>``. The name has to be new: it cannot be a
    built-in ``Medium``, and the file cannot replace any built-in ``Medium``.
    """
    import importlib
    import meep as mp

    materials = importlib.import_module("meep.materials")
    if not material_file:
        try:
            medium = getattr(materials, material_str)
        except AttributeError as e:
            raise ImportError(
                f"Material '{material_str}' not found in meep.materials. "
                f"Check spelling/case or available materials."
            ) from e
        if not isinstance(medium, mp.Medium):
            raise TypeError(
                f"meep.materials.{material_str} is not a Medium."
            )
        return medium

    reserved = _builtin_media(materials)
    if material_str in reserved:
        known = ", ".join(sorted(reserved))
        raise ValueError(
            f"Custom material '{material_str}' already exists in meep.materials. "
            f"Use a name that is not one of: {known}."
        )

    path = _resolve_material_file(material_file, input_path)
    names_before = set(vars(materials))
    try:
        _exec_material_file(path)
    except Exception:
        _rollback_material_module(materials, reserved, names_before)
        raise

    replaced = [
        name for name, old in reserved.items()
        if getattr(materials, name, None) is not old
    ]
    if replaced:
        _rollback_material_module(materials, reserved, names_before)
        raise ValueError(
            "Custom material file "
            f"'{path}' replaces built-in meep.materials "
            f"{', '.join(sorted(replaced))}. "
            "Choose a name that is not already a Meep material."
        )

    try:
        medium = getattr(materials, material_str)
    except AttributeError as e:
        _rollback_material_module(materials, reserved, names_before)
        raise ImportError(
            f"Custom material file '{path}' did not set meep.materials.{material_str}."
        ) from e
    if not isinstance(medium, mp.Medium):
        _rollback_material_module(materials, reserved, names_before)
        raise TypeError(
            f"meep.materials.{material_str} from '{path}' is not a Medium."
        )
    logger.info(f"Loaded custom material '{material_str}' from {path}")
    return medium


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


