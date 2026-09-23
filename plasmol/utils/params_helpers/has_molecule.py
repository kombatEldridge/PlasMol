"""params_helpers/has_molecule.py — gate `has_molecule`.
"""
from plasmol.quantum.geometry import normalize_rotation
from plasmol.utils.params_helpers.common import check_xc, get_nested_value, resolve_geometry_path
import logging

logger = logging.getLogger("main")


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """

    if not getattr(params, 'has_molecule', False):
        return
    self = params
    if self.has_plasmon:
        if not hasattr(self, 'plasmol_molecule_position'):
            raise RuntimeError("No 'plasmol_molecule_position' object found in 'plasmon' section, but quantum (molecule) is present. Please specify the 'plasmol_molecule_position' parameters in the 'plasmon' section.")
    else:
        if hasattr(self, 'driver_str') and self.driver_str == "plasmol":
            raise ValueError("Driver 'plasmol' requires a 'plasmon' section.")
    if self.has_comparison:
        if hasattr(self, 'molecule_basis') or hasattr(self, 'molecule_xc'):
            logger.info("Comparison modifier selected; ignoring basis set and xc. Using values given in additional_parameters.")
        else:
            # Setting singular basis and xc so it can pass the following checks. These will be ignored anyway.
            self.molecule_basis = '6-31g'
            self.molecule_xc = 'pbe0'
    for attr in ['molecule_geometry', 'molecule_geometry_units', 'molecule_basis', 'molecule_charge', 'molecule_spin']:
        if not hasattr(self, attr) or getattr(self, attr) == []:
            pretty = attr.removeprefix("molecule_")
            raise ValueError(f"Molecule requires '{pretty}' attribute.")
    lrc_parameters = [self.molecule_lrc_parameter] if hasattr(self, 'molecule_lrc_parameter') else []
    check_xc(self, self.molecule_xc, *lrc_parameters)
    if type(self.molecule_geometry) == str:
        path = resolve_geometry_path(self, self.molecule_geometry)
        if not path.exists():
            raise ValueError(f"Geometry file not found: {path}")
        if not path.suffix.lower() == '.xyz':
            raise ValueError("String input must be a path to a .xyz file.")
        self.molecule_geometry = str(path)
    else:
        for loc in self.molecule_geometry:
            if not isinstance(loc, dict):
                raise ValueError(f"Invalid molecule position '{loc}'; must be a dictionary (ex. {'atom': 'O', 'coord': [0.0, 0.0, -0.1302052882]}).")
    if hasattr(self, 'molecule_propagator_str'):
        self.molecule_propagator_str = self.molecule_propagator_str.lower()
        if self.molecule_propagator_str not in ['step', 'rk4', 'magnus2']:
            raise ValueError(f"Unsupported propagator: {self.molecule_propagator_str}. Acceptable: step, rk4, magnus2.")
    if not self.molecule_geometry_units in ['angstrom', 'bohr']:
        raise ValueError(f"Invalid 'molecule_geometry_units': '{self.molecule_geometry_units}'. Must be 'angstrom' or 'bohr'.")
    _BASIS_COORDS = {
        'spherical': False, 'sph': False,
        'cartesian': True, 'cart': True,
    }
    raw_coords = get_nested_value(getattr(self, 'preparams', {}) or {}, ['molecule', 'basis_coords'])
    raw_cart = get_nested_value(getattr(self, 'preparams', {}) or {}, ['molecule', 'cartesian'])
    if raw_coords not in (None, ''):
        if not isinstance(raw_coords, str):
            raise ValueError("Molecule 'basis_coords' must be a string ('spherical' or 'cartesian').")
        key = raw_coords.lower().strip()
        if key not in _BASIS_COORDS:
            raise ValueError(
                "Molecule 'basis_coords' must be 'spherical' or 'cartesian' "
                f"(aliases: sph, cart); got {raw_coords!r}."
            )
        want_cart = _BASIS_COORDS[key]
        if raw_cart is not None and bool(raw_cart) != want_cart:
            raise ValueError(
                "Molecule 'basis_coords' and 'cartesian' disagree "
                f"({raw_coords!r} vs cartesian={raw_cart!r}). Set only one."
            )
        self.molecule_cartesian = want_cart
        self.molecule_basis_coords = 'cartesian' if want_cart else 'spherical'
    else:
        self.molecule_cartesian = bool(getattr(self, 'molecule_cartesian', True))
        self.molecule_basis_coords = 'cartesian' if self.molecule_cartesian else 'spherical'
    if self.molecule_cartesian:
        logger.debug("Molecular basis uses Cartesian Gaussians (6 d functions; PlasMol default).")
    else:
        logger.info("Molecular basis uses spherical Gaussians (5 d functions; PySCF default).")
    grid_level = getattr(self, 'molecule_grid_level', None)
    if grid_level is not None:
        if not isinstance(grid_level, int) or isinstance(grid_level, bool) or grid_level < 0 or grid_level > 9:
            raise ValueError(
                f"Molecule 'grid_level' must be an integer 0–9, got {grid_level!r}."
            )
    rotation = getattr(self, 'molecule_rotation', None)
    if rotation:
        normalize_rotation(rotation)


    # Tuning ("tune" / {TUNE}) validation: must use driver="tune"
    tune_requested = False
    if hasattr(self, 'molecule_lrc_parameter') and self.molecule_lrc_parameter == "tune":
        tune_requested = True
    if hasattr(self, 'cap_eps0') and self.cap_eps0 == "tune":
        tune_requested = True
    xc_val = getattr(self, 'molecule_xc', None)
    if isinstance(xc_val, str) and "{TUNE}" in xc_val.upper():
        tune_requested = True

    if tune_requested:
        # Determine effective driver (mirrors logic in _attribute_formation but available here)
        eff_driver = getattr(self, 'driver_str', None)
        if eff_driver is None:
            if 'molecule' in self.simulation_types and 'plasmon' in self.simulation_types:
                eff_driver = 'plasmol'
            elif 'molecule' in self.simulation_types:
                eff_driver = 'quantum'
            elif 'plasmon' in self.simulation_types:
                eff_driver = 'classical'
        if eff_driver != 'tune':
            raise ValueError(
                "Use of 'tune' for lrc_parameter, cap 'eps0', or {TUNE} placeholder in xc is only allowed when "
                "using the dedicated tuning driver. Set \"driver\": \"tune\" under \"settings\"."
            )


def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    return

