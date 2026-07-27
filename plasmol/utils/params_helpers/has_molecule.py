"""params_helpers/has_molecule.py — gate `has_molecule`.
"""
from plasmol.utils.params_helpers.common import check_xc, resolve_geometry_path
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

