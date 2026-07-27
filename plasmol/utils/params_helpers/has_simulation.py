"""params_helpers/has_simulation.py — gate `has_simulation`.
"""
import json
import logging

from plasmol.utils import constants

logger = logging.getLogger("main")


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """

    if not getattr(params, 'has_simulation', False):
        return
    self = params
    # Plasmon simulation params
    if not self.has_simulation:
        raise ValueError("Invalid plasmon parameters. Please include \"simulation\" parameters.")

    if not hasattr(self, 'plasmon_resolution'):
        self.plasmon_resolution = round(self.plasmon_courant / (self.dt / constants.convertTimeMeep2Atomic))
    self.plasmon_pixel_length_um = 1/self.plasmon_resolution
    if self.has_molecule:
        if self.plasmon_tolerance_field_e <= 0:
            raise ValueError("'plasmon_tolerance_field_e' must be a positive value.")
    if hasattr(self, 'plasmon_cell_volume') and hasattr(self, 'plasmon_cell_length'):
        logger.debug("'cell_volume' is specified; overriding 'cell_length'.")
    elif hasattr(self, 'plasmon_cell_length') and self.plasmon_cell_length <= 0:
        raise ValueError("'plasmon_cell_length' must be a positive value.")
    if self.plasmon_pml_thickness <= 0:
        raise ValueError("'plasmon_pml_thickness' must be a positive value.")
    if not hasattr(self, 'plasmon_symmetries'):
        self.plasmon_symmetries = []
    else:
        if len(self.plasmon_symmetries) % 2 != 0:
            raise ValueError(f"Invalid plasmon symmetry '{self.plasmon_symmetries}'; list length must be even (pairs of axis and value)")
        for i, sym in enumerate(self.plasmon_symmetries):
            if i % 2 == 0:
                if not (isinstance(sym, str) and sym.lower() in self.xyz):
                    raise ValueError(f"Invalid plasmon symmetry '{self.plasmon_symmetries}'; even indices must be 'x', 'y', or 'z' (case-insensitive)")
            else:
                if not (isinstance(sym, int) and sym in [1, -1]):
                    raise ValueError(f"Invalid plasmon symmetry '{self.plasmon_symmetries}'; odd indices must be 1 or -1 (integers)")
    if self.plasmon_surrounding_material_index < 1.0:
        raise ValueError("'surrounding_material_index' must be >= 1.0 (vacuum).")
    elif self.plasmon_surrounding_material_index == 1.0:
        logger.debug("For 'surrounding_material_index' value of 1.0 (vacuum) is being used.")
    elif self.plasmon_surrounding_material_index > 5.0:
        logger.warning("'surrounding_material_index' is unusually high; please verify this value.")

def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    if not getattr(params, 'has_simulation', False):
        return
    self = params
    import meep as mp
    if hasattr(self, 'plasmon_cell_volume'):
        self.cell_volume = mp.Vector3(*self.plasmon_cell_volume)
    else:
        self.cell_volume = mp.Vector3(self.plasmon_cell_length, self.plasmon_cell_length, self.plasmon_cell_length)
    if hasattr(self, 'plasmon_symmetries'):
        symmetries_list = []
        self.plasmon_symmetries_text = self.plasmon_symmetries
        dir_map = {'X': mp.X, 'Y': mp.Y, 'Z': mp.Z}
        for i in range(0, len(self.plasmon_symmetries), 2):
            axis = self.plasmon_symmetries[i].upper()
            phase = int(self.plasmon_symmetries[i + 1])
            symmetries_list.append(mp.Mirror(dir_map[axis], phase=phase))
        self.plasmon_symmetries = symmetries_list if symmetries_list else None


def check_spatial_symmetries(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """
    self = params
    """Validate declared MEEP mirror symmetries and suggest compatible ones when omitted."""
    if not self.has_plasmon:
        return

    component_parity = {
        ('x', 'x'): -1, ('x', 'y'): 1, ('x', 'z'): 1,
        ('y', 'x'): 1, ('y', 'y'): -1, ('y', 'z'): 1,
        ('z', 'x'): 1, ('z', 'y'): 1, ('z', 'z'): -1,
    }

    def as_xyz(coord):
        if hasattr(coord, 'x'):
            return [coord.x, coord.y, coord.z]
        return list(coord)

    def axis_index(axis):
        return self.xyz.index(axis.lower())

    tol = max(1e-9, self.plasmon_pixel_length_um / 2)
    source_center = as_xyz(self.plasmon_source_object.sourceCenter) if self.has_plasmon_source else None
    nanoparticle_center = as_xyz(self.nanoparticle.center) if self.has_nanoparticle else None
    molecule_position = as_xyz(self.plasmol_molecule_position) if self.has_molecule_position else None

    def is_on_plane(coord, axis):
        return abs(coord[axis_index(axis)]) <= tol

    def required_phase(axis, component):
        parity = component_parity[(axis.lower(), component.lower())]
        if self.has_plasmon_source:
            amplitude = getattr(self, 'plasmon_source_amplitude', 1)
            if isinstance(amplitude, (int, float)) and amplitude < 0:
                parity *= -1
            elif isinstance(amplitude, complex) and amplitude.real < 0:
                parity *= -1
        return parity

    def format_suggestion(pairs):
        parts = []
        for axis, phase in pairs:
            parts.extend([axis.upper(), phase])
        return json.dumps(parts)

    def on_coordinate_axis(coord):
        """True if position is at the origin or on exactly one Cartesian axis line."""
        nonzero = sum(1 for c in coord if abs(c) > tol)
        return nonzero <= 1

    def spatial_compatible(axis):
        if source_center is not None and not is_on_plane(source_center, axis):
            return False
        if nanoparticle_center is not None and not is_on_plane(nanoparticle_center, axis):
            return False
        if molecule_position is not None and not is_on_plane(molecule_position, axis):
            return False
        return True

    compatible = []
    if self.has_plasmon_source:
        component = self.plasmon_source_component.lower()
        for axis in self.xyz:
            if spatial_compatible(axis):
                compatible.append((axis, required_phase(axis, component)))

    declared = self.preparams.get("plasmon", {}).get("simulation", {}).get("symmetries")
    if declared is None:
        declared = []

    declared_pairs = {}
    for i in range(0, len(declared), 2):
        declared_pairs[declared[i].lower()] = int(declared[i + 1])

    if declared:
        if self.has_molecule_position and getattr(self, 'plasmol_back_propagation', False):
            if molecule_position is not None and not on_coordinate_axis(molecule_position):
                x, y, z = molecule_position
                raise ValueError(
                    "Mirror symmetries cannot be used with back_propagation enabled when the molecule "
                    f"position ({x}, {y}, {z}) is not on a coordinate axis (the origin or a line along "
                    "x, y, or z). Back-propagation places induced-dipole sources at the probe position, "
                    "which breaks every mirror plane unless the position has at most one nonzero coordinate. "
                    "Move the molecule onto a coordinate axis, remove 'symmetries', or set "
                    "'back_propagation' to false."
                )
            if not getattr(self, 'plasmol_back_propagation_sym_override', False):
                raise ValueError(
                    "Mirror symmetries cannot be used with back_propagation enabled. "
                    "Back-propagation injects the molecule's induced dipole into the FDTD grid at each "
                    "timestep, and the molecular response to the surrounding E-field may not respect the "
                    "declared mirror symmetries. Remove 'symmetries' from the simulation section, set "
                    "'back_propagation' to false, or set 'back_propagation_sym_override' to true to "
                    "acknowledge this risk and proceed with a warning."
                )

        seen_axes = set()
        for i in range(0, len(declared), 2):
            axis = declared[i].lower()
            phase = int(declared[i + 1])
            axis_label = axis.upper()
            if axis in seen_axes:
                raise ValueError(
                    f"Invalid plasmon symmetry '{declared}'; duplicate axis '{axis_label}'."
                )
            seen_axes.add(axis)

            if self.has_plasmon_source:
                component = self.plasmon_source_component.lower()
                expected = required_phase(axis, component)
                if phase != expected:
                    raise ValueError(
                        f"Declared {axis_label} mirror symmetry has phase={phase:+d} but plasmon source "
                        f"component '{component}' requires phase={expected:+d}. "
                        f"Use {format_suggestion([(axis, expected)])} instead."
                    )
                if self.plasmon_source_type == 'custom':
                    logger.warning(
                        f"Custom plasmon source spatial symmetry under {axis_label} mirror cannot be "
                        "verified statically; ensure the source distribution is compatible."
                    )
                elif not is_on_plane(source_center, axis):
                    coord = source_center[axis_index(axis)]
                    raise ValueError(
                        f"Declared {axis_label} mirror symmetry (phase={phase:+d}) is incompatible with "
                        f"plasmon source: center {axis}={coord} is not on the symmetry plane ({axis}=0)."
                    )

            if nanoparticle_center is not None and not is_on_plane(nanoparticle_center, axis):
                coord = nanoparticle_center[axis_index(axis)]
                raise ValueError(
                    f"Declared {axis_label} mirror symmetry (phase={phase:+d}) is incompatible with "
                    f"nanoparticle: center {axis}={coord} is not on the symmetry plane ({axis}=0)."
                )

            if molecule_position is not None and not is_on_plane(molecule_position, axis):
                coord = molecule_position[axis_index(axis)]
                raise ValueError(
                    f"Declared {axis_label} mirror symmetry (phase={phase:+d}) is incompatible with "
                    f"molecule position: position {axis}={coord} is not on the symmetry plane ({axis}=0)."
                )

        if (self.has_molecule_position
                and getattr(self, 'plasmol_back_propagation', False)
                and getattr(self, 'plasmol_back_propagation_sym_override', False)):
            logger.warning(
                "Back-propagation is enabled with mirror symmetries (back_propagation_sym_override=true); "
                "induced dipole sources at the molecule probe position must remain symmetry-consistent "
                "during the simulation, but the molecular response to the surrounding E-field may not "
                "respect the declared symmetries."
            )

    backprop_blocks_symmetry_suggestions = (
        self.has_molecule_position and getattr(self, 'plasmol_back_propagation', False)
    )
    if compatible and not backprop_blocks_symmetry_suggestions:
        missing = [(axis, phase) for axis, phase in compatible if axis not in declared_pairs]
        if missing:
            if not declared_pairs:
                logger.warning(
                    "No symmetries specified; compatible mirror symmetries detected for this system: "
                    f"{format_suggestion(compatible)}. "
                    "Adding these can significantly reduce simulation time."
                )
            else:
                for axis, phase in missing:
                    logger.warning(
                        f"Detected compatible {axis.upper()} mirror symmetry (phase={phase:+d}) "
                        "not in your symmetries list; consider adding it to reduce simulation time."
                    )
                logger.warning(
                    "Recommended complete symmetries list for this system: "
                    f"{format_suggestion(compatible)}."
                )
    elif not declared_pairs:
        if backprop_blocks_symmetry_suggestions:
            logger.debug(
                "No symmetries specified, but since back_propagation is enabled, "
                "symmetries should not be applicable."
            )
        else:
            logger.warning(
                "No symmetries specified for plasmon simulation; operating without symmetries "
                "will cause longer simulation times."
            )

    if declared_pairs and hasattr(self, 'plasmon_symmetries_text'):
        logger.debug(f"Chosen symmetries {self.plasmon_symmetries_text} have been validated.")
