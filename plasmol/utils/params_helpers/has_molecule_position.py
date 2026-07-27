"""params_helpers/has_molecule_position.py — gate `has_molecule_position`.
"""
import numpy as np
import logging

logger = logging.getLogger("main")


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """

    if not getattr(params, 'has_molecule_position', False):
        return
    self = params
    # Molecule position param
    if self.has_molecule_position:
        if not self.has_molecule:
            raise ValueError(f"Molecule position properly specified in 'plasmon' section but without 'molecule' section present in input file.")
        for loc in self.plasmol_molecule_position:
            if not isinstance(loc, (int, float)):
                raise ValueError(f"Invalid molecule position '{loc}'; must be a number.")
        if len(self.plasmol_molecule_position) != 3:
            raise ValueError("Molecule position must be an array of three numbers [x, y, z].")
        if self.has_nanoparticle:
            p = np.asarray(self.plasmol_molecule_position, dtype=float)
            c = np.asarray(self.nanoparticle_center, dtype=float)
            if p.shape != (3,) or c.shape != (3,):
                raise ValueError("Point and center must be 3D coordinates (x, y, z)")
            d = np.linalg.norm(p - c)
            distance = abs(d - self.nanoparticle_radius)
            min_surface_distance = self.plasmon_pixel_length_um
            grid_tol = max(1e-9, min_surface_distance * 1e-3)
            if distance + grid_tol < min_surface_distance:
                raise ValueError(
                    f"Molecule position is too close to nanoparticle surface "
                    f"(dist = {distance:.9f} μm). Minimum distance required: "
                    f"{min_surface_distance:.9f} μm."
                )
            if self.nanoparticle_radius > d + grid_tol:
                raise ValueError(
                    f"Molecule position is inside the nanoparticle "
                    f"(dist from NP center = {d:.9f} μm). Minimum distance required: "
                    f"{self.nanoparticle_radius + min_surface_distance:.9f} μm."
                )


def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    return

