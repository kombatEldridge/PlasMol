"""params_helpers/has_nanoparticle.py — gate `has_nanoparticle`.
"""

import logging

logger = logging.getLogger("main")


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """

    if not getattr(params, 'has_nanoparticle', False):
        return
    self = params
    # Nanoparticle params
    if self.has_nanoparticle:
        for attr in ['nanoparticle_material', 'nanoparticle_radius', 'nanoparticle_center']:
            if not hasattr(self, attr):
                pretty = attr.removeprefix("nanoparticle_")
                raise ValueError(f"Nanoparticle requires '{pretty}' attribute.")
        for loc in self.nanoparticle_center:
            if not isinstance(loc, (int, float)):
                raise ValueError(f"Invalid nanoparticle center '{loc}'; must be a number.")


def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    return

