"""params_helpers/has_cap.py — gate `has_cap`.
"""

import logging

logger = logging.getLogger("main")


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """

    if not getattr(params, 'has_cap', False):
        return
    self = params
    # Lopata CAP params
    if self.has_cap:
        logger.debug("CAP modifier selected; applying Lopata CAP to spectra.")
        if self.cap_type.lower() not in ['static', 'dynamic']:
            raise ValueError("CAP 'type' must be 'static' or 'dynamic'.")
        if self.cap_gam0 <= 0:
            raise ValueError("CAP 'gam0' must be a positive value.")
        if self.cap_xi < 0:
            raise ValueError("CAP 'xi' must be a non-negative value.")
        if type(self.cap_eps0) == float and self.cap_eps0 < 0:
                raise ValueError("CAP 'eps0' must be a non-negative value.") 
        if self.cap_clamp <= 0:
            raise ValueError("CAP 'clamp' must be a positive value.")


def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    return

