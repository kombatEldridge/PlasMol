"""params_helpers/has_scatter_response_fxn.py — gate `has_scatter_response_fxn`.
"""

import logging

logger = logging.getLogger("main")


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """

    if not getattr(params, 'has_scatter_response_fxn', False):
        return
    self = params
    # No extra validation beyond param_defs typing for probe_points.
    pass


def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    return

