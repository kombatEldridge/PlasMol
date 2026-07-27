"""params_helpers/has_np_abs_cross_sec.py — gate `has_np_abs_cross_sec`.
"""

import logging

logger = logging.getLogger("main")


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """

    self = params
    # np_abs_cross_sec driver params
    if getattr(self, 'driver_str', None) == 'np_abs_cross_sec':
        if getattr(self, 'n_flux_freqs', 0) <= 0:
            raise ValueError("np_abs_cross_sec driver requires 'n_flux_freqs' to be a positive integer.")
        if getattr(self, 'flux_padding', -1) < 0:
            raise ValueError("np_abs_cross_sec driver requires 'flux_padding' to be a non-negative value.")


def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    return

