"""params_helpers/has_plasmon.py — gate `has_plasmon`.
"""

import logging

logger = logging.getLogger("main")


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """

    if not getattr(params, 'has_plasmon', False):
        return
    self = params
    if not self.has_simulation:
        raise ValueError("Invalid plasmon parameters. Please include \"simulation\" parameters.")

    if self.has_plasmon_source and getattr(self, 'decay_threshold', 0) <= 0:
        raise ValueError("additional_parameters 'decay_threshold' must be a positive value.")
    if self.has_plasmon_source and getattr(self, 'decay_stop', False):
        msap = getattr(self, 'plasmon_source_additional_parameters', {}) or {}
        if 'frequency' not in msap and 'wavelength' not in msap:
            raise ValueError(
                "additional_parameters 'decay_stop' requires 'frequency' or 'wavelength' "
                "in plasmon source additional_parameters."
            )
        logger.info(
            "Decay stop enabled; simulation will end when fields decay to "
            f"{self.decay_threshold} of their peak (or at t_end)."
        )

    if not self.has_plasmon_source:
        logger.info('No source chosen for simulation. Continuing without it.')


def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    return

