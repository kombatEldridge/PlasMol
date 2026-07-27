"""params_helpers/has_images.py — gate `has_images`.
"""

import logging

logger = logging.getLogger("main")


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """

    if not getattr(params, 'has_images', False):
        return
    self = params
    # Images params
    if self.has_images:
        if hasattr(self, 'images_additional_parameters'):
            for loc in self.images_additional_parameters:
                if not isinstance(loc, str):
                    raise ValueError(f"Invalid image additional parameter '{loc}'; must be a string.")
        if not hasattr(self, 'images_timesteps_between'):
            raise ValueError("Images requires 'timesteps_between' attribute.")
    else:
        logger.debug('No picture output chosen for simulation. Continuing without it.')


def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    return

