"""params_helpers/has_files.py — gate `has_files`.
"""

import logging

logger = logging.getLogger("main")


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """

    self = params
    # Files
    for file in ['field_e_filepath', 'field_p_filepath']:
        if hasattr(self, file):
            value = getattr(self, file)
            if not isinstance(value, str) or value in ['']:
                raise ValueError(f"Filepath for '{file}' must be a non-empty string.")


def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    return

