"""params_helpers/has_plasmon_source.py — gate `has_plasmon_source`.
"""
from plasmol.classical.sources import walk_through_src_funcs
import logging

logger = logging.getLogger("main")


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """

    if not getattr(params, 'has_plasmon_source', False):
        return
    self = params
    # Plasmon source params
    if self.has_plasmon_source:
        for attr in ['plasmon_source_type', 'plasmon_source_center', 'plasmon_source_size', 'plasmon_source_component']:
            if not hasattr(self, attr):
                pretty = attr.removeprefix("plasmon_source_")
                raise ValueError(f"Source requires '{pretty}' attribute.")
        for loc in self.plasmon_source_center:
            if not isinstance(loc, (int, float)):
                raise ValueError(f"Invalid plasmon source center '{loc}'; must be a number.")
        for loc in self.plasmon_source_size:
            if not isinstance(loc, (int, float)):
                raise ValueError(f"Invalid plasmon source size '{loc}'; must be a number.")
        if self.plasmon_source_component not in self.xyz:
            raise ValueError(f"Invalid plasmon source component '{self.plasmon_source_component}'; must be 'x', 'y', or 'z'.")
        if getattr(self, "plasmon_source_additional_parameters", None) is not None:
            if 'frequency' not in self.plasmon_source_additional_parameters and 'wavelength' not in self.plasmon_source_additional_parameters and (self.plasmon_source_type == 'continuous' or self.plasmon_source_type == 'gaussian'):
                raise ValueError(f"Either 'frequency' or 'wavelength' must be provided in 'plasmon_source_additional_parameters'.")
        elif getattr(self, "plasmon_source_frequency", None) is None and getattr(self, "plasmon_source_wavelength", None) is None and (self.plasmon_source_type == 'continuous' or self.plasmon_source_type == 'gaussian'):
            raise ValueError(f"Either 'frequency' or 'wavelength' must be provided for {self.plasmon_source_type} source.")
        if self.plasmon_source_type == 'custom':
            if not hasattr(self, 'plasmon_source_additional_parameters') or 'src_func' not in self.plasmon_source_additional_parameters:
                raise ValueError(f"Custom source requires 'src_func' in 'plasmon_source_additional_parameters' attribute.")
            try:
                walk_through_src_funcs(self.plasmon_source_additional_parameters['src_func'])
            except ValueError as e:
                raise ValueError(f"Error occurred while processing custom source function: {e}")
    else:
        logger.info('No source chosen for simulation. Continuing without it.')


def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    return

