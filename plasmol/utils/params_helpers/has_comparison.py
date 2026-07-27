"""params_helpers/has_comparison.py — gate `has_comparison`.
"""
from plasmol.utils.params_helpers.common import check_xc
import logging

logger = logging.getLogger("main")


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """

    if not getattr(params, 'has_comparison', False):
        return
    self = params
    # Comparison mode params
    if self.has_comparison:
        logger.info("Comparison modifier selected; preparing to run additional simulations for comparison to molecule results.")
        if self.has_plasmon:
            raise ValueError("Comparison mode is not supported with plasmon simulations. Please run with only molecule simulations.")
        if self.has_fourier:
            raise ValueError("Comparison mode is not supported with fourier simulations. Please run with only molecule simulations.")
        if not hasattr(self, 'comparison_bases') or not hasattr(self, 'comparison_xcs'):
            raise ValueError("Comparison mode requires both 'bases' and 'xcs' lists. See documentation for details.")
        for loc in self.comparison_bases:
            if not isinstance(loc, str):
                raise ValueError(f"Invalid comparison basis '{loc}'; must be a string.")
        for loc in self.comparison_xcs:
            if not isinstance(loc, str):
                raise ValueError(f"Invalid comparison xcs '{loc}'; must be a string.")
        # TODO: Implement LRC to comparison
        if hasattr(self, 'comparison_lrc_parameters'):
            if not isinstance(self.comparison_lrc_parameters, dict):
                raise ValueError("Comparison 'lrc_parameters' must be a dictionary.")
            for loc in self.comparison_lrc_parameters:
                if not isinstance(self.comparison_lrc_parameters[loc], (int, float)):
                    raise ValueError(f"Invalid comparison 'lrc_parameters' for '{loc}'; must be a number.")
                if loc not in self.comparison_xcs:
                    raise ValueError(f"Comparison 'lrc_parameters' for '{loc}' is not in the list of xcs.")
        for xc in self.comparison_xcs:
            omega = self.comparison_lrc_parameters.get(xc, None)
            check_xc(self, xc, omega)
        for loc in ['comparison_num_virtual', 'comparison_num_occupied', 'comparison_index_min', 'comparison_index_max']:
            if hasattr(self, loc):
                if getattr(self, loc) < 1: 
                    pretty = loc.removeprefix("comparison_")
                    raise ValueError(f"Comparison '{pretty}' must be at least 1.")


def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    return

