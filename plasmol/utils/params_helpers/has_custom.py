"""has_custom: driver resolution and get_driver binding."""
import logging

from plasmol.drivers import get_driver

logger = logging.getLogger("main")


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """

    return


def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    self = params
    self.driver_str = getattr(self, 'driver_str', None)
    if self.has_custom:
        if self.driver_str is None:
            raise ValueError(
                "A custom driver was selected but settings.driver has no name. "
                "Use a string or a dict with 'name'."
            )
        logging.debug(f"Custom driver specified: {self.driver_str}")
    elif 'molecule' in self.simulation_types and 'plasmon' in self.simulation_types:
        self.driver_str = 'plasmol'
    elif 'molecule' in self.simulation_types:
        self.driver_str = 'quantum'
    elif 'plasmon' in self.simulation_types:
        self.driver_str = 'classical'

    self.driver = get_driver(self.driver_str)
