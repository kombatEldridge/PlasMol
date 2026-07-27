"""Synthetic has_settings: dt/t_end validation and Meep timestep conformity."""
import math
import logging

from plasmol.utils import constants

logger = logging.getLogger("main")


def conform_dt_to_meep(params):
    self = params
    """
    Adjust dt (and t_end) so they match Meep's actual timestep.

    Meep chooses dt = courant / resolution, where resolution must be an
    integer. PlasMol targets the user-requested dt by rounding resolution,
    which generally yields a slightly different timestep than requested.
    """
    courant = getattr(self, 'plasmon_courant', 0.5)
    n_steps = round(self.t_end / self.dt)
    dt_meep_requested = self.dt / constants.convertTimeMeep2Atomic
    self.plasmon_resolution = round(courant / dt_meep_requested)
    self.dt_meep = courant / self.plasmon_resolution
    dt_actual = self.dt_meep * constants.convertTimeMeep2Atomic

    original_dt = self.dt
    original_t_end = self.t_end
    if math.isclose(original_dt, dt_actual, rel_tol=0, abs_tol=1e-15):
        return

    self.dt = dt_actual
    self.t_end = n_steps * self.dt
    logger.info(
        f"Adjusted dt from {original_dt} to {self.dt} au to match Meep's actual timestep "
        f"(resolution={self.plasmon_resolution} pixels/μm, courant={courant})."
    )
    if not math.isclose(original_t_end, self.t_end, rel_tol=0, abs_tol=1e-12):
        logger.info(
            f"Adjusted t_end from {original_t_end} to {self.t_end} au to preserve {n_steps} timesteps."
        )


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """
    self = params
    if not hasattr(self, 'dt'):
        raise ValueError("Missing required parameter: 'dt' in settings.")
    if not hasattr(self, 't_end'):
        raise ValueError("Missing required parameter: 't_end' in settings.")
    if self.dt <= 0:
        raise RuntimeError("'dt' must be a positive value.")
    if self.t_end <= 0:
        raise RuntimeError("'t_end' must be a positive value.")
    if self.dt > self.t_end:
        raise RuntimeError("'dt' cannot be larger than 't_end'.")
    if self.has_plasmon:
        conform_dt_to_meep(self)
    n_steps = round(self.t_end / self.dt)
    if abs(n_steps * self.dt - self.t_end) > 1e-9:
        raise RuntimeError("'t_end' must be a multiple of 'dt'")


def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    self = params
    dt_str = f"{self.dt:.10f}".rstrip('0')
    self.time_rounding_decimals = len(dt_str.split('.')[-1]) if '.' in dt_str else 0
