"""params_helpers/has_checkpoint.py — gate `has_checkpoint`.
"""
import math
import logging

logger = logging.getLogger("main")


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """

    if not getattr(params, 'has_checkpoint', False):
        return
    self = params
    # Checkpointing params
    if getattr(self, 'has_checkpoint', False):
        if self.has_plasmon:
            logger.warning(f"Checkpointing disabled because plasmon section is present in the simulation input (checkpointing is only supported for pure quantum simulations).")
            self.has_checkpoint = False
            for k in ('checkpoint_dict', 'checkpoint_filepath', 'checkpoint_frequency_steps', 'checkpoint_frequency_time'):
                if hasattr(self, k):
                    delattr(self, k)
        if self.has_checkpoint:
            logger.info("Checkpointing selected; preparing to save and load checkpoints during simulation.")
            if not hasattr(self, 'checkpoint_filepath') or self.checkpoint_filepath in ['']:
                raise ValueError("Checkpointing requires 'filepath' attribute for checkpoint file.")
            if not hasattr(self, 'checkpoint_frequency_steps') and not hasattr(self, 'checkpoint_frequency_time'):
                raise ValueError("Checkpointing requires 'frequency_steps' or 'frequency_time' attribute for snapshot frequency.")
            if hasattr(self, 'checkpoint_frequency_steps') and hasattr(self, 'checkpoint_frequency_time'):
                raise ValueError("Checkpointing requires either 'frequency_steps' or 'frequency_time' attribute, not both.")
            if hasattr(self, 'checkpoint_frequency_steps') and self.checkpoint_frequency_steps <= 0:
                raise ValueError("Checkpointing 'frequency_steps' must be a positive value.")
            if hasattr(self, 'checkpoint_frequency_time') and self.checkpoint_frequency_time <= 0:
                raise ValueError("Checkpointing 'frequency_time' must be a positive value.")
            if hasattr(self, 'checkpoint_frequency_time') and self.checkpoint_frequency_time > self.t_end:
                logger.warning(f"Checkpointing 'frequency_time' ({self.checkpoint_frequency_time}) is greater than simulation end time ({self.t_end}). Will only save checkpoint at simulation end.")
            if hasattr(self, 'checkpoint_frequency_time'):
                n_steps = round(self.checkpoint_frequency_time / self.dt)
                reconstructed = n_steps * self.dt
                if not math.isclose(reconstructed, self.checkpoint_frequency_time, rel_tol=1e-9, abs_tol=1e-12):
                    remainder = self.checkpoint_frequency_time % self.dt
                    raise ValueError(
                        f"Checkpointing 'frequency_time' ({self.checkpoint_frequency_time}) must be a multiple of "
                        f"the time step ({self.dt}), but got remainder = {remainder}"
                    )
                self.checkpoint_frequency_steps = n_steps
            if not self.has_molecule:
                raise ValueError("Checkpointing is only supported with molecule simulations.")


def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    return

