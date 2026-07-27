"""params_helpers/has_molecule_source.py — gate `has_molecule_source`.
"""

import logging

logger = logging.getLogger("main")


# Defaults applied only for Fourier kick sources when the user omits fields.
# (Fourier overwrites polarization per direction; component is a placeholder.)
_FOURIER_KICK_DEFAULTS = {
    'molecule_source_intensity': 0.001,
    'molecule_source_peak_time': 0.0,
    'molecule_source_width_steps': 1,
    'molecule_source_component': 'z',
}


def _is_fourier_run(params):
    return (
        getattr(params, 'has_fourier', False)
        or getattr(params, 'driver_str', None) == 'fourier'
    )


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """

    if not getattr(params, 'has_molecule_source', False):
        return
    self = params
    # Molecule Source params
    if self.has_plasmon_source and self.has_molecule_source:
        raise ValueError("Source found in both plasmon and molecule sections. Please specify only one.")
    elif self.has_molecule_source:
        is_fourier = _is_fourier_run(self)

        # Fourier runs only need type: "kick"; fill omitted kick fields with defaults.
        if is_fourier:
            src_type = getattr(self, 'molecule_source_type', None)
            if src_type is None:
                self.molecule_source_type = 'kick'
            elif str(src_type).lower().strip() != 'kick':
                logger.warning(
                    f"Non-'kick' source type '{src_type}' being ignored because "
                    f"Fourier driver/workflow is enabled."
                )
                self.molecule_source_type = 'kick'
            for attr, default in _FOURIER_KICK_DEFAULTS.items():
                if not hasattr(self, attr) or getattr(self, attr) is None:
                    setattr(self, attr, default)
                    pretty = attr.removeprefix("molecule_source_")
                    logger.debug(
                        f"Fourier kick source: defaulting '{pretty}' to {default!r}."
                    )
            # Component is chosen per direction by the Fourier driver.
            logger.debug(
                "Fourier workflow ignores molecule source 'component'; "
                "polarization is set per directional run."
            )
        else:
            for attr in [
                'molecule_source_intensity',
                'molecule_source_peak_time',
                'molecule_source_width_steps',
            ]:
                if not hasattr(self, attr):
                    pretty = attr.removeprefix("molecule_source_")
                    raise ValueError(f"Molecule source requires '{pretty}' attribute.")
            if not hasattr(self, 'molecule_source_component'):
                raise ValueError("Molecule source requires 'component' attribute.")
            if self.molecule_source_component not in self.xyz:
                raise ValueError(f"Molecule source component must be one of {self.xyz}.")

        if hasattr(self, 'molecule_source_peak_time'):
            value = getattr(self, 'molecule_source_peak_time')
            if value is not None and value < 0:
                raise ValueError("Molecule source peak_time must be a positive value.")
        for attr in ['molecule_source_intensity', 'molecule_source_width_steps']:
            if hasattr(self, attr):
                value = getattr(self, attr)
                if value is not None and value <= 0:
                    pretty = attr.removeprefix("molecule_source_")
                    raise ValueError(f"Molecule source '{pretty}' must be a positive value.")
        if not hasattr(self, "molecule_source_additional_parameters") and self.molecule_source_type == 'pulse':
            raise ValueError(
                "Molecule source of type 'pulse' requires 'additional_parameters' "
                "with 'wavelength' or 'frequency'."
            )
        if self.molecule_source_type.lower() == 'pulse':
            msap = getattr(self, 'molecule_source_additional_parameters', {}) or {}
            if 'wavelength' not in msap and 'frequency' not in msap:
                raise ValueError(
                    "Molecule source of type 'pulse' requires 'wavelength' or 'frequency' attribute."
                )
        if self.molecule_source_type.lower() not in ['pulse', 'kick', 'custom_shape']:
            raise ValueError(
                f"Molecule source must be of type 'pulse', 'kick', or 'custom_shape' "
                f"and not '{self.molecule_source_type}'."
            )
        if (
            hasattr(self, 'molecule_source_additional_parameters')
            and self.molecule_source_additional_parameters is not None
            and self.molecule_source_type.lower() == 'pulse'
        ):
            for attr in self.molecule_source_additional_parameters:
                value = self.molecule_source_additional_parameters.get(attr)
                if value is not None and isinstance(value, (int, float)) and value <= 0:
                    raise ValueError(f"Molecule source '{attr}' must be a positive value.")
            if (
                'frequency' not in self.molecule_source_additional_parameters
                and 'wavelength' not in self.molecule_source_additional_parameters
            ):
                raise ValueError(
                    "Either 'frequency' or 'wavelength' must be provided in "
                    "'molecule_source_additional_parameters'."
                )


def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    return

