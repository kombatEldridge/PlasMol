"""params_helpers/has_fourier.py — gate `has_fourier`.
"""
import os
import logging

logger = logging.getLogger("main")


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """

    if not getattr(params, 'has_fourier', False):
        return
    self = params
    # Fourier params
    if self.has_fourier:
        if not hasattr(self, 'fourier_reference_only'):
            self.fourier_reference_only = False
        if self.fourier_reference_only:
            if not self.has_plasmon:
                raise ValueError(
                    "Fourier 'reference_only' requires a plasmon section "
                    "(vacuum Meep reference runs need cell/source/molecule position)."
                )
            logger.info(
                "Fourier reference_only=True: will run vacuum E_inc simulations only "
                "(no production plasmol runs, no absorption spectrum)."
            )
        if self.has_plasmon:
            if not self.has_nanoparticle and not self.fourier_reference_only:
                logger.warning("Fourier runs with plasmon settings was not given a nanoparticle.")
        if not self.fourier_reference_only:
            if not hasattr(self, 'fourier_spectrum_filepath') or getattr(self, 'fourier_spectrum_filepath') in ['']:
                raise ValueError("Fourier driver requires 'spectrum_filepath' in additional_parameters.fourier or files.spectra_e_vs_p_filepath.")
        if self.fourier_min_ev < 0:
            raise ValueError("Fourier 'min_ev' must be a non-negative value.")
        if self.fourier_max_ev < 0:
            raise ValueError("Fourier 'max_ev' must be a non-negative value.")
        if self.fourier_max_ev <= self.fourier_min_ev:
            raise ValueError("Fourier 'max_ev' must be greater than 'min_ev'.")
        if self.fourier_gamma < 0:
            raise ValueError("Fourier 'gamma' must be a non-negative value.")
        if hasattr(self, 'fourier_tau'):
            if self.fourier_tau < 0:
                raise ValueError("Fourier 'tau' must be a positive value.")
            elif self.fourier_tau == 0:
                logger.info("Tau modifier = 0 selected; no damping will be applied to time-domain signals.")
            else:
                logger.info(f"Tau modifier = {self.fourier_tau} selected; preparing to apply damping to time-domain signals. See documentation for details.")
        else:
            self.fourier_tau = 0

        # Polarization mode: full (x+y+z) | parallel | perpendicular
        pol = getattr(self, 'fourier_polarization', None)
        if pol in [None, '']:
            self.fourier_polarization = 'full'
        else:
            if not isinstance(pol, str):
                raise ValueError("Fourier 'polarization' must be a string.")
            self.fourier_polarization = pol.lower().strip()
        if self.fourier_polarization not in ('full', 'parallel', 'perpendicular'):
            raise ValueError(
                "Fourier 'polarization' must be one of 'full', 'parallel', or "
                f"'perpendicular'; got '{self.fourier_polarization}'."
            )
        if self.fourier_polarization in ('parallel', 'perpendicular'):
            if not self.has_plasmon:
                raise ValueError(
                    f"Fourier polarization='{self.fourier_polarization}' requires a "
                    "plasmon section (NP–molecule axis is defined in the Meep cell)."
                )
            if not self.has_molecule_position:
                raise ValueError(
                    f"Fourier polarization='{self.fourier_polarization}' requires "
                    "plasmon.molecule.position (to define the NP–molecule axis)."
                )
            if self.fourier_reference_only:
                raise ValueError(
                    "Fourier 'reference_only' is not compatible with polarization "
                    "'parallel' or 'perpendicular' (use polarization='full' for "
                    "three-direction vacuum references, or run a single-pol spectrum)."
                )
        perp = getattr(self, 'fourier_perp_component', None)
        if perp not in [None, '']:
            if not isinstance(perp, str) or perp.lower().strip() not in self.xyz:
                raise ValueError(
                    "Fourier 'perp_component' must be 'x', 'y', or 'z' when set."
                )
            self.fourier_perp_component = perp.lower().strip()
        else:
            self.fourier_perp_component = None

        if self.has_plasmon:
            ref_fp = getattr(self, 'fourier_field_e_ref_filepath', None)
            if ref_fp in [None, '']:
                self.fourier_field_e_ref_filepath = 'field_e_ref.csv'
            elif not isinstance(ref_fp, str):
                raise ValueError("Fourier 'field_e_ref_filepath' must be a string path.")
            if self.fourier_reference_only:
                # Always recompute vacuum E_inc when only building the reference.
                if os.path.isfile(self.fourier_field_e_ref_filepath):
                    logger.warning(
                        f"Fourier reference_only=True: existing '{self.fourier_field_e_ref_filepath}' "
                        f"will be overwritten."
                    )
                self.fourier_use_existing_e_field_ref = False
                logger.info(
                    f"Fourier vacuum reference E_inc (time,xx,yy,zz) will be written to "
                    f"'{self.fourier_field_e_ref_filepath}'."
                )
            elif os.path.isfile(self.fourier_field_e_ref_filepath):
                logger.info(
                    f"Fourier reference file '{self.fourier_field_e_ref_filepath}' found; "
                    f"vacuum reference Meep runs will be skipped."
                )
                self.fourier_use_existing_e_field_ref = True
            else:
                logger.info(
                    f"Fourier vacuum reference E_inc (time,xx,yy,zz) will be written to "
                    f"'{self.fourier_field_e_ref_filepath}' after reference runs."
                )
                self.fourier_use_existing_e_field_ref = False
        else:
            self.fourier_use_existing_e_field_ref = False


def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    return

