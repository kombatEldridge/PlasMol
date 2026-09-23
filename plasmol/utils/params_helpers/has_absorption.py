"""params_helpers/has_absorption.py — gate `has_absorption`.
"""
import os
import logging

logger = logging.getLogger("main")


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """

    if not getattr(params, 'has_absorption', False):
        return
    self = params
    # Absorption-driver params
    if self.has_absorption:
        if not hasattr(self, 'absorption_reference_only'):
            self.absorption_reference_only = False
        if self.absorption_reference_only:
            if not self.has_plasmon:
                raise ValueError(
                    "Absorption 'reference_only' requires a plasmon section "
                    "(vacuum Meep reference runs need cell/source/molecule position)."
                )
            logger.info(
                "Absorption reference_only=True: will run vacuum E_inc simulations only "
                "(no production plasmol runs, no absorption spectrum)."
            )
        if self.has_plasmon:
            if not self.has_nanoparticle and not self.absorption_reference_only:
                logger.warning("Absorption runs with plasmon settings was not given a nanoparticle.")
        if not self.absorption_reference_only:
            spec = getattr(self, 'absorption_spectrum_filepath', None)
            if spec in [None, '']:
                files = (getattr(self, 'preparams', None) or {}).get('files') or {}
                fallback = files.get('spectra_e_vs_p_filepath')
                if fallback not in [None, '']:
                    self.absorption_spectrum_filepath = fallback
                    logger.info(
                        "Absorption spectrum_filepath defaulting to "
                        f"files.spectra_e_vs_p_filepath ('{fallback}')."
                    )
                else:
                    raise ValueError(
                        "absorption driver requires 'spectrum_filepath' on settings.driver "
                        "(or files.spectra_e_vs_p_filepath)."
                    )
        if self.absorption_min_ev < 0:
            raise ValueError("Absorption 'min_ev' must be a non-negative value.")
        if self.absorption_max_ev < 0:
            raise ValueError("Absorption 'max_ev' must be a non-negative value.")
        if self.absorption_max_ev <= self.absorption_min_ev:
            raise ValueError("Absorption 'max_ev' must be greater than 'min_ev'.")
        if self.absorption_gamma < 0:
            raise ValueError("Absorption 'gamma' must be a non-negative value.")
        if hasattr(self, 'absorption_tau'):
            if self.absorption_tau < 0:
                raise ValueError("Absorption 'tau' must be a positive value.")
            elif self.absorption_tau == 0:
                logger.info("Tau modifier = 0 selected; no damping will be applied to time-domain signals.")
            else:
                logger.info(f"Tau modifier = {self.absorption_tau} selected; preparing to apply damping to time-domain signals. See documentation for details.")
        else:
            self.absorption_tau = 0

        allowed_obs = ('cross_section', 'dissipative_power', 'A_raw')
        obs = getattr(self, 'absorption_observables', None)
        if obs in [None, '']:
            self.absorption_observables = ['cross_section']
        else:
            if not isinstance(obs, list) or len(obs) == 0:
                raise ValueError(
                    "Absorption 'observables' must be a non-empty list of "
                    f"{list(allowed_obs)}."
                )
            cleaned = []
            seen = set()
            for item in obs:
                if not isinstance(item, str):
                    raise ValueError(
                        "Absorption 'observables' entries must be strings; "
                        f"got {type(item).__name__}."
                    )
                name = item.strip()
                if name not in allowed_obs:
                    raise ValueError(
                        f"Unknown absorption observable '{name}'. "
                        f"Allowed: {list(allowed_obs)}."
                    )
                if name not in seen:
                    cleaned.append(name)
                    seen.add(name)
            self.absorption_observables = cleaned
        logger.info(
            "Absorption observables: "
            + ", ".join(self.absorption_observables)
            + "."
        )

        # Polarization: full (x+y+z) | parallel | perpendicular | single (JSON source)
        pol = getattr(self, 'absorption_polarization', None)
        if pol in [None, '']:
            if getattr(self, 'has_nanoparticle', False):
                raise ValueError(
                    "Absorption with a nanoparticle requires settings.driver.polarization "
                    "of 'parallel', 'perpendicular', or 'single' "
                    "('full' isotropic x+y+z is not allowed when an NP is present)."
                )
            self.absorption_polarization = 'full'
        else:
            if not isinstance(pol, str):
                raise ValueError("Absorption 'polarization' must be a string.")
            self.absorption_polarization = pol.lower().strip()
        if self.absorption_polarization not in ('full', 'parallel', 'perpendicular', 'single'):
            raise ValueError(
                "Absorption 'polarization' must be one of 'full', 'parallel', "
                f"'perpendicular', or 'single'; got '{self.absorption_polarization}'."
            )
        if self.absorption_polarization == 'full' and getattr(self, 'has_nanoparticle', False):
            raise ValueError(
                "Absorption polarization='full' is not allowed with a nanoparticle "
                "(x+y+z mixes inequivalent ∥ and ⊥ channels). "
                "Use 'parallel', 'perpendicular', or 'single'."
            )
        if self.absorption_polarization in ('parallel', 'perpendicular', 'single'):
            if not self.has_plasmon:
                raise ValueError(
                    f"Absorption polarization='{self.absorption_polarization}' requires a "
                    "plasmon section."
                )
            if not self.has_molecule_position:
                need = (
                    "to define the NP–molecule axis"
                    if self.absorption_polarization in ('parallel', 'perpendicular')
                    else "as the vacuum E_inc sample location"
                )
                raise ValueError(
                    f"Absorption polarization='{self.absorption_polarization}' requires "
                    f"plasmon.molecule.position ({need})."
                )
            if self.absorption_reference_only:
                raise ValueError(
                    "Absorption 'reference_only' is not compatible with polarization "
                    f"'{self.absorption_polarization}' (use polarization='full' for "
                    "three-direction vacuum references, or run a single-pol spectrum)."
                )
        if self.absorption_polarization == 'single':
            component = getattr(self, 'plasmon_source_component', None)
            if component in (None, ''):
                raise ValueError(
                    "Absorption polarization='single' uses the JSON source as given; "
                    "plasmon.source.component is required ('x', 'y', or 'z')."
                )
        perp = getattr(self, 'absorption_perp_component', None)
        if perp not in [None, '']:
            if not isinstance(perp, str) or perp.lower().strip() not in self.xyz:
                raise ValueError(
                    "Absorption 'perp_component' must be 'x', 'y', or 'z' when set."
                )
            self.absorption_perp_component = perp.lower().strip()
        else:
            self.absorption_perp_component = None

        if self.has_plasmon:
            ref_fp = getattr(self, 'absorption_field_e_ref_filepath', None)
            if ref_fp in [None, '']:
                self.absorption_field_e_ref_filepath = 'field_e_ref.csv'
            elif not isinstance(ref_fp, str):
                raise ValueError("Absorption 'field_e_ref_filepath' must be a string path.")
            if self.absorption_reference_only:
                # Always recompute vacuum E_inc when only building the reference.
                if os.path.isfile(self.absorption_field_e_ref_filepath):
                    logger.warning(
                        f"Absorption reference_only=True: existing '{self.absorption_field_e_ref_filepath}' "
                        f"will be overwritten."
                    )
                self.absorption_use_existing_e_field_ref = False
                logger.info(
                    f"Absorption vacuum reference E_inc (time,xx,yy,zz) will be written to "
                    f"'{self.absorption_field_e_ref_filepath}'."
                )
            elif os.path.isfile(self.absorption_field_e_ref_filepath):
                logger.info(
                    f"Absorption reference file '{self.absorption_field_e_ref_filepath}' found; "
                    f"vacuum reference Meep runs will be skipped."
                )
                self.absorption_use_existing_e_field_ref = True
            else:
                logger.info(
                    f"Absorption vacuum reference E_inc (time,xx,yy,zz) will be written to "
                    f"'{self.absorption_field_e_ref_filepath}' after reference runs."
                )
                self.absorption_use_existing_e_field_ref = False
        else:
            self.absorption_use_existing_e_field_ref = False


def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    return

