"""params_helpers/has_core_hole.py — gate `has_core_hole`.
"""

import logging

logger = logging.getLogger("main")

_CORE_HOLE_SECTION_KEYS = frozenset({
    'mo_removal_index_dict',
    'mo_occ_filepath',
    'watch_indices',
    'filter_by_amplitude',
    'amplitude_threshold',
})


def _coerce_mo_removal_index_dict(mo_dict):
    """Return {int MO index: 1 or 2} from the JSON mapping."""
    if mo_dict is None:
        raise ValueError(
            "Core-hole requires 'mo_removal_index_dict' on molecule.core_hole "
            "(dict mapping 0-based MO index → electrons to remove, e.g. {\"0\": 2})."
        )
    if not isinstance(mo_dict, dict) or len(mo_dict) == 0:
        raise ValueError(
            "Core-hole 'mo_removal_index_dict' must be a non-empty dictionary "
            "mapping MO indices to 1 or 2 electrons removed."
        )

    coerced = {}
    for raw_key, raw_val in mo_dict.items():
        try:
            key = int(raw_key)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Core-hole 'mo_removal_index_dict' key {raw_key!r} must be an integer "
                f"0-based MO index, got {type(raw_key).__name__}."
            ) from e
        if key < 0:
            raise ValueError(
                f"Core-hole 'mo_removal_index_dict' key {key} must be a non-negative 0-based MO index."
            )
        try:
            val = int(raw_val)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Core-hole 'mo_removal_index_dict' value for MO {key} must be an integer "
                f"(electrons to remove), got {raw_val!r}."
            ) from e
        if val not in (1, 2):
            raise ValueError(
                f"Core-hole 'mo_removal_index_dict' value for MO {key} must be 1 or 2, got {val}."
            )
        if key in coerced:
            raise ValueError(
                f"Core-hole 'mo_removal_index_dict' has duplicate MO index {key}."
            )
        coerced[key] = val
    return coerced


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """

    self = params
    is_survey = getattr(self, 'driver_str', None) == 'core_hole'
    if not is_survey and not getattr(self, 'has_core_hole', False):
        return

    section = getattr(self, 'core_hole_dict', None)
    if isinstance(section, dict):
        unknown = sorted(k for k in section if k not in _CORE_HOLE_SECTION_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown key(s) in molecule.core_hole: {unknown}. "
                f"Allowed: {sorted(_CORE_HOLE_SECTION_KEYS)}."
            )

    coerced = _coerce_mo_removal_index_dict(
        getattr(self, 'mo_removal_index_dict', None)
    )
    self.mo_removal_index_dict = coerced

    if is_survey:
        # Survey driver: report atom contributions and exit. Do not ionize.
        if getattr(self, 'core_hole_mo_occ_filepath', None):
            logger.warning(
                "The core_hole driver only surveys MO atom contributions. "
                "molecule.core_hole.mo_occ_filepath is ignored; use driver "
                "'quantum' or 'absorption' for sudden SCH/DCH."
            )
        self.has_core_hole = False
        self.force_open_shell = False
        logger.info(
            f"Core-hole MO contribution survey: will report atom contributions "
            f"for MOs {list(coerced.keys())} and exit before propagation."
        )
        return

    if len(coerced) not in (1, 2):
        raise ValueError(
            "Core-hole 'mo_removal_index_dict' must contain one or two MO indices "
            "(one MO with 2 e⁻ → DCH; two MOs with 1 e⁻ each → two SCH; "
            "one MO with 1 e⁻ → SCH). Use driver 'core_hole' to survey more MOs."
        )
    n_holes = sum(coerced.values())
    if n_holes not in (1, 2):
        raise ValueError(
            f"Core-hole 'mo_removal_index_dict' must remove 1 or 2 electrons total, got {n_holes}."
        )
    if len(coerced) == 2 and any(v != 1 for v in coerced.values()):
        raise ValueError(
            "When two MOs are listed in 'mo_removal_index_dict', each value must be 1 "
            "(one electron removed from each MO)."
        )
    if len(coerced) == 1 and n_holes == 2:
        mode = "double hole on one MO (DCH)"
    elif len(coerced) == 1 and n_holes == 1:
        mode = "single hole on one MO (SCH)"
    else:
        mode = "single hole on each of two MOs"
    # SCH / two-site holes need α/β channels. A singlet DCH ({i: 2})
    # stays restricted, matching NWChem DFT/RT-TDDFT (no odft, mult 1).
    parent_open = getattr(self, 'molecule_spin', 0) != 0
    if mode == "double hole on one MO (DCH)" and not parent_open:
        self.force_open_shell = False
        logger.debug(
            f"Core-hole mode: mo_removal_index_dict={coerced} ({mode}); "
            "closed-shell (RKS) propagation."
        )
    else:
        self.force_open_shell = True
        logger.debug(
            f"Core-hole mode: mo_removal_index_dict={coerced} ({mode}); "
            "forcing open-shell (UKS) calculation."
        )

    watch = getattr(self, 'core_hole_watch_indices', None)
    if watch is not None:
        if not isinstance(watch, list) or len(watch) == 0:
            raise ValueError(
                "Core-hole 'watch_indices' must be a non-empty list of 0-based MO "
                "indices to plot, or omitted to plot all logged MOs."
            )
        coerced_watch = []
        for raw in watch:
            try:
                idx = int(raw)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Core-hole 'watch_indices' entry {raw!r} must be an integer "
                    f"0-based MO index."
                ) from e
            if idx < 0:
                raise ValueError(
                    f"Core-hole 'watch_indices' entry {idx} must be a non-negative "
                    f"0-based MO index."
                )
            coerced_watch.append(idx)
        self.core_hole_watch_indices = coerced_watch
        logger.debug(f"Core-hole plot MO indices (watch_indices): {coerced_watch}")

    if not getattr(self, 'core_hole_mo_occ_filepath', None):
        raise ValueError(
            "molecule.core_hole requires 'mo_occ_filepath' for sudden SCH/DCH "
            "(CSV of time-dependent hole occupations)."
        )


def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    return

