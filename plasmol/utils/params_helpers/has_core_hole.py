"""params_helpers/has_core_hole.py — gate `has_core_hole`.
"""

import logging

logger = logging.getLogger("main")


def check(params):
    """Check that parameters for this section are consistent and free of errors.

    Validate required fields, types/ranges, and cross-parameter constraints so
    that invalid input is rejected before any derived objects are built.
    """

    self = params
    # Core-hole driver params (SCH / DCH modes)
    if getattr(self, 'driver_str', None) == 'core_hole' or getattr(self, 'has_core_hole', False):
        self.force_open_shell = True
        logger.debug("Core-hole driver forcing open-shell calculation.")

        check_contrib = getattr(self, 'check_mo_contrib_by_atom', False)
        mo_dict = getattr(self, 'mo_removal_index_dict', None)
        if mo_dict is None:
            raise ValueError(
                "Core-hole driver requires 'mo_removal_index_dict' under additional_parameters "
                "(dict mapping 0-based MO index → electrons to remove, e.g. {\"0\": 2})."
            )
        if not isinstance(mo_dict, dict) or len(mo_dict) == 0:
            raise ValueError(
                "Core-hole 'mo_removal_index_dict' must be a non-empty dictionary "
                "mapping MO indices to 1 or 2 electrons removed."
            )

        # JSON object keys are strings; coerce to int → int
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
        self.mo_removal_index_dict = coerced

        if check_contrib:
            logger.info(
                f"Core-hole MO contribution survey mode: will report atom contributions "
                f"for MOs {list(coerced.keys())}."
            )
        else:
            if len(coerced) not in (1, 2):
                raise ValueError(
                    "Core-hole 'mo_removal_index_dict' must contain one or two MO indices when "
                    "check_mo_contrib_by_atom is false "
                    "(one MO with 2 e⁻ → DCH; two MOs with 1 e⁻ each → two SCH; "
                    "one MO with 1 e⁻ → SCH)."
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
            logger.debug(f"Core-hole mode: mo_removal_index_dict={coerced} ({mode}).")

        # Optional final-plot MO selection (logging is always 0 .. LUMO+1)
        watch = getattr(self, 'core_hole_watch_indices', None)
        if watch is not None:
            if not isinstance(watch, list) or len(watch) == 0:
                raise ValueError(
                    "Core-hole 'core_hole_watch_indices' must be a non-empty list of 0-based MO "
                    "indices to plot, or omitted to plot all logged MOs."
                )
            coerced_watch = []
            for raw in watch:
                try:
                    idx = int(raw)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Core-hole 'core_hole_watch_indices' entry {raw!r} must be an integer "
                        f"0-based MO index."
                    ) from e
                if idx < 0:
                    raise ValueError(
                        f"Core-hole 'core_hole_watch_indices' entry {idx} must be a non-negative "
                        f"0-based MO index."
                    )
                coerced_watch.append(idx)
            self.core_hole_watch_indices = coerced_watch
            logger.debug(
                f"Core-hole plot MO indices (core_hole_watch_indices): {coerced_watch}"
            )

        if not getattr(self, 'core_hole_mo_occ_filepath', None):
            raise ValueError(
                "Core-hole driver requires 'core_hole_mo_occ_filepath' under additional_parameters."
            )


def form(params):
    """Build derived attributes and objects for this section.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    return

