"""Parse PlasMol JSON input files into a preparams dict."""
import re
import json
import logging

from plasmol.utils.struct import (
    DRIVER_PARAM_KEYS,
    LEGACY_NESTED_DRIVER_BLOCKS,
    SHARED_DRIVER_ADDL_KEYS,
)

logger = logging.getLogger("main")

# Legacy core-hole keys (on settings.driver or additional_parameters) → molecule.core_hole.
_CORE_HOLE_LEGACY_KEYS = {
    'mo_removal_index_dict': 'mo_removal_index_dict',
    'core_hole_mo_occ_filepath': 'mo_occ_filepath',
    'mo_occ_filepath': 'mo_occ_filepath',
    'core_hole_watch_indices': 'watch_indices',
    'watch_indices': 'watch_indices',
    'core_hole_filter_by_amplitude': 'filter_by_amplitude',
    'filter_by_amplitude': 'filter_by_amplitude',
    'core_hole_amplitude_threshold': 'amplitude_threshold',
    'amplitude_threshold': 'amplitude_threshold',
}


def _merge_core_hole_key(section, key, value, source):
    """Copy ``value`` into ``molecule.core_hole[key]``, erroring on a conflicting duplicate."""
    if key in section and section[key] != value:
        raise ValueError(
            f"Conflicting '{key}' in molecule.core_hole and {source}."
        )
    section.setdefault(key, value)


def migrate_core_hole_keys(params):
    """Move sudden-ionization keys onto ``molecule.core_hole``.

    Production SCH/DCH lives under the molecule section so quantum and
    absorption can both ionize. The ``core_hole`` driver only surveys MO
    atom contributions. Older files that put these keys on
    ``settings.driver`` or ``additional_parameters`` are rewritten here.
    """
    settings = params.setdefault('settings', {})
    addl = dict(params.get('additional_parameters') or {})
    raw = settings.get('driver')
    extras = {}
    name = None
    if isinstance(raw, dict):
        raw_name = raw.get('name')
        name = raw_name.strip() if isinstance(raw_name, str) else raw_name
        extras = {k: v for k, v in raw.items() if k != 'name'}
    elif isinstance(raw, str):
        name = raw.strip()

    pending = []
    for legacy, dest in _CORE_HOLE_LEGACY_KEYS.items():
        if legacy in addl:
            pending.append((dest, addl.pop(legacy), f"additional_parameters.{legacy}"))
    drop_survey_flag = 'check_mo_contrib_by_atom' in addl
    if drop_survey_flag:
        addl.pop('check_mo_contrib_by_atom')

    driver_pending = []
    drop_driver_survey_flag = False
    if extras:
        for legacy, dest in _CORE_HOLE_LEGACY_KEYS.items():
            if legacy in extras:
                driver_pending.append(
                    (dest, extras.pop(legacy), f"settings.driver.{legacy}")
                )
        if 'check_mo_contrib_by_atom' in extras:
            extras.pop('check_mo_contrib_by_atom')
            drop_driver_survey_flag = True

    if drop_survey_flag or drop_driver_survey_flag:
        logger.warning(
            "check_mo_contrib_by_atom is ignored; the core_hole driver always "
            "surveys per-atom MO contributions and exits."
        )

    to_move = pending + driver_pending
    if to_move:
        if not isinstance(params.get('molecule'), dict):
            raise ValueError(
                "Core-hole keys require a 'molecule' section "
                "(put sudden SCH/DCH under molecule.core_hole)."
            )
        section = params['molecule'].get('core_hole')
        if section is None:
            section = {}
            params['molecule']['core_hole'] = section
        elif not isinstance(section, dict):
            raise ValueError("molecule.core_hole must be a dictionary.")
        for dest, value, source in to_move:
            if name == 'core_hole':
                logger.warning(
                    f"{source} moved to molecule.core_hole.{dest}. "
                    "The core_hole driver only surveys MO atom contributions; "
                    "use driver 'quantum' or 'absorption' for sudden SCH/DCH."
                )
            else:
                logger.warning(
                    f"{source} is deprecated; put it on molecule.core_hole.{dest}."
                )
            _merge_core_hole_key(section, dest, value, source)

    if isinstance(raw, dict) and name and (driver_pending or drop_driver_survey_flag):
        settings['driver'] = {'name': name, **extras}

    section = (params.get('molecule') or {}).get('core_hole')
    if isinstance(section, dict) and 'check_mo_contrib_by_atom' in section:
        logger.warning(
            "molecule.core_hole.check_mo_contrib_by_atom is ignored; "
            "set settings.driver to 'core_hole' to survey MO atom contributions."
        )
        section = dict(section)
        section.pop('check_mo_contrib_by_atom')
        params['molecule']['core_hole'] = section

    if addl:
        params['additional_parameters'] = addl
    else:
        params.pop('additional_parameters', None)
    params['settings'] = settings
    return params


def _merge_driver_key(extras, key, value, source):
    """Copy ``value`` into ``extras[key]``, erroring on a conflicting duplicate."""
    if key in extras and extras[key] != value:
        raise ValueError(
            f"Conflicting '{key}' in settings.driver and {source}."
        )
    extras.setdefault(key, value)


def normalize_settings_driver(params):
    """Rewrite ``settings.driver`` to a dict ``{name, ...}``.

    Accepted input:
      * omitted / null — inferred later from which top-level sections exist
      * string — driver name, no extra keys
      * dict — must include ``name``; remaining keys are that driver's parameters

    Legacy ``additional_parameters.absorption`` / ``.comparison`` nested blocks
    and flat driver keys under ``additional_parameters`` are merged in (with a
    warning) so older input files still parse.
    """
    settings = params.setdefault('settings', {})
    addl = dict(params.get('additional_parameters') or {})
    raw = settings.get('driver')

    extras = {}
    name = None

    if raw in (None, ''):
        name = None
    elif isinstance(raw, str):
        name = raw.strip()
        if not name:
            raise ValueError(
                "settings.driver must be a non-empty string or a dict with 'name'."
            )
    elif isinstance(raw, dict):
        raw_name = raw.get('name')
        if raw_name in (None, ''):
            raise ValueError(
                "settings.driver dict requires a 'name' field (the driver to run)."
            )
        if not isinstance(raw_name, str):
            raise ValueError("settings.driver 'name' must be a string.")
        name = raw_name.strip()
        extras = {k: v for k, v in raw.items() if k != 'name'}
    else:
        raise ValueError(
            "settings.driver must be a string or a dict with 'name'; "
            f"got {type(raw).__name__}."
        )

    nested_legacy = [
        key for key in LEGACY_NESTED_DRIVER_BLOCKS
        if isinstance(addl.get(key), dict)
    ]
    if name is None and nested_legacy:
        if len(nested_legacy) > 1:
            raise ValueError(
                "Multiple driver blocks in additional_parameters "
                f"({', '.join(nested_legacy)}); set settings.driver to choose one."
            )
        name = nested_legacy[0]

    if name:
        if name in addl and isinstance(addl[name], dict):
            logger.warning(
                f"additional_parameters.{name} is deprecated; put those keys on "
                "settings.driver (a dict with 'name')."
            )
            for key, value in addl[name].items():
                _merge_driver_key(extras, key, value, f"additional_parameters.{name}")
            del addl[name]

        for key in list(addl.keys()):
            if key in DRIVER_PARAM_KEYS.get(name, ()):
                logger.warning(
                    f"additional_parameters.{key} is deprecated for driver "
                    f"'{name}'; put it on settings.driver."
                )
                _merge_driver_key(extras, key, addl[key], f"additional_parameters.{key}")
                del addl[key]

        # Peel plasmon-wide decay keys off the driver dict into additional_parameters.
        for key in list(extras):
            if key in SHARED_DRIVER_ADDL_KEYS:
                if key in addl and addl[key] != extras[key]:
                    raise ValueError(
                        f"Conflicting '{key}' in settings.driver and "
                        "additional_parameters."
                    )
                addl[key] = extras.pop(key)

        unknown = [k for k in extras if k not in DRIVER_PARAM_KEYS.get(name, ())]
        if unknown:
            allowed_keys = sorted(DRIVER_PARAM_KEYS.get(name, ()))
            extra = (
                f" Allowed keys besides 'name': {allowed_keys}."
                if allowed_keys
                else " This driver has no extra keys; use a string or {\"name\": \"%s\"}."
                % name
            )
            raise ValueError(
                f"Unknown key(s) on settings.driver for driver '{name}': "
                f"{sorted(unknown)}.{extra}"
            )

        settings['driver'] = {'name': name, **extras}

    leftover_nested = [
        key for key in LEGACY_NESTED_DRIVER_BLOCKS
        if isinstance(addl.get(key), dict)
    ]
    if leftover_nested and name is not None:
        raise ValueError(
            "additional_parameters still contains driver block(s) "
            f"{leftover_nested} that do not match settings.driver '{name}'."
        )

    other_driver_keys = {}
    for driver, keys in DRIVER_PARAM_KEYS.items():
        if driver == name:
            continue
        for key in keys:
            if key in addl:
                other_driver_keys[key] = driver
    if other_driver_keys:
        bits = ", ".join(
            f"{k} ({drv})" for k, drv in sorted(other_driver_keys.items())
        )
        raise ValueError(
            "additional_parameters contains keys for a different driver: "
            f"{bits}."
        )

    if addl:
        params['additional_parameters'] = addl
    else:
        params.pop('additional_parameters', None)

    params['settings'] = settings
    return params


def parse_input_file(args):
    """
    Load and prepare parameters from the input JSON file and CLI args.

    Strips line comments, determines simulation types from present sections,
    and returns a preparams dictionary for PARAMS population.

    Args:
        args: Command-line arguments containing 'input' (path to JSON file).

    Returns:
        dict: Preparams with 'settings', 'simulation_types', 'args', and optional
        'plasmon', 'molecule', 'files', 'additional_parameters'.

    Raises:
        RuntimeError: If neither molecule nor plasmon sections are present.
    """
    input_path = args.input
    with open(input_path, 'r') as f:
        # Removes comments
        content = ''.join(
            re.sub(r"(#|--|%|//)(.*)$", '', line)
            for line in f
            if not line.strip().startswith(('#', '--', '%', '//'))
        )
        params = json.loads(content)

    migrate_core_hole_keys(params)
    normalize_settings_driver(params)

    # Extract main sections; they are optional except settings
    settings_params = params.get('settings', {})
    plasmon_params = params.get('plasmon')
    molecule_params = params.get('molecule')
    files_params = params.get('files')
    addl_params = params.get('additional_parameters')

    # ---- Determine simulation type + validation ----
    simulation_types = []
    if molecule_params:
        simulation_types.append('molecule')
    if plasmon_params:
        simulation_types.append('plasmon')

    if not simulation_types:
        raise RuntimeError(
            "The minimum required parameters were not given. "
            "Please check guidelines for information on minimal requirements."
        )

    # Logging for single-simulation cases (same behaviour as before)
    if len(simulation_types) == 1:
        if simulation_types[0] == 'molecule':
            logger.info("Only 'molecule' parameters given. Running RT-TDDFT simulation only.")
        else:
            logger.info("Only 'plasmon' parameters given. Running MEEP simulation only.")

    # ---- Build preparams ----
    cli_args = {k: v for k, v in vars(args).items() if v is not None}

    preparams = {
        "settings": settings_params,
        "simulation_types": simulation_types,
        "args": cli_args,
    }
    if plasmon_params:
        preparams["plasmon"] = plasmon_params
    if molecule_params:
        preparams["molecule"] = molecule_params
    if files_params:
        preparams["files"] = files_params
    if addl_params:
        preparams["additional_parameters"] = addl_params

    return preparams
