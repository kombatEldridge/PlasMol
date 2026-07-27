"""Parse PlasMol JSON input files into a preparams dict."""
import re
import json
import logging

logger = logging.getLogger("main")


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
