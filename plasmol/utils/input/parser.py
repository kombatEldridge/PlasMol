# utils/input/parser.py
import json
import logging
import numpy as np
import re

logger = logging.getLogger("main")

def parseInputFile(args):
    """
    Prepares parameters from the input JSON file and CLI args.

    Loads the JSON file, performs necessary post-processing (e.g., for quantum geometry),
    and determines the simulation type based on present sections.

    Args:
        args: Command-line arguments containing 'input' (path to JSON file).

    Returns:
        dict: Preparams dictionary with 'settings', 'plasmon', 'molecule', 'args', and 'simulation_type'.

    Raises:
        RuntimeError: If required parameters are missing or invalid configuration.
    """
    input_path = args.input
    with open(input_path, 'r') as f:
        # Removes comments
        content = ''.join(re.sub(r"(#|--|%|//)(.*)$", '', line) for line in f if not line.strip().startswith(('#', '--', '%', '//')))
        params = json.loads(content)

    # Extract main sections; they are optional except settings
    settings_params = params.get('settings', {})
    plasmon_params = params.get('plasmon')
    molecule_params = params.get('molecule')

    # Validate required settings
    if 'dt' not in settings_params:
        raise RuntimeError("No 'dt' value given in settings. This value is required.")
    if 't_end' not in settings_params:
        raise RuntimeError("No 't_end' value given in settings. This value is required.")
    if settings_params['dt'] <= 0:
        raise RuntimeError("'dt' must be a positive value.")
    if settings_params['t_end'] <= 0:
        raise RuntimeError("'t_end' must be a positive value.")
    if settings_params['dt'] > settings_params['t_end']:
        raise RuntimeError("'dt' cannot be larger than 't_end'.")
    
    # Post-process quantum section if present (e.g., geometry conversion)
    if molecule_params:
        molecule_params = postprocess_molecule(molecule_params)

    # Determine simulation type
    if plasmon_params and molecule_params:
        simulation_type = 'PlasMol'
        if 'simulation_parameters' not in plasmon_params:
            raise RuntimeError("No 'simulation_parameters' object found in 'plasmon' section. Please specify the 'simulation_parameters' in the 'plasmon' section.")
        if 'molecule_position' not in plasmon_params:
            raise RuntimeError("No 'molecule_position' object found in 'plasmon' section, but quantum present. Please specify the 'molecule_position' parameters in the 'plasmon' section.")
    elif molecule_params:
        simulation_type = 'Molecule'
        logger.info("Only 'molecule' parameters given. Running RT-TDDFT simulation only.")
    elif plasmon_params:
        simulation_type = 'Plasmon'
        if 'simulation_parameters' not in plasmon_params:
            raise RuntimeError("No 'simulation_parameters' object found in 'plasmon' section. Please specify the 'simulation_parameters' in the 'plasmon' section.")
        logger.info("Only 'plasmon' parameters given. Running MEEP simulation only.")
    else:
        raise RuntimeError("The minimum required parameters were not given. Please check guidelines for information on minimal requirements.")

    preparams = {}
    preparams["settings"] = settings_params
    if plasmon_params:
        preparams["plasmon"] = plasmon_params
    if molecule_params:
        preparams["molecule"] = molecule_params

    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    preparams["args"] = args

    preparams["simulation_type"] = simulation_type
    return preparams


def postprocess_molecule(molecule_params):
    """
    Post-processes the molecule section, particularly handling geometry conversion to Bohr units
    and preparing molecule_coords string.

    Args:
        molecule_params (dict): The raw molecule dictionary from JSON.

    Returns:
        dict: Processed molecule dictionary with geometry transformed.
    """
    def prepareCoordinates(atoms, coords):
        molecule_coords = ""
        for index, atom in enumerate(atoms):
            molecule_coords += f" {atom} {coords[atom + str(index + 1)][0]} {coords[atom + str(index + 1)][1]} {coords[atom + str(index + 1)][2]}"
            if index != (len(atoms) - 1):
                molecule_coords += ";"
        return molecule_coords.strip()

    def convert_to_bohr(coords, units):
        if units.lower().startswith("angstrom"):
            factor = 1.8897259886
        elif units.lower().startswith("bohr"):
            factor = 1.0
        else:
            raise ValueError(f"Unknown units '{units}', must be 'angstrom' or 'bohr'")
        return {label: np.array(xyz) * factor for label, xyz in coords.items()}

    if 'geometry' not in molecule_params or not isinstance(molecule_params['geometry'], list):
        raise ValueError("The 'molecule' section must contain a 'geometry' list of atoms and coords.")

    geometry = {'atoms': [], 'coords': {}}
    raw_coords = {}
    for idx, atom_entry in enumerate(molecule_params['geometry'], start=1):
        if not isinstance(atom_entry, dict) or 'atom' not in atom_entry or 'coord' not in atom_entry:
            raise ValueError("Each geometry entry must be an object with 'atom' (str) and 'coord' (array of 3 floats).")
        atom = atom_entry['atom']
        coords = atom_entry['coord']
        if len(coords) != 3:
            raise ValueError(f"Coords for atom {atom} must be an array of 3 numbers.")
        geometry['atoms'].append(atom)
        label = f"{atom}{idx}"
        raw_coords[label] = np.array(coords)

    if 'geometry_units' not in molecule_params:
        raise ValueError("The 'molecule' section must specify 'geometry_units' for geometry (angstrom or bohr).")

    geometry['coords'] = convert_to_bohr(raw_coords, molecule_params['geometry_units'])
    molecule_params['geometry'] = geometry
    molecule_params['geometry']['coords'] = prepareCoordinates(geometry['atoms'], geometry['coords'])

    return molecule_params