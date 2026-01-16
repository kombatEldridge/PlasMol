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
        dict: Preparams dictionary with 'settings', 'classical', 'quantum', 'args', and 'simulation_type'.

    Raises:
        RuntimeError: If required parameters are missing or invalid configuration.
    """
    input_path = args.input
    with open(input_path, 'r') as f:
        # Strip any potential comments (JSON doesn't support them natively, but this handles simple -- or # lines)
        content = ''.join(re.sub(r"(#|--|%)(.*)$", '', line) for line in f if not line.strip().startswith(('#', '--', '%')))
        params = json.loads(content)

    # Extract main sections; they are optional except settings
    settings_params = params.get('settings', {})
    classical_params = params.get('classical')
    quantum_params = params.get('quantum')

    # Post-process quantum section if present (e.g., geometry conversion)
    if quantum_params:
        quantum_params = postprocess_quantum(quantum_params)

    # Validate required settings
    if 'dt' not in settings_params:
        raise RuntimeError("No 'dt' value given in settings. This value is required.")
    if 't_end' not in settings_params:
        raise RuntimeError("No 't_end' value given in settings. This value is required.")

    # Determine simulation type
    if classical_params and quantum_params:
        simulation_type = 'PlasMol'
        if 'simulation' not in classical_params:
            raise RuntimeError("No 'simulation' object found in classical. Please specify the 'simulation' parameters in the classical object.")
        if 'molecule' not in classical_params:
            raise RuntimeError("No 'molecule' object found in classical, but quantum present. Please specify the 'molecule' parameters in the classical object.")
    elif quantum_params:
        simulation_type = 'Quantum'
        logger.info("Only quantum parameters given. Running RT-TDDFT simulation only.")
    elif classical_params:
        if 'simulation' not in classical_params:
            raise RuntimeError("No 'simulation' object found in classical. Please specify the 'simulation' parameters in the classical object.")
        simulation_type = 'Classical'
        logger.info("Only classical parameters given. Running classical simulation only.")
    else:
        raise RuntimeError("The minimum required parameters were not given. Please check guidelines for information on minimal requirements.")

    preparams = {}
    preparams["settings"] = settings_params
    if classical_params:
        preparams["classical"] = classical_params
    if quantum_params:
        preparams["quantum"] = quantum_params

    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    preparams["args"] = args

    preparams["simulation_type"] = simulation_type
    return preparams


def postprocess_quantum(quantum_params):
    """
    Post-processes the quantum section, particularly handling geometry conversion to Bohr units
    and preparing molecule_coords string.

    Args:
        quantum_params (dict): The raw quantum dictionary from JSON.

    Returns:
        dict: Processed quantum dictionary with geometry transformed.
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

    if 'rttddft' not in quantum_params:
        return quantum_params

    rttddft = quantum_params['rttddft']
    if 'geometry' not in rttddft or not isinstance(rttddft['geometry'], list):
        raise ValueError("Quantum 'rttddft' must contain a 'geometry' list of atom objects.")

    geometry = {'atoms': [], 'coords': {}}
    raw_coords = {}
    for idx, atom_entry in enumerate(rttddft['geometry'], start=1):
        if not isinstance(atom_entry, dict) or 'atom' not in atom_entry or 'coords' not in atom_entry:
            raise ValueError("Each geometry entry must be an object with 'atom' (str) and 'coords' (array of 3 floats).")
        atom = atom_entry['atom']
        coords = atom_entry['coords']
        if len(coords) != 3:
            raise ValueError(f"Coords for atom {atom} must be an array of 3 numbers.")
        geometry['atoms'].append(atom)
        label = f"{atom}{idx}"
        raw_coords[label] = np.array(coords)

    if 'units' not in rttddft:
        raise ValueError("Quantum 'rttddft' must specify 'units' for geometry (angstrom or bohr).")

    geometry['coords'] = convert_to_bohr(raw_coords, rttddft['units'])
    rttddft['geometry'] = geometry
    rttddft['geometry']['molecule_coords'] = prepareCoordinates(geometry['atoms'], geometry['coords'])

    # Ensure other values are parsed correctly (e.g., if strings that need to be floats/ints)
    # This can be expanded if needed for other fields

    return quantum_params