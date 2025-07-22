# input/parser.py
import re
import logging
import numpy as np

logger = logging.getLogger("main")

def inputFilePrepare(args):
    input = args.input
    classical, quantum, settings = parseSections(input)
    settings_params = parseclassicalSection(settings)
    classical_params = parseclassicalSection(classical) if classical is not None else None
    quantum_params = parseQuantumSection(quantum) if quantum is not None else None

    if not 'dt' in settings_params:
        raise RuntimeError("No 'dt' value given in settings file. This value is required.")
    if not 't_end' in settings_params:
        raise RuntimeError("No 't_end' value given in settings file. This value is required.")

    if classical_params is not None and quantum_params is not None:
        simulation_type = 'PlasMol'
        if not 'simulation' in classical_params:
            raise RuntimeError("No 'simulation' block found in classical block. Please specify the 'simulation' parameters in the classical block.")
        if not 'molecule' in classical_params:
            raise RuntimeError("No 'molecule' block found in classical block, but Quantum block found. Please specify the 'molecule' parameters in the classical block.")
    elif quantum_params is not None:
        simulation_type = 'Quantum'
        logger.info("Only RT-TDDFT input file given. Running RT-TDDFT simulation only.")
    elif classical_params is not None:
        if not 'simulation' in classical_params:
            raise RuntimeError("No 'simulation' block found in classical block. Please specify the 'simulation' parameters in the classical block.")
        simulation_type = 'Classical'
        logger.info("Only classical input file given. Running classical simulation only.")
    else:
        raise RuntimeError("The minimum required parameters were not given. Please check guidelines for information on minimal requirements.")
    
    preparams = {}
    preparams["settings"] = settings_params
    if classical_params is not None:
        preparams["classical"] = classical_params 
    if quantum_params is not None:
        preparams["quantum"] = quantum_params

    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    preparams["args"] = args
    
    preparams["simulation_type"] = simulation_type
    return preparams


def parseSections(input_file):
    comment_pattern = re.compile(r"(#|--|%)(.*)$")

    in_classical_section = False
    in_quantum_section = False
    in_settings_section = False
    classical_section = None
    quantum_section = None
    settings_section = ""

    with open(input_file) as f:
        for raw_line in f:
            line = comment_pattern.sub('', raw_line).strip()
            if not line:
                continue

            parts = line.split()

            if parts[0] == 'start' and parts[1] == 'classical' and len(parts) == 2:
                in_classical_section = True
                classical_section = ""
                continue
            elif parts[0] == 'end' and parts[1] == 'classical' and len(parts) == 2:
                in_classical_section = False
                continue
            elif parts[0] == 'start' and parts[1] == 'quantum' and len(parts) == 2:
                in_quantum_section = True
                quantum_section = ""
                continue
            elif parts[0] == 'end' and parts[1] == 'quantum' and len(parts) == 2:
                in_quantum_section = False
                continue
            elif parts[0] == 'start' and (parts[1] == 'settings' or parts[1] == 'general') and len(parts) == 2:
                in_settings_section = True
                settings_section = ""
                continue
            elif parts[0] == 'end' and (parts[1] == 'settings' or parts[1] == 'general') and len(parts) == 2:
                in_settings_section = False
                continue

            if in_classical_section:
                classical_section += line + "\n"
            elif in_quantum_section:
                quantum_section += line + "\n"
            elif in_settings_section:
                settings_section += line + "\n"
            else:
                settings_section += line + "\n"

    return classical_section, quantum_section, settings_section


def evaluate_expression(expression):
    """Safely evaluate simple math expressions."""
    try:
        return eval(expression, {"__builtins__": None}, {})
    except:
        return expression


def parse_value_token(tok):
    """Bool, numeric, or leave string."""
    t = tok.lower()
    if t == 'true':  return True
    if t == 'false': return False
    try:
        val = evaluate_expression(tok)
        return float(val) if isinstance(val, (int, float)) else val
    except:
        return tok


def parse_value_token(token):
    """Attempt to parse a value as int, float, or leave as string."""
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            if token.lower() == "true":
                return True
            elif token.lower() == "false":
                return False
            return token


def parseclassicalSection(input):
    comment_pattern = re.compile(r"(#|--|%)(.*)$")
    params = {}
    stack = []  # Stack of tuples: (section_name, section_dict)

    for raw_line in input.splitlines():
        # Remove comments and whitespace
        line = comment_pattern.sub('', raw_line).strip()
        if not line:
            continue

        parts = line.split()
        kw = parts[0]

        if kw == 'start' and len(parts) == 2:
            name = parts[1]
            new_section = {}
            if stack:
                # Add the new nested section to the current parent section
                parent = stack[-1][1]
                if name in parent:
                    raise ValueError(f"Duplicate section '{name}' inside '{stack[-1][0]}'")
                parent[name] = new_section
            else:
                if name in params:
                    raise ValueError(f"Duplicate top-level section '{name}'")
                params[name] = new_section
            stack.append((name, new_section))

        elif kw == 'end' and len(parts) == 2:
            end_name = parts[1]
            if not stack:
                raise ValueError(f"Unmatched end {end_name}")
            open_name, _ = stack.pop()
            if open_name != end_name:
                raise ValueError(f"Mismatched end: expected end {open_name}, got end {end_name}")

        else:
            # Normal key-value pair, goes into the current section dict
            current = stack[-1][1] if stack else params
            key, *vals = parts
            if not vals:
                current[key] = None  # Support for keys without values
            elif len(vals) == 1:
                current[key] = parse_value_token(vals[0])
            else:
                current[key] = [parse_value_token(v) for v in vals]

    if stack:
        open_secs = ", ".join(n for n, _ in stack)
        raise ValueError(f"Unclosed section(s): {open_secs}")

    return params


def parseQuantumSection(input):
    def prepareCoordinates(atoms, coords):
        molecule_coords = ""
        for index, atom in enumerate(atoms):
            molecule_coords += " " + atom
            molecule_coords += " " + str(coords[atom+str(index+1)][0])
            molecule_coords += " " + str(coords[atom+str(index+1)][1])
            molecule_coords += " " + str(coords[atom+str(index+1)][2])
            if index != (len(atoms)-1):
                molecule_coords += ";"
        return molecule_coords

    def convert_to_bohr(coords, units):
        if units.lower().startswith("angstrom"):
            factor = 1.8897259886
        elif units.lower().startswith("bohr"):
            factor = 1.0
        else:
            raise ValueError(f"Unknown units '{units}', must be 'angstrom' or 'bohr'")
        return { label: xyz * factor for label, xyz in coords.items() }

    comment_pattern = re.compile(r"(#|--|%)(.*)$")
    params = {}
    stack = []  # Stack of tuples: (section_name, section_dict)
    geometry = {'atoms': [], 'coords': {}}
    raw_coords = {}
    atom_counts = 0
    in_mol = False

    for raw_line in input.splitlines():
        # Remove comments and whitespace
        line = comment_pattern.sub('', raw_line).strip()
        if not line:
            continue

        if line.lower().startswith('start geometry'):
            in_mol = True
            continue
        if line.lower().startswith('end geometry'):
            in_mol = False
            continue
        if in_mol:
            parts = line.split()
            atom = parts[0]
            coords = list(map(float, parts[1:4]))
            atom_counts += 1
            label = f"{atom}{atom_counts}"
            geometry['atoms'].append(atom)
            raw_coords[label] = np.array(coords)
            continue

        parts = line.split()
        kw = parts[0]

        if kw == 'start' and len(parts) == 2:
            name = parts[1]
            new_section = {}
            if stack:
                # Add the new nested section to the current parent section
                parent = stack[-1][1]
                if name in parent:
                    raise ValueError(f"Duplicate section '{name}' inside '{stack[-1][0]}'")
                parent[name] = new_section
            else:
                if name in params:
                    raise ValueError(f"Duplicate top-level section '{name}'")
                params[name] = new_section
            stack.append((name, new_section))

        elif kw == 'end' and len(parts) == 2:
            end_name = parts[1]
            if not stack:
                raise ValueError(f"Unmatched end {end_name}")
            open_name, _ = stack.pop()
            if open_name != end_name:
                raise ValueError(f"Mismatched end: expected end {open_name}, got end {end_name}")

        else:
            # Normal key-value pair, goes into the current section dict
            current = stack[-1][1] if stack else params
            key, *vals = parts
            if not vals:
                current[key] = None  # Support for keys without values
            elif len(vals) == 1:
                current[key] = parse_value_token(vals[0])
            else:
                current[key] = [parse_value_token(v) for v in vals]

    if stack:
        open_secs = ", ".join(n for n, _ in stack)
        raise ValueError(f"Unclosed section(s): {open_secs}")

    geometry['coords'] = convert_to_bohr(raw_coords, params["rttddft"]["units"])
    params["rttddft"]['geometry'] = geometry
    params["rttddft"]['geometry']['molecule_coords'] = prepareCoordinates(params["rttddft"]['geometry']['atoms'], params["rttddft"]['geometry']['coords'])
    
    return params