# input_parser.py
# import os
import re
# import sys
import logging
# import sources
# import argparse
# import meep as mp
import numpy as np
# import simulation as sim
# from datetime import datetime

logger = logging.getLogger("main")

def inputFilePrepare(pmif, mif, qif):
    # Declare priority of input format
    if sum(x is not None for x in (pmif, mif, qif)) >= 2:
        raise RuntimeError("Note, you have submitted too many input file paths. If you want to run a PlasMol simulation (Meep + RT-TDDFT), please use '-p' or '--pmif'. \nIf you want to just run a Meep simulation, please use '-m' or '--mif'. If you want to just run a RT-TDDFT simulation, please use '-q' or '--qif'.")
    elif sum(x is not None for x in (pmif, mif, qif)) == 0:
        raise RuntimeError("You haven't submitted any input file paths. If you want to run a PlasMol simulation (Meep + RT-TDDFT), please use '-p' or '--pmif'. \nIf you want to just run a Meep simulation, please use '-m' or '--mif'. If you want to just run a RT-TDDFT simulation, please use '-q' or '--qif'.")

    # Values given in input files will be overwritten by args given in cli
    if pmif is not None:
        building_params = parsePlasMolInputFile(pmif)
        if not hasattr(building_params, 'simulation'):
            raise RuntimeError("No 'simulation' block found in PlasMol input file. Is this a RT-TDDFT only input file? Please use '-q' or '--qif' to submit the RT-TDDFT (Quantum) Input File (aka qif).")
        if hasattr(building_params, 'molecule'):
            simulation_type = 'PlasMol'
            if not minimum_molecule_sufficiency(building_params.get('rttddft', None)):
                if not hasattr(building_params['molecule'], 'quantumInputFile'):
                    raise RuntimeError("No geometry nor quantumInputFile path found, but molecule block found. Please either specify the RT-TDDFT parameters, or remover the 'molecule' block.")
                else:
                    logger.warning("Minimum RT-TDDFT parameters not found in PlasMol input file. Using given quantumInputFile instead.")
                    qif_params = parseRTTDDFTInputFile(building_params['molecule']['quantumInputFile'])
                    logger.info("Ignoring RT-TDDFT relevant parameters in PlasMol.")
                    for key in qif_params:
                        building_params["molecule"][key] = qif_params[key]
            elif hasattr(building_params['molecule'], 'quantumInputFile'):
                logger.warning("Minimum RT-TDDFT parameters found in PlasMol input file. Ignoring given quantumInputFile.")
        else:
            logger.warning("No molecule block found in PlasMol input file. This simulation will only include a Meep simulation. In the future, please input a meep input file only using the '-m' or '--mif' flag.")
            simulation_type = 'Meep'
    elif qif is not None:
        logger.info("Only RT-TDDFT input file given. Running RT-TDDFT simulation only.")
        building_params = parseRTTDDFTInputFile(qif)
        simulation_type = 'RT-TDDFT'
    elif mif is not None:
        logger.info("Only Meep input file given. Running Meep simulation only.")
        building_params = parsePlasMolInputFile(mif)
        simulation_type = 'Meep'
    else:
        raise RuntimeError("The minimum required parameters were not given. Please check guidelines for information on minimal requirements.")
    
    return simulation_type, simulation_type


def parsePlasMolInputFile(filepath):
    """
    Parses an input file and converts its parameters into a Simulation object.

    Now also captures key/value pairs outside of any start…end block
    and places them in the top‐level params dict.

    Args:
        filepath (str): Path to the input file.

    Returns:
        dict: A nested dict of sections and top‐level settings.
    """
    comment_pattern = re.compile(r'#.*')
    params = {}
    stack = []   # holds tuples (section_name, section_dict)
    in_geometry = False
    geometry_lines = []

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

    with open(filepath) as f:
        for raw_line in f:
            # strip comments & whitespace  
            line = comment_pattern.sub('', raw_line).strip()
            if not line:
                continue

            parts = line.split()
            kw = parts[0]

            # entering a geometry block under rttddft?
            if kw == 'start' and len(parts) == 2 and parts[1] == 'geometry' and stack and stack[-1][0] == 'rttddft':
                geom_dict = {}
                stack[-1][1]['geometry'] = geom_dict
                stack.append(('geometry', geom_dict))
                in_geometry = True
                geometry_lines = []
                continue

            # if we're inside geometry, just collect raw lines until 'end geometry'
            if in_geometry:
                if kw == 'end' and parts[1] == 'geometry':
                    atoms = []
                    coords = {}
                    for idx, ln in enumerate(geometry_lines):
                        elems = ln.split()
                        atom = elems[0]
                        x, y, z = map(float, elems[1:4])
                        atoms.append(atom)
                        key = f"{atom}{idx+1}"
                        coords[key] = (x, y, z)
                    stack[-1][1]['atoms']  = atoms
                    stack[-1][1]['coords'] = coords
                    stack.pop()
                    in_geometry = False
                    geometry_lines = []
                else:
                    geometry_lines.append(line)
                continue

            # normal nested start…
            if kw == 'start' and len(parts) == 2:
                name = parts[1]
                new_section = {}
                if stack:
                    stack[-1][1][name] = new_section
                else:
                    params[name] = new_section
                stack.append((name, new_section))

            # normal end…
            elif kw == 'end' and len(parts) == 2:
                end_name = parts[1]
                if not stack:
                    raise ValueError(f"Unmatched end {end_name}")
                open_name, _ = stack.pop()
                if open_name != end_name:
                    raise ValueError(f"Mismatched end: expected end {open_name}, got end {end_name}")

            # key/value lines
            else:
                # if we're in a section, attach to it; otherwise attach to top‐level params
                current = stack[-1][1] if stack else params
                key, *vals = parts
                if len(vals) == 1:
                    current[key] = parse_value_token(vals[0])
                else:
                    current[key] = [parse_value_token(v) for v in vals]

    if stack:
        open_secs = ", ".join(n for n,_ in stack)
        raise ValueError(f"Unclosed section(s): {open_secs}")

    try:
        params['molecule_coords'] = prepareCoordinates(params['rttddft']['geometry']['atoms'], params['rttddft']['geometry']['coords'])
    except:
        pass

    return params


def parseRTTDDFTInputFile(filepath):
    def convert_to_bohr(coords, units):
        """
        coords: dict mapping labels (e.g. "O1") -> numpy.array([x,y,z])
        units: "angstrom" or "bohr"
        Returns a new dict with all coordinates in Bohr.
        """
        if units.lower().startswith("angstrom"):
            factor = 1.8897259886
        elif units.lower().startswith("bohr"):
            factor = 1.0
        else:
            raise ValueError(f"Unknown units '{units}', must be 'angstrom' or 'bohr'")
        return { label: xyz * factor for label, xyz in coords.items() }

    def _parse_value(value):
        """
        Try to convert a string value to int or float; leave as string if conversion fails.
        """
        try:
            if any(c in value.lower() for c in ('.', 'e')):
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value

    params = {}
    units = 'bohr'
    geometry = {'atoms': [], 'coords': {}}
    raw_coords = {}
    atom_counts = 0
    in_mol = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Detect start/end of molecule block
            if line.lower().startswith('start molecule'):
                in_mol = True
                continue
            if line.lower().startswith('end molecule'):
                in_mol = False
                continue

            # Parse molecule block
            if in_mol:
                parts = line.split()
                atom = parts[0]
                coords = list(map(float, parts[1:4]))
                atom_counts += 1
                label = f"{atom}{atom_counts}"

                geometry['atoms'].append(atom)
                raw_coords[label] = np.array(coords)
                continue

            # Parse key-value pairs
            key_val = line.split(None, 1)
            if len(key_val) == 2:
                key, val = key_val
                val = val.strip()
                if key.lower() == 'units':
                    units = val
                    params['units'] = val
                else:
                    params[key] = _parse_value(val)

    # Convert raw coordinates to Bohr
    geometry['coords'] = convert_to_bohr(raw_coords, units)
    params['geometry'] = geometry

    params['molecule_coords'] = prepareCoordinates(params['geometry']['atoms'], params['geometry']['coords'])
    return params


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


def minimum_molecule_sufficiency(params):
    return True


# def setParameters(parameters):
#     """
#     Sets up a Simulation object with the provided parameters.

#     Args:
#         parameters (dict): The simulation parameters.

#     Returns:
#         Simulation: A Simulation object initialized with the given parameters.
#     """
#     simObj = sim.Simulation(
#         simParams=getSimulation(parameters.get('simulation', {})),
#         bohrInputFile=parameters.get('molecule', {}).get('inputFile', bohrinputfile),
#         molecule=getMolecule(parameters.get('molecule', None)),
#         sourceType=getSource(parameters.get('source', None)),
#         symmetries=getSymmetry(parameters.get('simulation', {}).get('symmetries', None)),
#         objectNP=getObject(parameters.get('object', None)),
#         outputPNG=getOutputPNG(parameters.get('outputPNG', None)),
#         matplotlib=getMatPlotLib(parameters.get('matplotlib', None)),
#         loggerStatus=args.verbose
#     )

#     return simObj

# def getMatPlotLib(matParams):
#     if not matParams:
#         logging.info('No matplotlib chosen for simulation. Continuing without it.')
#         return None

#     matParams['output'] = matParams.get('output', None)
#     matParams['CSVlocation'] = matParams.get('CSVlocation', None)
#     matParams['IMGlocation'] = matParams.get('IMGlocation', None)
#     return matParams

# def getMolecule(molParams):
#     if not molParams:
#         logging.info('No molecule chosen for simulation. Continuing without it.')
#         return None
    
#     return molParams

# def getSimulation(simParams):
#     if not simParams:
#         logging.info('No simulation parameters chosen for simulation. Continuing with default values.')
    
#     simParams['cellLength'] = simParams.get('cellLength', 0.1)
#     simParams['pmlThickness'] = simParams.get('pmlThickness', 0.01)
#     simParams['resolution'] = simParams.get('resolution', 1000)
#     simParams['responseCutOff'] = simParams.get('responseCutOff', 1e-12)
#     simParams['surroundingMaterialIndex'] = simParams.get('surroundingMaterialIndex', 1.33)
        
#     totalTime = simParams.get('totalTime', None)
#     if totalTime:
#         simParams['totalTime'] = totalTime[0]
#         simParams['totalTimeUnit'] = totalTime[1]
#     simParams['timeLength'] = simParams.get('timeLength', None)
#     if simParams['timeLength'] is None and totalTime is None:
#         raise ValueError("Must provide either timeLength or totalTime with proper unit. Neither found.")

#     return simParams

# def getSource(sourceParams):
#     """
#     Creates and returns the appropriate source object (ContinuousSource or GaussianSource) based on the parameters.

#     Args:
#         sourceParams (dict): Parameters defining the source type and its attributes.

#     Returns:
#         Source: A source object for the simulation, or None if invalid input.
#     """
#     if not sourceParams:
#         logging.info('No source chosen for simulation. Continuing without it.')
#         return None

#     source_type = sourceParams['source_type']

#     # sourceCenter recommended: -0.5 * cellLength + pmlThickness
#     if source_type == 'continuous':
#         source = sources.ContinuousSource(
#             sourceCenter=sourceParams['sourceCenter'],
#             sourceSize=sourceParams['sourceSize'],
#             frequency=sourceParams.get('frequency', None),
#             start_time=sourceParams.get('start_time', None),
#             end_time=sourceParams.get('end_time', None),
#             width=sourceParams.get('width', None),
#             fwidth=sourceParams.get('fwidth', None),
#             slowness=sourceParams.get('slowness', None),
#             wavelength=sourceParams.get('wavelength', None),
#             is_integrated=sourceParams.get('is_integrated', None),
#             component=sourceParams.get('component', None)
#         )

#     elif source_type == 'gaussian':
#         source = sources.GaussianSource(
#             sourceCenter=sourceParams['sourceCenter'],
#             sourceSize=sourceParams['sourceSize'],
#             frequency=sourceParams.get('frequency', None),
#             width=sourceParams.get('width', None),
#             fwidth=sourceParams.get('fwidth', None),
#             start_time=sourceParams.get('start_time', None),
#             cutoff=sourceParams.get('cutoff', None),
#             is_integrated=sourceParams.get('is_integrated', None),
#             wavelength=sourceParams.get('wavelength', None),
#             component=sourceParams.get('component', None)
#         )
        
#     elif source_type == 'chirped':
#         source = sources.ChirpedSource(
#             sourceCenter=sourceParams['sourceCenter'],
#             sourceSize=sourceParams['sourceSize'],
#             frequency=sourceParams.get('frequency', None),
#             wavelength=sourceParams.get('wavelength', None),
#             width=sourceParams.get('width', None),
#             peakTime=sourceParams.get('peakTime', None),
#             chirpRate=sourceParams.get('chirpRate', None),
#             start_time=sourceParams.get('start_time', None),
#             end_time=sourceParams.get('end_time', None),
#             is_integrated=sourceParams.get('is_integrated', None),
#             component=sourceParams.get('component', None)
#         )

#     elif source_type == 'pulse':
#         source = sources.PulseSource(
#             sourceCenter=sourceParams['sourceCenter'],
#             sourceSize=sourceParams['sourceSize'],
#             frequency=sourceParams.get('frequency', None),
#             wavelength=sourceParams.get('wavelength', None),
#             width=sourceParams.get('width', None),
#             peakTime=sourceParams.get('peakTime', None),
#             start_time=sourceParams.get('start_time', None),
#             end_time=sourceParams.get('end_time', None),
#             is_integrated=sourceParams.get('is_integrated', None),
#             component=sourceParams.get('component', None)
#         )

#     else:
#         raise ValueError(f"Unsupported source type: {source_type}")

#     return source

# def getObject(objParams):
#     """
#     Creates and returns an object for the simulation based on material and geometric parameters.

#     Args:
#         objParams (dict): Parameters defining the object (e.g., material and radius).

#     Returns:
#         mp.Sphere: A nanoparticle object for the simulation.
#     """
#     if not objParams:
#         logging.info('No object chosen for simulation. Continuing without it.')
#         return None

#     if objParams['material'] == 'Au':
#         from mp.materials import Au_JC_visible as Au
#         material = Au
#     elif objParams['material'] == 'Ag':
#         from mp.materials import Ag
#         material = Ag
#     else:
#         raise ValueError(
#             "Unsupported material type: {}".format(objParams['material']))

#     objectNP = mp.Sphere(radius=objParams['radius'], center=objParams['center'], material=material)
#     return objectNP

# def getSymmetry(symParams):
#     """
#     Creates and returns a list of symmetry conditions for the simulation.

#     Args:
#         symParams (list): List of symmetry conditions and associated phase values.

#     Returns:
#         list: A list of symmetry conditions for the simulation.
#     """
#     if not symParams:
#         logging.info('No symmetries chosen for simulation. Continuing without them.')
#         return None
    
#     symmetries = []
#     for i in range(len(symParams)):
#         if symParams[i] in ['X', 'Y', 'Z']:
#             if i + 1 < len(symParams):
#                 try:
#                     phase = int(symParams[i + 1])
#                 except ValueError:
#                     raise ValueError(
#                         f"Symmetry '{symParams[i]}' is not followed by a valid integer.")

#                 if symParams[i] == 'X':
#                     symmetries.append(mp.Mirror(mp.X, phase=phase))
#                 elif symParams[i] == 'Y':
#                     symmetries.append(mp.Mirror(mp.Y, phase=phase))
#                 elif symParams[i] == 'Z':
#                     symmetries.append(mp.Mirror(mp.Z, phase=phase))
#             else:
#                 raise ValueError(
#                     f"Symmetry '{symParams[i]}' has no value following it.")
#     if not symmetries:
#         raise ValueError(f"Unsupported symmetry type: {symParams}")
#     else:
#         return symmetries

# def getOutputPNG(pngParams):
#     if not pngParams:
#         logging.info('No picture output chosen for simulation. Continuing without it.')
#         return None

#     if any(key not in pngParams for key in ['timestepsBetween', 'intensityMin', 'intensityMax']):
#         raise ValueError("If you want to generate pictures, you must provide timestepsBetween, intensityMin, and intensityMax.")

#     if 'imageDirName' not in pngParams:
#         pngParams['imageDirName'] = f"meep-{datetime.now().strftime('%m%d%Y_%H%M%S')}"
#         logging.info(f"Directory for images: {os.path.abspath(pngParams['imageDirName'])}")

#     return pngParams

# def processArguments():
#     """
#     Parses command line arguments for the Meep simulation script.

#     Command line arguments:
#     - `-m` or `--meep`: Path to the Meep input file (required).
#     - `-b` or `--bohr`: Path to the Bohr input file (optional).
#     - `-l` or `--log`: Log file name.
#     - `-v` or `--verbose`: Increase verbosity of logging.

#     Returns:
#         argparse.Namespace: Parsed arguments.

#     Exits:
#         Exits the program with status code 1 if required arguments are not provided.
#     """
#     logging.debug("Processing command line arguments.")
#     parser = argparse.ArgumentParser(description="Meep simulation with Bohr dipole moment calculation.")
#     parser.add_argument('-m', '--meep', '--input', type=str, help="Path to the Meep input file.")
#     parser.add_argument('-b', '--bohr', '--molecule', type=str, help="Path to the Bohr input file.")
#     parser.add_argument('-l', '--log', help="Log file name")
#     parser.add_argument('-v', '--verbose', action='count', default=0, help="Increase verbosity")

#     args = parser.parse_args()

#     if args.log and args.verbose == 0:
#         args.verbose = 1

#     if not args.meep:
#         for fallback_file in ['meep.in', 'sim.in']:
#             if os.path.isfile(fallback_file):
#                 logging.info(f"Using fallback file: {fallback_file}")
#                 args.meep = fallback_file
#                 break
#         else:
#             logging.error("Meep input file not provided, and no fallback file found. Exiting.")
#             sys.exit(1)
    
#     logging.info(f"Meep input file: {os.path.abspath(args.meep)}")
#     with open(os.path.abspath(args.meep), 'r') as file:
#         for line in file:
#             if line.strip().startswith('#') or line.strip().startswith('--') or line.strip().startswith('%'):
#                 continue
#             logger.info('\t%s', line.rstrip('\n'))

#     logger.info("")
    
#     if args.bohr:
#         logging.info(f"Bohr input file: {os.path.abspath(args.bohr)}")
#         with open(os.path.abspath(args.bohr), 'r') as file:
#             for line in file:
#                 if line.strip().startswith('#') or line.strip().startswith('--') or line.strip().startswith('%'):
#                     continue
#                 logger.info('\t%s', line.rstrip('\n'))
#         logger.info("")

#     return args

# def formatDict(d, tabs=1):
#     """
#     Formats a dictionary as a string with a specified number of tab indentations.

#     Args:
#         d (dict): The dictionary to format.
#         tabs (int): The number of tab characters to use for indentation (default is 3).

#     Returns:
#         str: The formatted dictionary as a string with tab indentations.
#     """
#     import pprint

#     formatted = pprint.pformat(d, indent=4)
#     tab_prefix = '\t' * tabs
#     return '\n'.join(tab_prefix + line for line in formatted.splitlines())
