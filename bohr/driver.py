# /Users/bldrdge1/.conda/envs/meep1.29/bin/python /Users/bldrdge1/Downloads/repos/PlasMol/bohr/driver.py -m meep.in -b pyridine.in -l plasmol.log -vv
import sys
import argparse
import logging
import sys
import os
import meep as mp
import simulation as sim
import sources
import re
from datetime import datetime
import matrix_handler as mh

class PrintLogger(object):
    """Intercepts print statements and redirects them to a logger."""
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, message):
        message = message.rstrip()
        if message:
            self.logger.log(self.level, message)

    def flush(self):
        pass


def parseInputFile(filepath):
    """
    Parses an input file and converts its parameters into a Simulation object.

    Args:
        filepath (str): Path to the input file.

    Returns:
        Simulation: A configured Simulation object.
    """
    params = {}
    current_section = None

    comment_pattern = re.compile(r"(#|--|%)(.*)$")

    def evaluate_expression(expression):
        """Safely evaluate simple math expressions."""
        try:
            return eval(expression, {"__builtins__": None}, {})
        except:
            return expression

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            line = comment_pattern.split(line)[0].strip()

            if not line:
                continue

            if line.startswith('start '):
                current_section = line.split()[1]
                params[current_section] = {}
            elif line.startswith('end '):
                current_section = None
            elif current_section:
                key, *values = line.split()
                if len(values) == 1:
                    value = values[0]
                    if value.lower() == 'true':
                        params[current_section][key] = True
                    elif value.lower() == 'false':
                        params[current_section][key] = False
                    else:
                        try:
                            evaluated = evaluate_expression(value)
                            params[current_section][key] = float(evaluated) if isinstance(evaluated, (int, float)) else evaluated
                        except:
                            params[current_section][key] = value
                else:
                    processed_values = []
                    for v in values:
                        try:
                            evaluated = evaluate_expression(v)
                            processed_values.append(float(evaluated) if isinstance(evaluated, (int, float)) else evaluated)
                        except:
                            processed_values.append(v)
                    params[current_section][key] = processed_values

    simObj = setParameters(params)
    return simObj


def setParameters(parameters):
    """
    Sets up a Simulation object with the provided parameters.

    Args:
        parameters (dict): The simulation parameters.

    Returns:
        Simulation: A Simulation object initialized with the given parameters.
    """
    simObj = sim.Simulation(
        simParams=getSimulation(parameters.get('simulation', {})),
        bohrInputFile=parameters.get('molecule', {}).get('inputFile', bohrinputfile),
        molecule=getMolecule(parameters.get('molecule', None)),
        sourceType=getSource(parameters.get('source', None)),
        symmetries=getSymmetry(parameters.get('simulation', {}).get('symmetries', None)),
        objectNP=getObject(parameters.get('object', None)),
        outputPNG=getOutputPNG(parameters.get('outputPNG', None)),
        matplotlib=getMatPlotLib(parameters.get('matplotlib', None)),
        loggerStatus=args.verbose
    )

    return simObj


def getMatPlotLib(matParams):
    if not matParams:
        logging.info('No matplotlib chosen for simulation. Continuing without it.')
        return None

    matParams['output'] = matParams.get('output', None)
    matParams['CSVlocation'] = matParams.get('CSVlocation', None)
    matParams['IMGlocation'] = matParams.get('IMGlocation', None)
    return matParams


def getMolecule(molParams):
    if not molParams:
        logging.info('No molecule chosen for simulation. Continuing without it.')
        return None
    
    return molParams


def getSimulation(simParams):
    if not simParams:
        logging.info('No simulation parameters chosen for simulation. Continuing with default values.')
    
    simParams['cellLength'] = simParams.get('cellLength', 0.1)
    simParams['pmlThickness'] = simParams.get('pmlThickness', 0.01)
    simParams['resolution'] = simParams.get('resolution', 1000)
    simParams['responseCutOff'] = simParams.get('responseCutOff', 1e-12)
    simParams['surroundingMaterialIndex'] = simParams.get('surroundingMaterialIndex', 1.33)
        
    totalTime = simParams.get('totalTime', None)
    if totalTime:
        simParams['totalTime'] = totalTime[0]
        simParams['totalTimeUnit'] = totalTime[1]
    simParams['timeLength'] = simParams.get('timeLength', None)
    if simParams['timeLength'] is None and totalTime is None:
        raise ValueError("Must provide either timeLength or totalTime with proper unit. Neither found.")

    return simParams


def getSource(sourceParams):
    """
    Creates and returns the appropriate source object (ContinuousSource or GaussianSource) based on the parameters.

    Args:
        sourceParams (dict): Parameters defining the source type and its attributes.

    Returns:
        Source: A source object for the simulation, or None if invalid input.
    """
    if not sourceParams:
        logging.info('No source chosen for simulation. Continuing without it.')
        return None

    source_type = sourceParams['source_type']

    # sourceCenter recommended: -0.5 * cellLength + pmlThickness
    if source_type == 'continuous':
        source = sources.ContinuousSource(
            sourceCenter=sourceParams['sourceCenter'],
            sourceSize=sourceParams['sourceSize'],
            frequency=sourceParams.get('frequency', None),
            start_time=sourceParams.get('start_time', None),
            end_time=sourceParams.get('end_time', None),
            width=sourceParams.get('width', None),
            fwidth=sourceParams.get('fwidth', None),
            slowness=sourceParams.get('slowness', None),
            wavelength=sourceParams.get('wavelength', None),
            is_integrated=sourceParams.get('is_integrated', None),
            component=sourceParams.get('component', None)
        )

    elif source_type == 'gaussian':
        source = sources.GaussianSource(
            sourceCenter=sourceParams['sourceCenter'],
            sourceSize=sourceParams['sourceSize'],
            frequency=sourceParams.get('frequency', None),
            width=sourceParams.get('width', None),
            fwidth=sourceParams.get('fwidth', None),
            start_time=sourceParams.get('start_time', None),
            cutoff=sourceParams.get('cutoff', None),
            is_integrated=sourceParams.get('is_integrated', None),
            wavelength=sourceParams.get('wavelength', None),
            component=sourceParams.get('component', None)
        )
        
    elif source_type == 'chirped':
        source = sources.ChirpedSource(
            sourceCenter=sourceParams['sourceCenter'],
            sourceSize=sourceParams['sourceSize'],
            frequency=sourceParams.get('frequency', None),
            wavelength=sourceParams.get('wavelength', None),
            width=sourceParams.get('width', None),
            peakTime=sourceParams.get('peakTime', None),
            chirpRate=sourceParams.get('chirpRate', None),
            start_time=sourceParams.get('start_time', None),
            end_time=sourceParams.get('end_time', None),
            is_integrated=sourceParams.get('is_integrated', None),
            component=sourceParams.get('component', None)
        )

    elif source_type == 'pulse':
        source = sources.PulseSource(
            sourceCenter=sourceParams['sourceCenter'],
            sourceSize=sourceParams['sourceSize'],
            frequency=sourceParams.get('frequency', None),
            wavelength=sourceParams.get('wavelength', None),
            width=sourceParams.get('width', None),
            peakTime=sourceParams.get('peakTime', None),
            start_time=sourceParams.get('start_time', None),
            end_time=sourceParams.get('end_time', None),
            is_integrated=sourceParams.get('is_integrated', None),
            component=sourceParams.get('component', None)
        )

    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    return source


def getObject(objParams):
    """
    Creates and returns an object for the simulation based on material and geometric parameters.

    Args:
        objParams (dict): Parameters defining the object (e.g., material and radius).

    Returns:
        mp.Sphere: A nanoparticle object for the simulation.
    """
    if not objParams:
        logging.info('No object chosen for simulation. Continuing without it.')
        return None

    if objParams['material'] == 'Au':
        from mp.materials import Au_JC_visible as Au
        material = Au
    elif objParams['material'] == 'Ag':
        from mp.materials import Ag
        material = Ag
    else:
        raise ValueError(
            "Unsupported material type: {}".format(objParams['material']))

    objectNP = mp.Sphere(radius=objParams['radius'], center=objParams['center'], material=material)
    return objectNP


def getSymmetry(symParams):
    """
    Creates and returns a list of symmetry conditions for the simulation.

    Args:
        symParams (list): List of symmetry conditions and associated phase values.

    Returns:
        list: A list of symmetry conditions for the simulation.
    """
    if not symParams:
        logging.info('No symmetries chosen for simulation. Continuing without them.')
        return None
    
    symmetries = []
    for i in range(len(symParams)):
        if symParams[i] in ['X', 'Y', 'Z']:
            if i + 1 < len(symParams):
                try:
                    phase = int(symParams[i + 1])
                except ValueError:
                    raise ValueError(
                        f"Symmetry '{symParams[i]}' is not followed by a valid integer.")

                if symParams[i] == 'X':
                    symmetries.append(mp.Mirror(mp.X, phase=phase))
                elif symParams[i] == 'Y':
                    symmetries.append(mp.Mirror(mp.Y, phase=phase))
                elif symParams[i] == 'Z':
                    symmetries.append(mp.Mirror(mp.Z, phase=phase))
            else:
                raise ValueError(
                    f"Symmetry '{symParams[i]}' has no value following it.")
    if not symmetries:
        raise ValueError(f"Unsupported symmetry type: {symParams}")
    else:
        return symmetries


def getOutputPNG(pngParams):
    if not pngParams:
        logging.info('No picture output chosen for simulation. Continuing without it.')
        return None

    if any(key not in pngParams for key in ['timestepsBetween', 'intensityMin', 'intensityMax']):
        raise ValueError("If you want to generate pictures, you must provide timestepsBetween, intensityMin, and intensityMax.")

    if 'imageDirName' not in pngParams:
        pngParams['imageDirName'] = f"meep-{datetime.now().strftime('%m%d%Y_%H%M%S')}"
        logging.info(f"Directory for images: {os.path.abspath(pngParams['imageDirName'])}")

    return pngParams


def processArguments():
    """
    Parses command line arguments for the Meep simulation script.

    Command line arguments:
    - `-m` or `--meep`: Path to the Meep input file (required).
    - `-b` or `--bohr`: Path to the Bohr input file (optional).
    - `-l` or `--log`: Log file name.
    - `-v` or `--verbose`: Increase verbosity of logging.

    Returns:
        argparse.Namespace: Parsed arguments.

    Exits:
        Exits the program with status code 1 if required arguments are not provided.
    """
    logging.debug("Processing command line arguments.")
    parser = argparse.ArgumentParser(description="Meep simulation with Bohr dipole moment calculation.")
    parser.add_argument('-m', '--meep', '--input', type=str, help="Path to the Meep input file.")
    parser.add_argument('-b', '--bohr', '--molecule', type=str, help="Path to the Bohr input file.")
    parser.add_argument('-l', '--log', help="Log file name")
    parser.add_argument('-v', '--verbose', action='count', default=0, help="Increase verbosity")

    args = parser.parse_args()

    if args.log and args.verbose == 0:
        args.verbose = 1

    if not args.meep:
        for fallback_file in ['meep.in', 'sim.in']:
            if os.path.isfile(fallback_file):
                logging.info(f"Using fallback file: {fallback_file}")
                args.meep = fallback_file
                break
        else:
            logging.error("Meep input file not provided, and no fallback file found. Exiting.")
            sys.exit(1)
    
    logging.info(f"Meep input file: {os.path.abspath(args.meep)}")
    with open(os.path.abspath(args.meep), 'r') as file:
        for line in file:
            if line.strip().startswith('#') or line.strip().startswith('--') or line.strip().startswith('%'):
                continue
            logger.info('\t%s', line.rstrip('\n'))

    logger.info("")
    
    if args.bohr:
        logging.info(f"Bohr input file: {os.path.abspath(args.bohr)}")
        with open(os.path.abspath(args.bohr), 'r') as file:
            for line in file:
                if line.strip().startswith('#') or line.strip().startswith('--') or line.strip().startswith('%'):
                    continue
                logger.info('\t%s', line.rstrip('\n'))
        logger.info("")

    return args


def formatDict(d, tabs=1):
    """
    Formats a dictionary as a string with a specified number of tab indentations.

    Args:
        d (dict): The dictionary to format.
        tabs (int): The number of tab characters to use for indentation (default is 3).

    Returns:
        str: The formatted dictionary as a string with tab indentations.
    """
    import pprint

    formatted = pprint.pformat(d, indent=4)
    tab_prefix = '\t' * tabs
    return '\n'.join(tab_prefix + line for line in formatted.splitlines())


if __name__ == "__main__":
    # log_format = '(%(filename)s)\t%(levelname)s:\t%(message)s'
    # log_format = "%(asctime)s - %(name)s - %(filename)s - %(levelname)s - %(message)s"
    log_format = '%(levelname)s: %(message)s'
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--meep', '--input', type=str, help="Path to the Meep input file.")
    parser.add_argument('-b', '--bohr', '--molecule', type=str, help="Path to the Bohr input file.")
    parser.add_argument('-l', '--log', help="Log file name")
    parser.add_argument('-v', '--verbose', action='count', default=0, help="Increase verbosity")
    temp_args = parser.parse_known_args()[0]

    if temp_args.log and temp_args.verbose == 0:
        temp_args.verbose = 1

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if temp_args.log:
        file_handler = logging.FileHandler(temp_args.log, mode='w')
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)
    
    if temp_args.verbose >= 2:
        logger.setLevel(logging.DEBUG)
        mp.verbosity(3)
    elif temp_args.verbose == 1:
        logger.setLevel(logging.INFO)
        mp.verbosity(2)
    else:
        logger.setLevel(logging.WARNING)
        mp.verbosity(0)

    sys.stdout = PrintLogger(logger, logging.INFO)
    logging.getLogger("h5py").setLevel(logging.INFO)

    mh.clear_Matrix_Files()

    args = processArguments()
    
    meepinputfile = args.meep
    bohrinputfile = args.bohr if args.bohr else None
    simDriver = parseInputFile(meepinputfile)
    logging.info("Input file successfully parsed. Beginning simulation")
    simDriver.run()
