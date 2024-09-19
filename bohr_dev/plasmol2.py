# /Users/bldrdge1/.conda/envs/meep/bin/python ../bohr_dev/plasmol2.py -m meep.in -b pyridine.in
import sys
import argparse
import logging
import sys
import os
import numpy as np
import meep as mp
import gif
import bohr
from collections import defaultdict
from datetime import datetime

class ContinuousSource:
    def __init__(self,
                 frequency,
                 sourceCenter,
                 sourceSize,
                 is_integrated=True,
                 component=mp.Ez):
        """
        Initializes a ContinuousSource object.

        Args:
            frequency (float): The frequency of the continuous source.
            sourceCenter (tuple): The center coordinates of the source.
            sourceSize (tuple): The size dimensions of the source.
            is_integrated (bool): If True, integrates the source over time.
            component (mp.Vector3): The component of the electric field for the source.
        """
        logging.debug(f"Initializing ContinuousSource with frequency: {frequency}")
        self.frequency = frequency
        self.is_integrated = is_integrated
        self.sourceCenter = mp.Vector3(sourceCenter)
        self.sourceSize = mp.Vector3(sourceSize[0], sourceSize[1], sourceSize[2])
        self.component = component
        self.source = mp.Source(
            mp.ContinuousSource(frequency=self.frequency,
                                is_integrated=self.is_integrated),
            center=self.sourceCenter,
            size=self.sourceSize,
            component=self.component
        )


class GaussianSource:
    def __init__(self,
                 frequencyCenter,
                 frequencyWidth,
                 sourceCenter,
                 sourceSize,
                 is_integrated=True,
                 component=mp.Ez):
        """
        Initializes a GaussianSource object.

        Args:
            frequencyCenter (float): The center frequency of the Gaussian source.
            frequencyWidth (float): The width of the Gaussian source in frequency.
            sourceCenter (tuple): The center coordinates of the source.
            sourceSize (tuple): The size dimensions of the source.
            is_integrated (bool): If True, integrates the source over time.
            component (mp.Vector3): The component of the electric field for the source.
        """
        logging.debug(f"Initializing GaussianSource with frequencyCenter: {frequencyCenter}, frequencyWidth: {frequencyWidth}")
        self.frequencyCenter = frequencyCenter
        self.frequencyWidth = frequencyWidth
        self.is_integrated = is_integrated
        self.sourceCenter = mp.Vector3(sourceCenter)
        self.sourceSize = mp.Vector3(sourceSize[0], sourceSize[1], sourceSize[2])
        self.component = component
        self.source = mp.Source(
            mp.GaussianSource(frequency=frequencyCenter,
                              width=frequencyWidth, is_integrated=is_integrated),
            center=self.sourceCenter,
            size=self.sourceSize,
            component=self.component
        )


class Simulation:
    def __init__(self,
                 inputFile,
                 simParams, 
                 molecule,
                 sourceType, 
                 symmetries, 
                 objectNP, 
                 outputPNG):
        """
        Initializes a Simulation object with various physical parameters.

        Args:
            inputfile (str): The input file used for Bohr calculations.
            sourceType (object): Source object (ContinuousSource or GaussianSource).
            cellLength (float): The length of the simulation cell.
            pmlThickness (float): Thickness of the perfectly matched layer (PML).
            positionMolecule (tuple): Position of the molecule in the simulation.
            symmetries (list): Symmetry conditions for the simulation.
            objectNP (mp.Object): The object or nanoparticle within the simulation.
            intensityMin (float): Minimum intensity value for visualization.
            intensityMax (float): Maximum intensity value for visualization.
            timeLength (float): Duration of the simulation in Meep time.
            resolution (int): Spatial resolution of the simulation grid.
            imageDirName (str): Directory name for storing images.
            responseCutOff (float): Electric field response cutoff for calling Bohr.
            surroundingMaterialIndex (float): Refractive index of the surrounding material.
        """

        # Define simulation parameters
        self.inputFile = inputFile
        self.cellLength = simParams['cellLength']
        self.pmlThickness = simParams['pmlThickness']
        self.timeLength = simParams['timeLength']
        self.resolution = simParams['resolution']
        self.responseCutOff = simParams['responseCutOff']

        logging.info(f"Initializing simulation with cellLength: {self.cellLength}, resolution: {self.resolution}")

        self.positionMolecule = mp.Vector3(
            molecule['center'][0], 
            molecule['center'][1], 
            molecule['center'][2]) if molecule else None
        self.sourceType = sourceType if sourceType else None
        self.imageDirName = outputPNG['imageDirName'] if outputPNG else None
        self.timestepsBetween = outputPNG['timestepsBetween'] if outputPNG else None
        self.intensityMin = outputPNG['intensityMin'] if outputPNG else None
        self.intensityMax = outputPNG['intensityMax'] if outputPNG else None

        # Conversion factors
        self.convertTimeMeeptoSI = 10 / 3
        self.convertTimeBohrtoSI = 0.024188843
        self.convertFieldMeeptoBohr = 1 / 1e-6 / 8.8541878128e-12 / 299792458.0 / 0.51422082e12

        # Simulation runtime variables
        self.timeStepMeep = None
        self.timeStepBohr = None
        self.dipoleResponse = {component: defaultdict(
            list) for component in ['x', 'y', 'z']}
        self.electricFieldArray = {component: []
                                   for component in ['x', 'y', 'z']}
        self.indexForComponents = ['x', 'y', 'z']
        self.fieldComponents = [mp.Ex, mp.Ey, mp.Ez]
        self.decimalPlaces = None
        self.cellVolume = mp.Vector3(
            self.cellLength, self.cellLength, self.cellLength)
        self.frameCenter = self.cellLength * self.resolution / 2
        self.sourcesList = [
            mp.Source(
                mp.CustomSource(src_func=self.chirpx, is_integrated=True),
                center=self.positionMolecule,
                component=mp.Ex
            ),
            mp.Source(
                mp.CustomSource(src_func=self.chirpy, is_integrated=True),
                center=self.positionMolecule,
                component=mp.Ey
            ),
            mp.Source(
                mp.CustomSource(src_func=self.chirpz, is_integrated=True),
                center=self.positionMolecule,
                component=mp.Ez
            )
        ] if molecule else []
        if self.sourceType: self.sourcesList.append(self.sourceType.source)
        self.pmlList = [mp.PML(thickness=self.pmlThickness)]
        self.symmetriesList = symmetries
        self.objectList = [objectNP] if objectNP else []
        self.sim = mp.Simulation(
            resolution=self.resolution,
            cell_size=self.cellVolume,
            boundary_layers=self.pmlList,
            sources=self.sourcesList,
            symmetries=self.symmetriesList,
            geometry=self.objectList,
            default_material=mp.Medium(index=simParams['surroundingMaterialIndex']),
            # Courant=0.3
        )

        # TODO: Lets get that timestepBohr number down (by 1/10?)
        self.timeStepMeep = self.sim.Courant / self.sim.resolution
        self.timeStepBohr = 2 * self.timeStepMeep * self.convertTimeMeeptoSI / self.convertTimeBohrtoSI

        for component in self.indexForComponents:
            self.dipoleResponse[component].update({
                str(0 * self.timeStepMeep): 0,
                str(0.5 * self.timeStepMeep): 0,
                str(1 * self.timeStepMeep): 0
            })

        # Determine the number of decimal places for time steps
        halfTimeStepString = str(self.timeStepMeep / 2)
        self.decimalPlaces = len(halfTimeStepString.split('.')[1])
        logging.debug(f"Decimal places for time steps: {self.decimalPlaces}")

    def chirpx(self, t):
        """
        Chirp function for the x-component of the dipole response.
        """
        return self.dipoleResponse['x'].get(str(round(t, self.decimalPlaces)), 0)

    def chirpy(self, t):
        """
        Chirp function for the y-component of the dipole response.
        """
        return self.dipoleResponse['y'].get(str(round(t, self.decimalPlaces)), 0)

    def chirpz(self, t):
        """
        Chirp function for the z-component of the dipole response.
        """
        return self.dipoleResponse['z'].get(str(round(t, self.decimalPlaces)), 0)

    def getElectricField(self, sim):
        """
        Retrieves the electric field values at the molecule's position during the simulation.

        Args:
            sim (mp.Simulation): The Meep simulation object.
        """

        # TODO: Subtract the electric field dumped by bohr from the previous step
        logging.info(f"Getting Electric Field at the molecule at time {np.round(sim.meep_time() * self.convertTimeMeeptoSI, 4)} fs")
        for i, componentName in enumerate(self.indexForComponents):
            field = np.mean(sim.get_array(component=self.fieldComponents[i],
                                          center=self.positionMolecule,
                                          size=mp.Vector3(1E-20, 1E-20, 1E-20)))
            self.electricFieldArray[componentName].append(
                field * self.convertFieldMeeptoBohr)
        return 0

    def callBohr(self, sim):
        """
        Calls Bohr calculations if the electric field exceeds the response cutoff.

        Args:
            sim (mp.Simulation): The Meep simulation object.
        """
        
        if sim.timestep() > 2:
            # averageFields = {
            #     'x': np.mean(self.electricFieldArray['x']),
            #     'y': np.mean(self.electricFieldArray['y']),
            #     'z': np.mean(self.electricFieldArray['z'])
            # }

            # Check if any field component is above the cutoff to decide if Bohr needs to be called
            if any(abs(self.electricFieldArray[component][2]) >= self.responseCutOff for component in ['x', 'y', 'z']):
                logging.info(f"Calling Bohr at time {np.round(sim.meep_time() * self.convertTimeMeeptoSI, 4)} fs")
                logging.info(f'\tElectric field given to Bohr:\n {formatDict(self.electricFieldArray)} in AU')
                bohrResults = bohr.run(
                    self.inputFile,
                    self.electricFieldArray['x'],
                    self.electricFieldArray['y'],
                    self.electricFieldArray['z'],
                    self.timeStepBohr
                )
                logging.info(f"\tBohr calculation results: {bohrResults} in AU")

                for i, componentName in enumerate(self.indexForComponents):
                    self.dipoleResponse[componentName][str(round(sim.meep_time() + (0.5 * self.timeStepMeep), self.decimalPlaces))] \
                        = bohrResults[i] / self.convertFieldMeeptoBohr
                    self.dipoleResponse[componentName][str(round(sim.meep_time() + self.timeStepMeep, self.decimalPlaces))] \
                        = bohrResults[i] / self.convertFieldMeeptoBohr

            # Remove first entry to make room for next entry
            for componentName in self.electricFieldArray:
                self.electricFieldArray[componentName].pop(0)

        return 0

    def run(self):
        """
        Runs the Meep simulation and generates a GIF of the electric field evolution.
        """
        if self.imageDirName: gif.clear_directory(self.imageDirName)
        logging.info("Meep simulation started.")
        if self.imageDirName: self.sim.use_output_directory(self.imageDirName)

        try:
            run_functions = []
            if self.positionMolecule:
                run_functions.append(mp.at_every(self.timeStepMeep, self.getElectricField))
                run_functions.append(mp.at_every(self.timeStepMeep, self.callBohr))

            if self.imageDirName:
                run_functions.append(mp.at_every(self.timestepsBetween * self.timeStepMeep,
                                                mp.output_png(mp.Ez, f"-X 10 -Y 10 -m {self.intensityMin} -M {self.intensityMax} -z {self.frameCenter} -Zc dkbluered")))

            self.sim.run(*run_functions, until=self.timeLength * self.timeStepMeep)
            # self.sim.run(
            #     mp.at_every(self.timeStepMeep, self.getElectricField),
            #     mp.at_every(self.timeStepMeep, self.callBohr),
            #     mp.at_every(self.timestepsBetween * self.timeStepMeep, mp.output_png(mp.Ez,
            #                 f"-X 10 -Y 10 -m {self.intensityMin} -M {self.intensityMax} -z {self.frameCenter} -Zc dkbluered")),
            #     until=self.timeLength * self.timeStepMeep
            # )
            logging.info("Simulation completed successfully!")

        except Exception as e:
            logging.error(f"Simulation failed with error: {e}")
        finally:
            if self.imageDirName: 
                gif.make_gif(self.imageDirName)
                logging.info("GIF creation completed")


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

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#') or line.startswith('--') or line.startswith('%') or not line:
                continue
            if line.startswith('start'):
                current_section = line.split()[1]
                params[current_section] = {}
            elif line.startswith('end'):
                current_section = None
            elif current_section:
                key, *values = line.split()
                if len(values) == 1:
                    if values[0].lower() == 'true':
                        params[current_section][key] = True
                    elif values[0].lower() == 'false':
                        params[current_section][key] = False
                    else:
                        try:
                            params[current_section][key] = float(values[0])
                        except ValueError:
                            params[current_section][key] = values[0]
                else:
                    processed_values = []
                    for v in values:
                        if v.lstrip('-').replace('.', '', 1).isdigit():
                            processed_values.append(float(v) if '.' in v else int(v))
                        else:
                            processed_values.append(v)
                    
                    params[current_section][key] = processed_values  
    
    formatted_dict = formatDict(params)
    logging.debug(f"Input file parsed contents:\n{formatted_dict}")
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
    simObj = Simulation(
        inputFile=bohrinputfile, # will always return something 
        simParams=getSimulation(parameters.get('simulation', {})), # will always return something 
        molecule=getMolecule(parameters.get('molecule', None)), # could return None
        sourceType=getSource(parameters.get('source', None)), # could return None
        symmetries=getSymmetry(parameters.get('simulation', {}).get('symmetries', None)), # could return None
        objectNP=getObject(parameters.get('object', None)), # could return None
        outputPNG=getOutputPNG(parameters.get('outputPNG', None)), # could return None
    )

    return simObj


def getMolecule(molParams):
    if not molParams:
        logging.debug('No molecule chosen for simulation. Continuing without it.')
        return None
    
    # Room to add future molecule params, otherwise should delete.

    return molParams


def getSimulation(simParams):
    if not simParams:
        logging.debug('No simulation parameters chosen for simulation. Continuing with default values.')
    
    simParams['cellLength'] = simParams.get('cellLeeeeeength', 0.1)
    simParams['pmlThickness'] = simParams.get('pmlThickness', 0.01)
    simParams['timeLength'] = simParams.get('timeLength', 500)
    simParams['resolution'] = simParams.get('resolution', 1000)
    simParams['responseCutOff'] = simParams.get('responseCutOff', 1e-12)
    simParams['surroundingMaterialIndex'] = simParams.get('surroundingMaterialIndex', 1.33)
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
        logging.debug('No source chosen for simulation. Continuing without it.')
        return None

    source_type = sourceParams['source_type']

    # sourceCenter recommended: -0.5 * cellLength + pmlThickness
    if source_type == 'continuous':
        source = ContinuousSource(
            frequency=sourceParams['frequency'],
            is_integrated=sourceParams['is_integrated'],
            sourceCenter=sourceParams['sourceCenter'],
            sourceSize=sourceParams['sourceSize']
        )

    elif source_type == 'gaussian':
        if 'frequencyCenter' in sourceParams and 'frequencyWidth' in sourceParams:
            frequency_center = sourceParams['frequencyCenter']
            frequency_width = sourceParams['frequencyWidth']
        elif 'frequencyMin' in sourceParams and 'frequencyMax' in sourceParams:
            frequency_center = (
                sourceParams['frequencyMin'] + sourceParams['frequencyMax']) / 2
            frequency_width = sourceParams['frequencyMax'] - sourceParams['frequencyMin']
        else:
            raise ValueError(
                "Either frequencyCenter and frequencyWidth or frequencyMin and frequencyMax must be provided for a GaussianSource.")

        source = GaussianSource(
            frequencyCenter=frequency_center,
            frequencyWidth=frequency_width,
            is_integrated=sourceParams['is_integrated'],
            sourceCenter=sourceParams['sourceCenter'],
            sourceSize=sourceParams['sourceSize']
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
        logging.debug('No object chosen for simulation. Continuing without it.')
        return None

    if objParams['material'] == 'Au':
        from meep.materials import Au_JC_visible as Au
        material = Au
    elif objParams['material'] == 'Ag':
        from meep.materials import Ag
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
        logging.debug('No symmetries chosen for simulation. Continuing without them.')
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
        logging.debug('No picture output chosen for simulation. Continuing without them.')
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
    - `-b` or `--bohr`: Path to the Bohr input file (required).
    - `-l` or `--log`: Log file name.
    - `-v` or `--verbose`: Increase verbosity of logging.

    Returns:
        argparse.Namespace: Parsed arguments.

    Exits:
        Exits the program with status code 1 if required arguments are not provided.
    """
    logging.debug("Processing command line arguments.")
    parser = argparse.ArgumentParser(description="Meep simulation with Bohr dipole moment calculation.")
    parser.add_argument('-m', '--meep', type=str, help="Path to the Meep input file.", required=True)
    parser.add_argument('-b', '--bohr', type=str, help="Path to the Bohr input file.", required=True)
    parser.add_argument('-l', '--log', help="Log file name")
    parser.add_argument('-v', '--verbose', action='count', default=0, help="Increase verbosity")

    args = parser.parse_args()

    if not args.meep:
        logging.error("Meep input file not provided. Exiting.")
        sys.exit(1)
    
    if not args.bohr:
        logging.error("Bohr input file not provided. Exiting.")
        sys.exit(1)

    logging.debug(f"Meep input file: {os.path.abspath(args.meep)}")
    logging.debug(f"Bohr input file: {os.path.abspath(args.bohr)}")
    
    return args


def formatDict(d, tabs=3):
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


if __name__ == "__main__":
    # log_format = '(%(filename)s)\t%(levelname)s:\t%(message)s'
    log_format = '%(levelname)s: %(message)s'
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log', help="Log file name")
    parser.add_argument('-v', '--verbose', action='count', default=0)
    temp_args = parser.parse_known_args()[0]

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

    args = processArguments()
    
    meepinputfile = args.meep
    bohrinputfile = args.bohr
    simObj = parseInputFile(meepinputfile)
    logging.info("Input file successfully parsed. Beginning simulation")
    simObj.run()
