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
import pprint

class ContinuousSource:
    def __init__(self,
                 frequency,
                 sourceCenter,
                 sourceSize,
                 is_integrated=True,
                 component=mp.Ez):
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
                 inputfile,
                 sourceType,
                 cellLength,
                 pmlThickness,
                 positionMolecule,
                 symmetries,
                 objectNP,
                 intensityMin,
                 intensityMax,
                 timeLength=500,
                 resolution=1000,
                 imageDirName=None,
                 responseCutOff=1e-12,
                 surroundingMaterialIndex=1.33):

        logging.info(f"Initializing simulation with cellLength: {cellLength}, resolution: {resolution}")
        if imageDirName is None:
            imageDirName = f"meep-{datetime.now().strftime('%m%d%Y_%H%M%S')}"
            logging.info(f"Directory for images: {os.path.abspath(imageDirName)}")

        # Define simulation parameters
        self.inputfile = inputfile
        self.resolution = resolution
        self.imageDirName = imageDirName
        self.responseCutOff = responseCutOff
        self.cellLength = cellLength
        self.pmlThickness = pmlThickness
        self.timeLength = timeLength
        self.positionMolecule = mp.Vector3(
            positionMolecule[0], positionMolecule[1], positionMolecule[2])
        self.sourceType = sourceType
        self.intensityMin = intensityMin
        self.intensityMax = intensityMax

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
        ]
        self.sourcesList.append(self.sourceType.source)
        self.pmlList = [mp.PML(thickness=self.pmlThickness)]
        self.symmetriesList = symmetries
        self.objectList = [objectNP]

        self.sim = mp.Simulation(
            resolution=self.resolution,
            cell_size=self.cellVolume,
            boundary_layers=self.pmlList,
            sources=self.sourcesList,
            symmetries=self.symmetriesList,
            geometry=self.objectList,
            default_material=mp.Medium(index=surroundingMaterialIndex)
        )

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
        return self.dipoleResponse['x'].get(str(round(t, self.decimalPlaces)), 0)

    def chirpy(self, t):
        return self.dipoleResponse['y'].get(str(round(t, self.decimalPlaces)), 0)

    def chirpz(self, t):
        return self.dipoleResponse['z'].get(str(round(t, self.decimalPlaces)), 0)

    def getElectricField(self, sim):
        logging.info(f"Getting Electric Field at the molecule at time {np.round(sim.meep_time() * self.convertTimeMeeptoSI, 4)} fs")
        for i, componentName in enumerate(self.indexForComponents):
            field = np.mean(sim.get_array(component=self.fieldComponents[i],
                                          center=self.positionMolecule,
                                          size=mp.Vector3(1E-20, 1E-20, 1E-20)))
            self.electricFieldArray[componentName].append(
                field * self.convertFieldMeeptoBohr)
        return 0

    def callBohr(self, sim):
        logging.debug(f"Calling Bohr at time step {sim.timestep()}")
        if sim.timestep() > 2:
            averageFields = {
                'x': np.mean(self.electricFieldArray['x']),
                'y': np.mean(self.electricFieldArray['y']),
                'z': np.mean(self.electricFieldArray['z'])
            }

            # Check if any field component is above the cutoff to decide if Bohr needs to be called
            if any(abs(averageFields[component]) >= self.responseCutOff for component in ['x', 'y', 'z']) and len(self.electricFieldArray['x']) == 3:
                bohrResults = bohr.run(
                    self.inputfile,
                    self.electricFieldArray['x'],
                    self.electricFieldArray['y'],
                    self.electricFieldArray['z'],
                    self.timeStepBohr
                )
                logging.info(f"Bohr calculation results: {bohrResults}")

                for i, componentName in enumerate(self.indexForComponents):
                    self.dipoleResponse[componentName][str(round(sim.meep_time(
                    ) + (0.5 * self.timeStepMeep), self.decimalPlaces))] = bohrResults[i] / self.convertFieldMeeptoBohr
                    self.dipoleResponse[componentName][str(round(sim.meep_time(
                    ) + self.timeStepMeep, self.decimalPlaces))] = bohrResults[i] / self.convertFieldMeeptoBohr

            # Remove first entry to make room for next entry
            for componentName in self.electricFieldArray:
                self.electricFieldArray[componentName].pop(0)

        return 0

    def run(self):
        gif.clear_directory(self.imageDirName)
        logging.info("Meep simulation started.")
        self.sim.use_output_directory(self.imageDirName)

        try:
            self.sim.run(
                mp.at_every(self.timeStepMeep, self.getElectricField),
                mp.at_every(self.timeStepMeep, self.callBohr),
                mp.at_every(10 * self.timeStepMeep, mp.output_png(mp.Ez,
                            f"-X 10 -Y 10 -m {self.intensityMin} -M {self.intensityMax} -z {self.frameCenter} -Zc dkbluered")),
                until=self.timeLength * self.timeStepMeep
            )
            logging.info("Simulation completed successfully!")
        except Exception as e:
            logging.error(f"Simulation failed with error: {e}")
        finally:
            gif.make_gif(self.imageDirName)
            logging.info("GIF creation completed")


def parseInputFile(filepath):
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
    
    formatted_dict = format_dict_with_tabs(params)
    logging.debug(f"Input file parsed contents:\n{formatted_dict}")
    simObj = setParameters(params)
    return simObj


def setParameters(parameters):
    simObj = Simulation(
    inputfile=bohrinputfile,  # Mandatory
    sourceType=setSource(parameters['source']),  # Mandatory
    cellLength=parameters['simulation']['cellLength'],  # Mandatory
    pmlThickness=parameters['simulation']['pmlThickness'],  # Mandatory
    positionMolecule=parameters['molecule']['center'],  # Mandatory
    symmetries=setSymmetry(parameters['simulation']['symmetries']),  # Mandatory
    objectNP=setObject(parameters['object']),  # Mandatory
    intensityMin=parameters['simulation']['intensityMin'],  # Mandatory
    intensityMax=parameters['simulation']['intensityMax'],  # Mandatory
    timeLength=parameters['simulation'].get('timeLength', 500),  # Optional
    resolution=parameters['simulation'].get('resolution', 1000),  # Optional
    imageDirName=parameters['simulation'].get('imageDirName', None),  # Optional
    responseCutOff=parameters['simulation'].get('responseCutOff', 1e-12),  # Optional
    surroundingMaterialIndex=parameters['simulation'].get('surroundingMaterialIndex', 1.33)  # Optional
    )

    return simObj


def setSource(sourceParams):
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
        raise ValueError("Unsupported source type: {}".format(source_type))

    return source


def setObject(objParams):
    if objParams['material'] == 'Au':
        from meep.materials import Au_JC_visible as Au
        material = Au
    elif objParams['material'] == 'Ag':
        from meep.materials import Ag
        material = Ag
    else:
        raise ValueError(
            "Unsupported material type: {}".format(objParams['material']))

    objectNP = mp.Sphere(
        radius=objParams['radius'], center=objParams['center'], material=material)
    return objectNP


def setSymmetry(symParams):
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


def processArguments():
    logging.debug("Processing command line arguments.")
    parser = argparse.ArgumentParser(description="Meep simulation with Bohr dipole moment calculation.")
    parser.add_argument('-m', '--meep', type=str, help="Path to the Meep input file.")
    parser.add_argument('-b', '--bohr', type=str, help="Path to the Bohr input file.")
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


def format_dict_with_tabs(d, tabs=3):
    formatted = pprint.pformat(d, indent=4)
    tab_prefix = '\t' * tabs
    return '\n'.join(tab_prefix + line for line in formatted.splitlines())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=0)
    temp_args = parser.parse_known_args()[0]

    if temp_args.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG, format='(%(filename)s) \t%(levelname)s:\t%(message)s')
        mp.verbosity(3)
    elif temp_args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='(%(filename)s) \t%(levelname)s:\t%(message)s')
        mp.verbosity(2)
    else:
        logging.basicConfig(level=logging.WARNING, format='(%(filename)s) \t%(levelname)s:\t%(message)s')
        mp.verbosity(0)
        
    args = processArguments()
    
    meepinputfile = args.meep
    bohrinputfile = args.bohr
    simObj = parseInputFile(meepinputfile)
    logging.info("Input file successfully parsed. Beginning simulation")
    simObj.run()
