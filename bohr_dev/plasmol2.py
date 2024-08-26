import os
import numpy as np
import meep as mp
from scipy import constants
from gif import make_gif
import bohr
from meep.materials import Au_JC_visible as Au
from collections import defaultdict

class Simulation:
    def __init__(self, inputfile, resolution=1000, imageDirName="testingGifOutput2", responseCutOff=1e-12):
        self.inputfile = inputfile
        self.resolution = resolution
        self.imageDirName = imageDirName
        self.responseCutOff = responseCutOff

        # Conversion factors
        self.convertTimeMeeptoSI = 10 / 3
        self.convertTimeBohrtoSI = 0.024188843
        self.convertFieldMeeptoSI = 1 / 1e-6 / constants.epsilon_0 / constants.c / 0.51422082e12

        # Simulation parameters
        self.timeStepMeep = None
        self.timeStepBohr = None
        self.dipoleResponse = {component: defaultdict(list) for component in ['x', 'y', 'z']}
        self.electricFieldArray = {component: [] for component in ['x', 'y', 'z']}
        self.indexForComponents = ['x', 'y', 'z']
        self.fieldComponents = [mp.Ex, mp.Ey, mp.Ez]
        self.decimalPlaces = None

        # Initialize the MEEP simulation
        self.initialize_simulation()

    def initialize_simulation(self, radiusNP=0.025, cellLength=0.1, moleculeOffset=0.010):
        # Define simulation parameters
        self.radiusNP = radiusNP
        self.cellLength = cellLength
        self.cellVolume = mp.Vector3(self.cellLength, self.cellLength, self.cellLength)
        self.positionNP = mp.Vector3(0, 0, 0)
        self.positionMolecule = mp.Vector3(self.radiusNP + moleculeOffset, 0, 0)

        self.frequencyMin = 1 / 0.800
        self.frequencyMax = 1 / 0.400
        self.frequencyCenter = 0.5 * (self.frequencyMin + self.frequencyMax)
        self.frequencyWidth = self.frequencyMax - self.frequencyMin
        self.frequencySamplePts = 50

        self.pmlThickness = 0.01
        self.pmlList = [mp.PML(thickness=self.pmlThickness)]
        self.symmetriesList = [mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)]
        self.frameCenter = self.cellLength * self.resolution / 2

        self.sourcesList = [
            mp.Source(
                mp.ContinuousSource(frequency=100, is_integrated=True),
                center=mp.Vector3(-0.5 * self.cellLength + self.pmlThickness),
                size=mp.Vector3(0, self.cellLength, self.cellLength),
                component=mp.Ez
            ),
            # mp.Source(
            #     mp.GaussianSource(self.frequencyCenter, fwidth=self.frequencyWidth, is_integrated=True),
            #     center=mp.Vector3(-0.5*self.cellLength + self.pmlThickness),
            #     size=mp.Vector3(0, self.cellLength, self.cellLength),
            #     component=mp.Ez,
            # ),
            mp.Source(
                mp.CustomSource(src_func=self.chirpx),
                center=self.positionMolecule,
                component=mp.Ex
            ),
            mp.Source(
                mp.CustomSource(src_func=self.chirpy),
                center=self.positionMolecule,
                component=mp.Ey
            ),
            mp.Source(
                mp.CustomSource(src_func=self.chirpz),
                center=self.positionMolecule,
                component=mp.Ez
            )
        ]

        self.objectList = [mp.Sphere(radius=self.radiusNP, 
                                     center=self.positionNP, 
                                     material=mp.Medium(index=1.33))]

        self.sim = mp.Simulation(
            resolution=self.resolution,
            cell_size=self.cellVolume,
            boundary_layers=self.pmlList,
            sources=self.sourcesList,
            symmetries=self.symmetriesList,
            geometry=self.objectList,
            default_material=mp.Medium(index=1.33)
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

    def chirpx(self, t):
        return self.dipoleResponse['x'].get(str(round(t, self.decimalPlaces)), 0)

    def chirpy(self, t):
        return self.dipoleResponse['y'].get(str(round(t, self.decimalPlaces)), 0)

    def chirpz(self, t):
        return self.dipoleResponse['z'].get(str(round(t, self.decimalPlaces)), 0)

    def run(self):
        clear_directory(self.imageDirName)

        self.sim.use_output_directory(self.imageDirName)

        # For color scaling in the output (TODO: automate this)
        intensityMin = -3
        intensityMax = 3

        self.sim.run(
            mp.at_every(self.timeStepMeep, getElectricField),
            mp.at_every(3 * self.timeStepMeep, callBohr),
            mp.at_every(10 * self.timeStepMeep, mp.output_png(mp.Ez, f"-X 10 -Y 10 -m {intensityMin} -M {intensityMax} -z {self.frameCenter} -Zc dkbluered")),
            until=500 * self.timeStepMeep
        )

        make_gif(self.imageDirName)

def getElectricField(sim):
    global simObj
    print(f"Getting Electric Field at the molecule at time {np.round(sim.meep_time() * simObj.convertTimeMeeptoSI, 4)} fs")
    for i in range(3):
        componentName = simObj.indexForComponents[i]
        field = np.mean(sim.get_array(component=simObj.fieldComponents[i], 
                                        center=simObj.positionMolecule, 
                                        size=mp.Vector3(1E-20, 1E-20, 1E-20)))
        simObj.electricFieldArray[componentName].append(field * simObj.convertFieldMeeptoSI)
        simObj.dipoleResponse[componentName][str(round(sim.meep_time() + (0.5*simObj.timeStepMeep), simObj.decimalPlaces))] = 0
        simObj.dipoleResponse[componentName][str(round(sim.meep_time() + simObj.timeStepMeep, simObj.decimalPlaces))] = 0

def callBohr(sim):
    global simObj
    # Compute average electric field components
    averageFields = {
        'x': np.mean(simObj.electricFieldArray['x']),
        'y': np.mean(simObj.electricFieldArray['y']),
        'z': np.mean(simObj.electricFieldArray['z'])
    }

    # Check if any field component is above the cutoff to decide if Bohr needs to be called
    if any(averageFields[component] >= simObj.responseCutOff for component in ['x', 'y', 'z']):
        # Call bohr.run once with all required data
        bohrResults = bohr.run(
            simObj.inputfile,
            simObj.electricFieldArray['x'],
            simObj.electricFieldArray['y'],
            simObj.electricFieldArray['z'],
            simObj.timeStepBohr
        )
        
        # Update dipoleResponse based on bohrResults
        for i, componentName in enumerate(simObj.indexForComponents):
            simObj.dipoleResponse[componentName][str(round(sim.meep_time() + (0.5 * simObj.timeStepMeep), simObj.decimalPlaces))] = bohrResults[i]
            simObj.dipoleResponse[componentName][str(round(sim.meep_time() + simObj.timeStepMeep, simObj.decimalPlaces))] = bohrResults[i]

    # Print directly in a single line with join
    print(f"\t Molecule's Response: " +
        ', '.join(f"{componentName}: {simObj.dipoleResponse[componentName][str(round(sim.meep_time() + simObj.timeStepMeep, simObj.decimalPlaces))]}" 
                    for componentName in simObj.indexForComponents) + "\n")
    
    # Reset electricFieldArray for the next iteration
    simObj.electricFieldArray = {component: [] for component in simObj.indexForComponents}

def clear_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file_name in files:
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        print(f"All files in {directory_path} have been deleted.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage Example:
if __name__ == "__main__":
    import sys
    inputfile = sys.argv[1]
    simObj = Simulation(inputfile)
    simObj.run()
