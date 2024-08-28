import os
import numpy as np
import meep as mp
import gif
import bohr
from meep.materials import Au_JC_visible as Au
from collections import defaultdict

class ContinuousSource:
    def __init__(self, 
                 frequency, 
                 is_integrated=True, 
                 center=mp.Vector3(-0.5 * 0.1 + 0.01), 
                 size=mp.Vector3(0, 0.1, 0.1), 
                 component=mp.Ez):
        self.frequency = frequency 
        self.is_integrated = is_integrated 
        self.center = center 
        self.size = size 
        self.component = component 

        self.update()

    def update(self):
        self.source = mp.Source(
            mp.ContinuousSource(frequency=self.frequency, is_integrated=self.is_integrated),
            center=self.center,
            size=self.size,
            component=self.component
        )

class GaussianSource:
    def __init__(self, 
                 frequencyCenter=None, 
                 frequencyWidth=None, 
                 frequencyMin=None,
                 frequencyMax=None,
                 is_integrated=True, 
                 center=mp.Vector3(-0.5 * 0.1 + 0.01), 
                 size=mp.Vector3(0, 0.1, 0.1), 
                 component=mp.Ez):
        if frequencyCenter is not None and frequencyWidth is not None:
            self.frequencyCenter = frequencyCenter
            self.frequencyWidth = frequencyWidth
        elif frequencyMin is not None and frequencyMax is not None:
            self.frequencyCenter = 0.5 * (frequencyMin + frequencyMax)
            self.frequencyWidth = frequencyMax - frequencyMin
        else:
            raise ValueError("You must provide either (frequencyCenter and frequencyWidth) or (frequencyMin and frequencyMax).")

        self.is_integrated = is_integrated 
        self.center = center
        self.size = size
        self.component = component

        self.update()

    def update(self):
        self.source = mp.Source(
            mp.GaussianSource(frequency=self.frequencyCenter, width=self.frequencyWidth, is_integrated=self.is_integrated),
            center=self.center,
            size=self.size,
            component=self.component
        )


class Simulation:
    def __init__(self, 
                 inputfile, 
                 sourceType,
                 resolution=1000, 
                 imageDirName="testingGifOutput2", 
                 responseCutOff=1e-12, 
                 radiusNP=0.025, 
                 cellLength=0.1, 
                 pmlThickness=0.01, 
                 moleculeOffset=0.010):
        self.inputfile = inputfile
        self.resolution = resolution
        self.imageDirName = imageDirName
        self.responseCutOff = responseCutOff

        # Conversion factors
        self.convertTimeMeeptoSI = 10 / 3
        self.convertTimeBohrtoSI = 0.024188843
        self.convertFieldMeeptoBohr = 1 / 1e-6 / 8.8541878128e-12 / 299792458.0 / 0.51422082e12

        # Simulation parameters
        self.timeStepMeep = None
        self.timeStepBohr = None
        self.dipoleResponse = {component: defaultdict(list) for component in ['x', 'y', 'z']}
        self.electricFieldArray = {component: [] for component in ['x', 'y', 'z']}
        self.indexForComponents = ['x', 'y', 'z']
        self.fieldComponents = [mp.Ex, mp.Ey, mp.Ez]
        self.decimalPlaces = None

        # Define simulation parameters
        self.radiusNP = radiusNP
        self.cellLength = cellLength
        self.cellVolume = mp.Vector3(self.cellLength, self.cellLength, self.cellLength)
        self.pmlThickness = pmlThickness
        self.frameCenter = self.cellLength * self.resolution / 2
        self.positionNP = mp.Vector3(0, 0, 0)
        self.positionMolecule = mp.Vector3(self.radiusNP + moleculeOffset, 0, 0)
        
        # Initialize the MEEP simulation
        self.initialize_simulation(sourceType)

    def initialize_simulation(self, sourceType):
        self.pmlList = [mp.PML(thickness=self.pmlThickness)]
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

        sourceType.center = mp.Vector3(-0.5 * self.cellLength + self.pmlThickness) 
        sourceType.size = mp.Vector3(0, self.cellLength, self.cellLength)
        sourceType.update()
        self.sourcesList.append(sourceType.source)

        self.symmetriesList = [mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)]
        self.objectList = [mp.Sphere(radius=self.radiusNP, center=self.positionNP, material=Au)]

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

    def getElectricField(self, sim):
        # print(f"Getting Electric Field at the molecule at time {np.round(sim.meep_time() * self.convertTimeMeeptoSI, 4)} fs")
        for i, componentName in enumerate(self.indexForComponents):
            field = np.mean(sim.get_array(component=self.fieldComponents[i], 
                                            center=self.positionMolecule, 
                                            size=mp.Vector3(1E-20, 1E-20, 1E-20)))
            self.electricFieldArray[componentName].append(field * self.convertFieldMeeptoBohr)
        # print(simObj.electricFieldArray)
        return 0
    
    def callBohr(self, sim):
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
                
                for i, componentName in enumerate(self.indexForComponents):
                    self.dipoleResponse[componentName][str(round(sim.meep_time() + (0.5 * self.timeStepMeep), self.decimalPlaces))] = bohrResults[i] / self.convertFieldMeeptoBohr
                    self.dipoleResponse[componentName][str(round(sim.meep_time() + self.timeStepMeep, self.decimalPlaces))] = bohrResults[i] / self.convertFieldMeeptoBohr

                print("Molecule's Response: ", bohrResults)
                # print(self.electricFieldArray)

            # Remove first entry to make room for next entry
            for componentName in self.electricFieldArray:
                self.electricFieldArray[componentName].pop(0)

        return 0

    def run(self):
        gif.clear_directory(self.imageDirName)

        self.sim.use_output_directory(self.imageDirName)

        # For color scaling in the output (TODO: automate this)
        intensityMin = -3
        intensityMax = 3

        try:
            self.sim.run(
                mp.at_every(self.timeStepMeep, self.getElectricField),
                mp.at_every(self.timeStepMeep, self.callBohr),
                # mp.at_every(10 * self.timeStepMeep, mp.output_png(mp.Ez, f"-X 10 -Y 10 -z {self.frameCenter} -Zc dkbluered")),
                mp.at_every(10 * self.timeStepMeep, mp.output_png(mp.Ez, f"-X 10 -Y 10 -m {intensityMin} -M {intensityMax} -z {self.frameCenter} -Zc dkbluered")),
                until = 500 * self.timeStepMeep
            )
        except Exception as e:
            print(f"Simulation failed with error: {e}")
        finally:
            gif.make_gif(self.imageDirName)

# Usage Example:
if __name__ == "__main__":
    import sys
    inputfile = sys.argv[1]
    cellLength=0.1
    pmlThickness=0.01
    continuousSource = ContinuousSource(frequency = 100)
    gaussianSource = GaussianSource(frequencyMin = 1 / 0.800, frequencyMax = 1 / 0.400)
    simObj = Simulation(inputfile, continuousSource)
    simObj.run()
