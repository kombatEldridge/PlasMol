# /Users/bldrdge1/.conda/envs/meep/bin/python ../bohr_dev/plasmol2.py -m meep.in -b pyridine.in
import logging
import numpy as np
import meep as mp
import gif
import bohr
from collections import defaultdict

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
        self.turnOnMolecule = molecule.get('turnOn', True)
        self.sourceType = sourceType if sourceType else None
        self.imageDirName = outputPNG['imageDirName'] if outputPNG else None
        self.timestepsBetween = outputPNG['timestepsBetween'] if outputPNG else None
        self.intensityMin = outputPNG['intensityMin'] if outputPNG else None
        self.intensityMax = outputPNG['intensityMax'] if outputPNG else None

        # Conversion factors
        self.convertTimeMeeptoSI = 10 / 3
        self.convertTimeBohrtoSI = 0.024188843
        self.convertFieldMeeptoBohr = 1 / 1e-6 / 8.8541878128e-12 / 299792458.0 / 0.51422082e12
        self.convertMomentBohrtoMeep = 8.4783536198e-30 * 299792458.0 / 1 / 1e-6 / 1e-6 

        # Simulation runtime variables
        self.timeStepMeep = None
        self.timeStepBohr = None
        self.dipoleResponse = {component: defaultdict(list) for component in ['x', 'y', 'z']}
        self.electricFieldArray = {component: [] for component in ['x', 'y', 'z']}
        self.permaArray = {component: defaultdict(list) for component in ['x', 'y', 'z']}
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
                component=mp.Ez, 
            )
        ] if (molecule and self.turnOnMolecule) else []
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

        # Determine the number of decimal places for time steps
        halfTimeStepString = str(self.timeStepMeep / 2)
        self.decimalPlaces = len(halfTimeStepString.split('.')[1])
        logging.debug(f"Decimal places for time steps: {self.decimalPlaces}")

        for component in self.indexForComponents:
            for i in np.arange(self.timeLength):
                self.dipoleResponse[component].update({
                    str(round(0 * i * self.timeStepMeep, self.decimalPlaces)): 0,
                    str(round(0.5 * i * self.timeStepMeep, self.decimalPlaces)): 0,
                    str(round(1 * i * self.timeStepMeep, self.decimalPlaces)): 0
                })
                self.permaArray[component].update({
                    str(round(0 * i * self.timeStepMeep, self.decimalPlaces)): 0,
                    # str(round(0.5 * i * self.timeStepMeep, self.decimalPlaces)): 0,
                    str(round(1 * i * self.timeStepMeep, self.decimalPlaces)): 0
                })


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
        logging.debug(f"CustomSource being called at {np.round(t * self.convertTimeMeeptoSI, self.decimalPlaces)} fs. Returning: {self.dipoleResponse['z'].get(str(round(t, self.decimalPlaces)), 0)} in MU.")
        return self.dipoleResponse['z'].get(str(round(t, self.decimalPlaces)), 0)

    def getElectricField(self, sim):
        """
        Retrieves the electric field values at the molecule's position during the simulation.

        Args:
            sim (mp.Simulation): The Meep simulation object.
        """

        # TODO: Subtract the electric field dumped by bohr from the previous step
        logging.info(f"Getting Electric Field at the molecule at time {np.round(sim.meep_time() * self.convertTimeMeeptoSI, 6)} fs")
        for i, componentName in enumerate(self.indexForComponents):
            field = np.mean(sim.get_array(component=self.fieldComponents[i],
                                          center=self.positionMolecule,
                                          size=mp.Vector3(1E-20, 1E-20, 1E-20)))
            self.electricFieldArray[componentName].append(field * self.convertFieldMeeptoBohr)
            self.permaArray[componentName][str(round(sim.meep_time() + (0.5 * self.timeStepMeep), self.decimalPlaces))] \
                        = field * self.convertFieldMeeptoBohr

        return 0

    def callBohr(self, sim):
        """
        Calls Bohr calculations if the electric field exceeds the response cutoff.

        Args:
            sim (mp.Simulation): The Meep simulation object.
        """
        if sim.timestep() > 2:
            # Check if any field component is above the cutoff to decide if Bohr needs to be called
            if any(abs(self.electricFieldArray[component][2]) >= self.responseCutOff for component in ['x', 'y', 'z']):
                logging.info(f"Calling Bohr at time {np.round(sim.meep_time() * self.convertTimeMeeptoSI, 4)} fs")
                logging.info(f'\tElectric field given to Bohr:\n {self.electricFieldArray} in AU')
                bohrResults = bohr.run(
                    self.inputFile,
                    self.electricFieldArray['x'],
                    self.electricFieldArray['y'],
                    self.electricFieldArray['z'],
                    self.timeStepBohr
                )
                logging.info(f"\tBohr calculation results: {np.array(bohrResults)* self.convertMomentBohrtoMeep} in MU")

                for i, componentName in enumerate(self.indexForComponents):
                    self.dipoleResponse[componentName][str(round(sim.meep_time() + (0.5 * self.timeStepMeep), self.decimalPlaces))] \
                        = bohrResults[i] * self.convertMomentBohrtoMeep
                    self.dipoleResponse[componentName][str(round(sim.meep_time() + self.timeStepMeep, self.decimalPlaces))] \
                        = bohrResults[i] * self.convertMomentBohrtoMeep

            # Remove first entry to make room for next entry
            for componentName in self.electricFieldArray:
                self.electricFieldArray[componentName].pop(0)

        return 0
    

    def run(self):
        """
        Runs the Meep simulation and generates a GIF of the electric field evolution.
        """
        if self.imageDirName: gif.clear_directory(self.imageDirName)
        if self.imageDirName: self.sim.use_output_directory(self.imageDirName)

        logging.info("Meep simulation started.")
        try:
            run_functions = []
            if self.positionMolecule:
                run_functions.append(mp.at_every(self.timeStepMeep, self.getElectricField))
                run_functions.append(mp.at_every(self.timeStepMeep, self.callBohr))

            if self.imageDirName:
                run_functions.append(mp.at_every(self.timestepsBetween * self.timeStepMeep,
                                                mp.output_png(mp.Ez, f"-X 10 -Y 10 -m {self.intensityMin} -M {self.intensityMax} -z {self.frameCenter} -Zc dkbluered")))

            self.sim.run(*run_functions, until=self.timeLength * self.timeStepMeep)
            logging.info("Simulation completed successfully!")
        except Exception as e:
            logging.error(f"Simulation failed with error: {e}")
        finally:
            print("self.permaArray in AU: ", self.permaArray)
            print("self.dipoleResponse in MU", self.dipoleResponse)
            show(self.permaArray)
            if self.imageDirName: 
                gif.make_gif(self.imageDirName)
                logging.info("GIF creation completed")

def show(data):
    import matplotlib.pyplot as plt
    from collections import defaultdict

    timestamps = list(data['x'].keys())

    timestamps = [float(t) for t in timestamps]

    x_values = list(data['x'].values())
    y_values = list(data['y'].values())
    z_values = list(data['z'].values())

    plt.plot(timestamps, x_values, label='x', marker='o')
    plt.plot(timestamps, y_values, label='y', marker='o')
    plt.plot(timestamps, z_values, label='z', marker='o')

    # Add labels and title
    plt.xlabel('Timestamps')
    plt.ylabel('Values')
    plt.title('X, Y, Z Values Over Time')
    plt.legend()

    # Display the plot
    plt.show()