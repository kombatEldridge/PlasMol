# /Users/bldrdge1/.conda/envs/meep/bin/python ../bohr_dev/plasmol2.py -m meep.in -b pyridine.in
import logging
import numpy as np
import pandas as pd
import csv
import meep as mp
import gif
import os
import bohr
from collections import defaultdict

class Simulation:
    def __init__(self,
                 bohrInputFile,
                 meepInputFile,
                 simParams, 
                 molecule,
                 sourceType, 
                 symmetries, 
                 objectNP, 
                 outputPNG, 
                 matplotlib):
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
        self.inputFile = bohrInputFile
        self.cellLength = simParams['cellLength']
        self.pmlThickness = simParams['pmlThickness']
        self.timeLength = simParams['timeLength']
        self.resolution = simParams['resolution']
        self.responseCutOff = simParams['responseCutOff']

        logging.debug(f"Initializing simulation with cellLength: {self.cellLength}, resolution: {self.resolution}")

        self.positionMolecule = mp.Vector3(
            molecule['center'][0], 
            molecule['center'][1], 
            molecule['center'][2]) if molecule else None
        self.turnOnMolecule = molecule.get('turnOn', True) if molecule else None
        self.sourceType = sourceType if sourceType else None
        self.imageDirName = outputPNG['imageDirName'] if outputPNG else None
        self.timestepsBetween = outputPNG['timestepsBetween'] if outputPNG else None
        self.intensityMin = outputPNG['intensityMin'] if outputPNG else None
        self.intensityMax = outputPNG['intensityMax'] if outputPNG else None
        self.matplotlib = True if matplotlib else None
        self.matplotlib_output = matplotlib['output'] if matplotlib else None
        self.eFieldFileName = f'{self.matplotlib_output}-E-Field.csv' if matplotlib else None
        self.pFieldFileName = f'{self.matplotlib_output}-P-Field.csv' if matplotlib else None

        with open(os.path.abspath(meepInputFile), 'r') as file:
            self.formattedDict = ""
            for line in file:
                if line.strip().startswith('#') or line.strip().startswith('--') or line.strip().startswith('%'):
                    continue
                self.formattedDict = "\t".join([self.formattedDict, line])

        # Conversion factors
        self.convertTimeMeeptoSI = 10 / 3
        self.convertTimeBohrtoSI = 0.024188843
        self.convertFieldMeeptoBohr = 1 / 1e-6 / 8.8541878128e-12 / 299792458.0 / 0.51422082e12
        self.convertMomentBohrtoMeep = 8.4783536198e-30 * 299792458.0 / 1 / 1e-6 / 1e-6 

        # Simulation runtime variables
        self.timeStepMeep = None
        self.timeStepBohr = None
        self.measuredDipoleResponse = {component: defaultdict(list) for component in ['x', 'y', 'z']}
        self.measuredElectricField = {component: [] for component in ['x', 'y', 'z']}
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
        )

        self.timeStepMeep = self.sim.Courant / self.sim.resolution
        self.timeStepBohr = 2 * self.timeStepMeep * self.convertTimeMeeptoSI / self.convertTimeBohrtoSI
        logging.info(f"The timestep for this simulation is\n\t{self.timeStepMeep} in Meep Units,\n\t{self.timeStepBohr} in Atomic Units, and\n\t{self.timeStepBohr*self.convertTimeBohrtoSI} in fs")

        # Determine the number of decimal places for time steps
        halfTimeStepString = str(self.timeStepMeep / 2)
        self.decimalPlaces = len(halfTimeStepString.split('.')[1])
        logging.debug(f"Decimal places for time steps: {self.decimalPlaces}")

        for component in self.indexForComponents:
            for i in np.arange(0, round(self.timeStepMeep * self.timeLength, self.decimalPlaces), self.timeStepMeep):
                self.measuredDipoleResponse[component].update({
                    str(round(0 * i, self.decimalPlaces)): 0,
                    str(round(0.5 * i, self.decimalPlaces)): 0,
                    str(round(1 * i, self.decimalPlaces)): 0
                })

    def chirpx(self, t):
        """
        Chirp function for the x-component of the dipole response.
        """
        return self.measuredDipoleResponse['x'].get(str(round(t, self.decimalPlaces)), 0)

    def chirpy(self, t):
        """
        Chirp function for the y-component of the dipole response.
        """
        return self.measuredDipoleResponse['y'].get(str(round(t, self.decimalPlaces)), 0)

    def chirpz(self, t):
        """
        Chirp function for the z-component of the dipole response.
        """
        logging.debug(f"CustomSource being called at {np.round(t, self.decimalPlaces)} fs. Returning: {self.measuredDipoleResponse['z'].get(str(round(t, self.decimalPlaces)), 0)} in MU.")
        return self.measuredDipoleResponse['z'].get(str(round(t, self.decimalPlaces)), 0)

    def getElectricField(self, sim):
        """
        Retrieves the electric field values at the molecule's position during the simulation.

        Args:
            sim (mp.Simulation): The Meep simulation object.
        """
        # TODO: Subtract the electric field dumped by bohr from the previous step, maybe?
        logging.info(f"Getting Electric Field at the molecule at time {np.round(sim.meep_time() * self.convertTimeMeeptoSI, 6)} fs")
        for i, componentName in enumerate(self.indexForComponents):
            field = np.mean(sim.get_array(component=self.fieldComponents[i],
                                          center=self.positionMolecule,
                                          size=mp.Vector3(1E-20, 1E-20, 1E-20)))
            self.measuredElectricField[componentName].append(field * self.convertFieldMeeptoBohr)
        
        if self.matplotlib:
            self.updateCSV(filename=self.eFieldFileName, 
                        timestamp=str(round((sim.meep_time() + (self.timeStepMeep)) * self.convertTimeMeeptoSI, self.decimalPlaces)), 
                        x_value=self.measuredElectricField['x'][-1], 
                        y_value=self.measuredElectricField['y'][-1], 
                        z_value=self.measuredElectricField['z'][-1])


    def callBohr(self, sim):
        """
        Calls Bohr calculations if the electric field exceeds the response cutoff.

        Args:
            sim (mp.Simulation): The Meep simulation object.
        """
        if sim.timestep() > 2:
            if any(abs(self.measuredElectricField[component][-1]) >= self.responseCutOff for component in ['x', 'y', 'z']):
                logging.info(f"Calling Bohr at time {np.round(sim.meep_time() * self.convertTimeMeeptoSI, 4)} fs")
                logging.debug(f'Electric field given to Bohr: {self.measuredElectricField} in atomic units')
                
                bohrResults = bohr.run(
                    self.inputFile,
                    self.measuredElectricField['x'],
                    self.measuredElectricField['y'],
                    self.measuredElectricField['z'],
                    self.timeStepBohr
                )
                logging.debug(f"Bohr calculation results: {np.array(bohrResults)} in atomic units")

                for i, componentName in enumerate(self.indexForComponents):
                    for offset in [0.5 * self.timeStepMeep, self.timeStepMeep]:
                        timestamp = str(round(sim.meep_time() + offset, self.decimalPlaces))
                        self.measuredDipoleResponse[componentName][timestamp] = (
                            bohrResults[i] * self.convertMomentBohrtoMeep
                        )
                
                timestamp = str(round(sim.meep_time() + self.timeStepMeep, self.decimalPlaces))
                self.updateCSV(
                    filename=self.pFieldFileName,
                    timestamp=timestamp,
                    x_value=self.measuredDipoleResponse['x'][timestamp] / self.convertMomentBohrtoMeep,
                    y_value=self.measuredDipoleResponse['y'][timestamp] / self.convertMomentBohrtoMeep,
                    z_value=self.measuredDipoleResponse['z'][timestamp] / self.convertMomentBohrtoMeep
                )
            else:
                self.updateCSV(
                    filename=self.pFieldFileName,
                    timestamp=str(round((sim.meep_time() + self.timeStepMeep) * self.convertTimeMeeptoSI, self.decimalPlaces)),
                    x_value=0.0,
                    y_value=0.0,
                    z_value=0.0
                )
            
            for componentName in self.measuredElectricField:
                self.measuredElectricField[componentName].pop(0)
        else:
            self.updateCSV(
                filename=self.pFieldFileName,
                timestamp=str(round((sim.meep_time() + self.timeStepMeep) * self.convertTimeMeeptoSI, self.decimalPlaces)),
                x_value=0.0,
                y_value=0.0,
                z_value=0.0
            )
            

    def updateCSV(self, filename, timestamp=None, x_value=None, y_value=None, z_value=None, comment=None):
        if self.matplotlib:
            if comment:
                with open(filename, 'w', newline='') as file:
                    for line in comment.splitlines():
                        file.write(f"# {line}\n")
                    file.write("\n")
                    writer = csv.writer(file)
                    writer.writerow(['Timestamps (fs)', 'X Values', 'Y Values', 'Z Values'])
            
            if timestamp is not None and x_value is not None and y_value is not None and z_value is not None:
                with open(filename, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, x_value, y_value, z_value])

    def show(self):
        logging.debug(f"Reading CSV files: {self.eFieldFileName} and {self.pFieldFileName}")
        logging.getLogger('matplotlib').setLevel(logging.INFO)
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        data1 = pd.read_csv(self.eFieldFileName, comment='#')
        data2 = pd.read_csv(self.pFieldFileName, comment='#')

        timestamps1 = data1['Timestamps (fs)']
        x_values1 = data1['X Values']
        y_values1 = data1['Y Values']
        z_values1 = data1['Z Values']

        ax1.plot(timestamps1, x_values1, label='x', marker='o')
        ax1.plot(timestamps1, y_values1, label='y', marker='o')
        ax1.plot(timestamps1, z_values1, label='z', marker='o')
        ax1.set_title('Incident Electric Field')
        ax1.set_xlabel('Timestamps (fs)')
        ax1.set_ylabel('Electric Field Magnitude')
        ax1.legend()

        timestamps2 = data2['Timestamps (fs)']
        x_values2 = data2['X Values']
        y_values2 = data2['Y Values']
        z_values2 = data2['Z Values']

        ax2.plot(timestamps2, x_values2, label='x', marker='o')
        ax2.plot(timestamps2, y_values2, label='y', marker='o')
        ax2.plot(timestamps2, z_values2, label='z', marker='o')
        ax2.set_title('Molecule\'s Response')
        ax2.set_xlabel('Timestamps (fs)')
        ax2.set_ylabel('Polarization Field Magnitude')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f'{self.matplotlib_output}.png', dpi=1000)
        logging.debug(f"Matplotlib image written: {self.matplotlib_output}.png")

    
    def run(self):
        """
        Runs the Meep simulation and generates a GIF of the electric field evolution.
        """
        if self.imageDirName: gif.clear_directory(self.imageDirName)
        if self.imageDirName: self.sim.use_output_directory(self.imageDirName)

        logging.info("Meep simulation started.")

        if self.matplotlib:
            self.updateCSV(filename=self.eFieldFileName, comment=f"Simulation's Electric Field measured\nat the molecule's position in atomic units.\nJob Input:\n{self.formattedDict}")
            self.updateCSV(filename=self.pFieldFileName, comment=f"Molecule's Polarizability Field measured\nat the molecule's position in atomic units.\nJob Input:\n{self.formattedDict}")

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
            for component in self.indexForComponents:
                for i in np.arange(round(0.5 * self.timeStepMeep, self.decimalPlaces), round(self.timeStepMeep * (self.timeLength + 1.5), self.decimalPlaces), self.timeStepMeep):
                    try:
                        self.measuredDipoleResponse[component].pop(str(round(i, self.decimalPlaces)))
                    except:
                        continue
            if self.matplotlib:
                self.show()
            if self.imageDirName: 
                gif.make_gif(self.imageDirName)


def debugObj(obj):
    # obj = self.sourceType.source
    attributes = dir(obj)
    for attribute in attributes:
        if not attribute.startswith('__'):
            try:
                value = getattr(obj, attribute)
                print(f"{attribute}: {value}")
            except Exception as e:
                print(f"{attribute}: Could not retrieve (error: {e})")
