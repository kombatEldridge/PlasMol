import logging
import numpy as np
import meep as mp
import gif
import driver_rttddft
from collections import defaultdict
from csv_handler import initCSV, updateCSV, show_eField_pField

class Simulation:
    def __init__(self,
                 simParams,
                 bohrInputFile=None,
                 molecule=None,
                 sourceType=None,
                 symmetries=None,
                 objectNP=None,
                 outputPNG=None,
                 matplotlib=None,
                 loggerStatus=None,
                 ):
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
        self.cellLength = simParams['cellLength']
        self.pmlThickness = simParams['pmlThickness']
        self.totalTime = simParams['totalTime']
        self.totalTimeUnit = simParams['totalTimeUnit']
        self.timeLength = simParams['timeLength']
        self.resolution = simParams['resolution']
        self.responseCutOff = simParams['responseCutOff']

        logging.debug(f"Initializing simulation with cellLength: {self.cellLength}, resolution: {self.resolution}")
        
        self.molecule = True if molecule else None
        self.inputFile = bohrInputFile
        self.matplotlib = True if matplotlib else None
        self.outputPNG = True if outputPNG else None
        self.sourceType = sourceType if sourceType else None
        self.positionMolecule = mp.Vector3(
            molecule['center'][0],
            molecule['center'][1],
            molecule['center'][2]) if self.molecule else mp.Vector3(0, 0, 0)
        self.imageDirName = outputPNG['imageDirName'] if self.outputPNG else None
        self.timestepsBetween = outputPNG['timestepsBetween'] if self.outputPNG else None
        self.intensityMin = outputPNG['intensityMin'] if self.outputPNG else None
        self.intensityMax = outputPNG['intensityMax'] if self.outputPNG else None
        self.matplotlibOutput = matplotlib['output'] if self.matplotlib else None
        self.matplotlibLocationCSV = matplotlib['CSVlocation'] if self.matplotlib and matplotlib['CSVlocation'] else ""
        self.matplotlibLocationIMG = matplotlib['IMGlocation'] if self.matplotlib and matplotlib['IMGlocation'] else ""
        self.eFieldFileName = f'{self.matplotlibLocationCSV}{self.matplotlibOutput}-E-Field.csv' if self.matplotlib else None
        self.pFieldFileName = f'{self.matplotlibLocationCSV}{self.matplotlibOutput}-P-Field.csv' if self.matplotlib and self.molecule else None
        self.energyFileName = f'{self.matplotlibLocationCSV}{self.matplotlibOutput}-energy.csv' if self.matplotlib and self.molecule else None
        
        # Conversion factors
        self.convertTimeMeep2fs = 10 / 3
        self.convertTimeAtomic2fs = 0.024188843
        self.convertTimeMeep2Atomic = self.convertTimeMeep2fs / self.convertTimeAtomic2fs
        self.convertFieldMeep2Atomic = 1 / 1e-6 / 8.8541878128e-12 / 299792458.0 / 0.51422082e12
        self.convertMomentAtomic2Meep = 8.4783536198e-30 * 299792458.0 / 1 / 1e-6 / 1e-6

        # Simulation runtime variables
        self.xyz = ['x', 'y', 'z']
        self.measuredDipoleResponse = {component: defaultdict(list) for component in self.xyz}
        self.mapDirectionToDigit = {'x': 0, 'y': 1, 'z': 2}
        self.char_to_field = {'x': mp.Ex, 'y': mp.Ey, 'z': mp.Ez}
        self.decimalPlaces = None
        self.cellVolume = mp.Vector3(self.cellLength, self.cellLength, self.cellLength)
        self.frameCenter = self.cellLength * self.resolution / 2

        self.sourcesList = []
        if self.molecule:
            self.sourcesList.append(
                mp.Source(
                    mp.CustomSource(src_func=self.chirpx,is_integrated=True),
                    center=self.positionMolecule,
                    component=mp.Ex
                )
            )
            self.sourcesList.append(
                mp.Source(
                    mp.CustomSource(src_func=self.chirpy,is_integrated=True),
                    center=self.positionMolecule,
                    component=mp.Ey
                )
            )
            self.sourcesList.append(
                mp.Source(
                    mp.CustomSource(src_func=self.chirpz,is_integrated=True),
                    center=self.positionMolecule,
                    component=mp.Ez
                )
            )
            logging.debug("Emitter for the molecule added to simulation")

        if self.sourceType:
            self.sourcesList.append(self.sourceType.source)
        self.pmlList = [mp.PML(thickness=self.pmlThickness)]
        self.symmetries = symmetries
        self.objectList = [objectNP] if objectNP else []
        self.sim = mp.Simulation(
            resolution=self.resolution,
            cell_size=self.cellVolume,
            boundary_layers=self.pmlList,
            sources=self.sourcesList,
            symmetries=self.symmetries,
            geometry=self.objectList,
            default_material=mp.Medium(
                index=simParams['surroundingMaterialIndex']),
        )

        self.timeStepMeep = self.sim.Courant / self.sim.resolution
        self.timeStepBohr = 2 * self.timeStepMeep * self.convertTimeMeep2fs / self.convertTimeAtomic2fs

        # Determine the number of decimal places for time stepså
        halfTimeStepString = str(self.timeStepMeep / 2)
        self.decimalPlaces = len(halfTimeStepString.split('.')[1])

        voxelVol = (self.cellLength / self.sim.resolution)**3
        logging.debug(f"Voxel volume: {voxelVol} µm³ == {voxelVol * 1e12} Å³")

        logging.info(
            f"The timestep for this simulation is\n\t{round(self.timeStepMeep, self.decimalPlaces)} in Meep Units,\n\t{round(self.timeStepMeep * self.convertTimeMeep2Atomic, self.decimalPlaces)} in Atomic Units, and\n\t{round(self.timeStepMeep * self.convertTimeMeep2fs, self.decimalPlaces)} in fs")
        logging.debug(
            f"The timestep that will be handed to Bohr for the RK4 method is\n\t{round(self.timeStepBohr * self.convertTimeAtomic2fs, self.decimalPlaces)} fs ({round(self.timeStepBohr * self.convertTimeAtomic2fs * 1000, self.decimalPlaces)} as)")
        logging.debug(f"Decimal places for time steps: {self.decimalPlaces}")

        if self.totalTime:
            if self.totalTimeUnit == 'fs':
                self.timeLength = round(
                    self.totalTime / round(self.timeStepMeep * self.convertTimeMeep2fs, self.decimalPlaces))
            elif self.totalTimeUnit == 'as':
                self.timeLength = round(self.totalTime / 1000 / round(
                    self.timeStepMeep * self.convertTimeMeep2fs, self.decimalPlaces))
            elif self.totalTimeUnit == 'au':
                self.timeLength = round(
                    self.totalTime / round(self.timeStepBohr, self.decimalPlaces))
            elif self.totalTimeUnit == 'mu':
                self.timeLength = round(
                    self.totalTime / round(self.timeStepMeep, self.decimalPlaces))
            else:
                logging.error(
                    f"Time Limit Unit not recognized: {self.totalTimeUnit}")

            logging.debug(
                f"Using totalTime of {self.totalTime} {self.totalTimeUnit} to define timeLength of {self.timeLength}")
            
            if loggerStatus == 2:
                self.formattedDict = self.debug_format_input_params()
            elif loggerStatus == 1:
                self.formattedDict = "test"
            elif loggerStatus == 0:
                self.formattedDict = None

        if self.molecule:
            import molecule as mol
            self.moleculeObject = mol.MOLECULE(self.inputFile)
            self.method = self.moleculeObject.method["propagator"]
            self.coords = self.moleculeObject.molecule["coords"]
            self.wfn = self.moleculeObject.wfn
            self.D_mo_0 = self.moleculeObject.D_mo_0

    def chirpx(self, t):
        """
        Chirp function for the x-component of the dipole response.
        CustomSource with isIntegrated=True expects Polarization density
        https://github.com/NanoComp/meep/discussions/2809#discussioncomment-8929239
        """
        logging.debug(
            f"chirpx being called at {round(t * self.convertTimeMeep2fs, self.decimalPlaces)} fs. Emitting {self.measuredDipoleResponse['x'].get(str(round(t, self.decimalPlaces)), 0) * self.convertFieldMeep2Atomic} in atomic units.")
        return self.measuredDipoleResponse['x'].get(str(round(t, self.decimalPlaces)), 0) * self.convertMomentAtomic2Meep

    def chirpy(self, t):
        """
        Chirp function for the y-component of the dipole response.
        CustomSource with isIntegrated=True expects Polarization density
        https://github.com/NanoComp/meep/discussions/2809#discussioncomment-8929239
        """
        logging.debug(
            f"chirpy being called at {round(t * self.convertTimeMeep2fs, self.decimalPlaces)} fs. Emitting {self.measuredDipoleResponse['y'].get(str(round(t, self.decimalPlaces)), 0) * self.convertFieldMeep2Atomic} in atomic units.")
        return self.measuredDipoleResponse['y'].get(str(round(t, self.decimalPlaces)), 0) * self.convertMomentAtomic2Meep

    def chirpz(self, t):
        """
        Chirp function for the z-component of the dipole response.
        CustomSource with isIntegrated=True expects Polarization density
        https://github.com/NanoComp/meep/discussions/2809#discussioncomment-8929239
        """
        logging.debug(
            f"chirpz being called at {round(t * self.convertTimeMeep2fs, self.decimalPlaces)} fs. Emitting {self.measuredDipoleResponse['z'].get(str(round(t, self.decimalPlaces)), 0) * self.convertFieldMeep2Atomic} in atomic units.")
        return self.measuredDipoleResponse['z'].get(str(round(t, self.decimalPlaces)), 0) * self.convertMomentAtomic2Meep

    def getElectricField(self, sim):
        """
        Retrieves the electric field values at the molecule's position during the simulation.

        Args:
            sim (mp.Simulation): The Meep simulation object.
        """
        # TODO: Subtract the electric field dumped by bohr from the previous step, maybe?
        logging.info(f"Getting Electric Field at the molecule at time {round(sim.meep_time() * self.convertTimeMeep2fs, 4)} fs")
        eField = {component: [] for component in self.xyz}
        for componentName in self.xyz:
            field = np.mean(sim.get_array(component=self.char_to_field[componentName],
                                          center=self.positionMolecule,
                                          size=mp.Vector3(1E-20, 1E-20, 1E-20)))
            
            eField[componentName] = field * self.convertFieldMeep2Atomic

        if self.molecule:
            self.callBohr(sim, eField)

        timestamp = str(round((sim.meep_time() + self.timeStepMeep) * self.convertTimeMeep2fs, self.decimalPlaces))
        updateCSV(self.eFieldFileName, timestamp, eField['x'], eField['y'], eField['z'])

    def callBohr(self, sim, eField):
        """
        Calls Bohr calculations if the electric field exceeds the response cutoff.

        Args:
            sim (mp.Simulation): The Meep simulation object.
        """
        if sim.timestep() > 2:
            if any(abs(eField[component]) >= self.responseCutOff for component in self.xyz):
                logging.info(f"Calling Bohr at time {round(sim.meep_time() * self.convertTimeMeep2fs, 4)} fs")

                eArr = [eField['x'],eField['y'],eField['z']]

                logging.debug(f'Electric field given to Bohr: {eArr} in atomic units')

                # Will be expecting a matrix of [p_x, p_y, p_z] where p is the dipole moment
                bohrResults, energy = driver_rttddft.run(self.timeStepBohr, eArr, self.method, self.coords, self.wfn, self.D_mo_0)
                logging.debug(f"Bohr calculation results: {bohrResults} in atomic units")
                
                for componentName in self.xyz:
                    for offset in [0.5 * self.timeStepMeep, self.timeStepMeep]:
                        timestamp = str(round(sim.meep_time() + offset, self.decimalPlaces))
                        self.measuredDipoleResponse[componentName][timestamp] = bohrResults[self.mapDirectionToDigit[componentName]]

            timestamp = str(round((sim.meep_time() + self.timeStepMeep) * self.convertTimeMeep2fs, self.decimalPlaces))
            updateCSV(self.pFieldFileName, timestamp, bohrResults[0], bohrResults[1], bohrResults[2])
            updateCSV(self.energyFileName, timestamp, energy)

    def run(self):
        """
        Runs the Meep simulation and generates a GIF of the electric field evolution.
        """
        if self.outputPNG:
            gif.clear_directory(self.imageDirName)
            self.sim.use_output_directory(self.imageDirName)

        logging.info("Meep simulation started.")

        if self.matplotlib:
            e_field_comment = f"Simulation's Electric Field measured\nat the molecule's position in atomic units."
            if self.formattedDict:
                e_field_comment += f"\nJob Input:\n{self.formattedDict}"
            
            initCSV(self.eFieldFileName, e_field_comment)

            if self.molecule:
                p_field_comment = f"Molecule's Polarizability Field measured\nat the molecule's position in atomic units."
                if self.formattedDict:
                    p_field_comment += f"\nJob Input:\n{self.formattedDict}"
                
                initCSV(self.pFieldFileName, p_field_comment)
                
                energy_comment = f"Molecule's energy measured in atomic units."
                if self.formattedDict:
                    energy_comment += f"\nJob Input:\n{self.formattedDict}"

                initCSV(self.energyFileName, energy_comment)

        try:
            run_functions = [mp.at_every(self.timeStepMeep, self.getElectricField)]

            if self.outputPNG:
                run_functions.append(mp.at_every(self.timestepsBetween * self.timeStepMeep,
                                                 mp.output_png(mp.Ez, f"-X 10 -Y 10 -m {self.intensityMin} -M {self.intensityMax} -z {self.frameCenter} -Zc dkbluered")))

            self.sim.run(*run_functions, until=self.timeLength * self.timeStepMeep)
            logging.info("Simulation completed successfully!")
        except Exception as e:
            logging.error(f"Simulation failed with error: {e}", exc_info=True)
        finally:
            if self.matplotlib:
                try:
                    show_eField_pField(self.eFieldFileName, self.pFieldFileName, self.matplotlibLocationIMG, self.matplotlibOutput)
                except Exception as e:
                    logging.error(f"Graph failed to be made with error: {e}")
            if self.outputPNG:
                gif.make_gif(self.imageDirName)

    def debug_format_input_params(self):
        # Get all instance variables
        params = vars(self)

        # Function to recursively format nested structures (e.g., dicts, lists of objects)
        def recursive_format(value):
            if isinstance(value, dict):
                return {key: recursive_format(val) for key, val in value.items()}
            elif isinstance(value, list):
                return [recursive_format(item) for item in value]
            elif hasattr(value, "__dict__"):  # If the object has attributes, use dir()
                # Use dir() to get all the attributes and print them
                attributes = dir(value)
                attribute_details = {}
                for attr in attributes:
                    # Skip attributes or methods starting with '__' or '_'
                    if attr.startswith('__') or attr.startswith('_'):
                        continue
                    try:
                        attr_value = getattr(value, attr)
                        if callable(attr_value):
                            continue
                        attribute_details[attr] = attr_value
                    except AttributeError:
                        attribute_details[attr] = "<no value or method>"
                return attribute_details
            else:
                return str(value)

        # Format the instance variables
        formatted_params = {key: recursive_format(value) for key, value in params.items() if value is not None}
        
        return formatted_params

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
