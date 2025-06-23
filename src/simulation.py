import logging
import meep as mp
import numpy as np
import constants
from collections import defaultdict



from csv_utils import updateCSV

class Simulation:
    def __init__(self, params, molecule):
        self.molecule = molecule
        self.eField_path = params.eField_path
        self.pField_path = params.pField_path
        self.dt = params.dt
        self.dt_meep = self.dt / constants.convertTimeMeep2Atomic
        self.t_end = params.t_end
        self.t_end_meep = self.t_end / constants.convertTimeMeep2Atomic

        # Define simulation parameters
        self.cellLength = params.simParams['cellLength']
        self.pmlThickness = params.simParams['pmlThickness']
        self.resolution = params.simParams['resolution']
        self.eFieldCutOff = params.simParams['eFieldCutOff']

        logging.debug(f"Initializing simulation with cellLength: {self.cellLength}, resolution: {self.resolution}")
        
        self.moleculeBool = True if params.meepmolecule else None
        self.positionMolecule = mp.Vector3(
            params.meepmolecule['center'][0],
            params.meepmolecule['center'][1],
            params.meepmolecule['center'][2]) if self.moleculeBool else None
        
        self.hdf5Bool = True if params.hdf5 else None
        self.imageDirName = params.hdf5['imageDirName'] if self.hdf5Bool else None
        self.timestepsBetween = params.hdf5['timestepsBetween'] if self.hdf5Bool else None
        self.intensityMin = params.hdf5['intensityMin'] if self.hdf5Bool else None
        self.intensityMax = params.hdf5['intensityMax'] if self.hdf5Bool else None

        # Simulation runtime variables
        self.xyz = ['x', 'y', 'z']
        self.measuredDipoleResponse = {component: defaultdict(list) for component in self.xyz}
        self.mapDirectionToDigit = {'x': 0, 'y': 1, 'z': 2}
        self.char_to_field = {'x': mp.Ex, 'y': mp.Ey, 'z': mp.Ez}
        self.decimalPlaces = None
        self.cellVolume = mp.Vector3(self.cellLength, self.cellLength, self.cellLength)
        self.frameCenter = self.cellLength * self.resolution / 2

        self.sourcesList = []
        if self.moleculeBool:
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
        
        self.sourceType = params.sourceType
        if self.sourceType is not None:
            self.sourcesList.append(self.sourceType.source)

        self.pmlList = [mp.PML(thickness=self.pmlThickness)]
        self.symmetries = params.symmetries
        self.objectList = [params.objectNP] if params.objectNP is not None else []
        self.default_material = mp.Medium(index=params.simParams['surroundingMaterialIndex'])

        self.sim = mp.Simulation(
            resolution=self.resolution,
            cell_size=self.cellVolume,
            boundary_layers=self.pmlList,
            sources=self.sourcesList,
            symmetries=self.symmetries,
            geometry=self.objectList,
            default_material=self.default_material
        )

        # Determine the number of decimal places for time steps
        halfTimeStepString = str(params.dt / 2)
        self.decimalPlaces = len(halfTimeStepString.split('.')[1])

    def chirpx(self, t):
        """
        Chirp function for the x-component of the dipole response.
        CustomSource with isIntegrated=True expects Polarization density
        https://github.com/NanoComp/meep/discussions/2809#discussioncomment-8929239
        """
        logging.debug(f"chirpx being called at {round(t * constants.convertTimeMeep2Atomic, self.decimalPlaces)} au. Emitting {self.measuredDipoleResponse['x'].get(str(round(t, self.decimalPlaces)), 0) * constants.convertFieldMeep2Atomic} in au.")
        return self.measuredDipoleResponse['x'].get(str(round(t, self.decimalPlaces)), 0) * constants.convertMomentAtomic2Meep

    def chirpy(self, t):
        """
        Chirp function for the y-component of the dipole response.
        CustomSource with isIntegrated=True expects Polarization density
        https://github.com/NanoComp/meep/discussions/2809#discussioncomment-8929239
        """
        logging.debug(f"chirpy being called at {round(t * constants.convertTimeMeep2Atomic, self.decimalPlaces)} au. Emitting {self.measuredDipoleResponse['y'].get(str(round(t, self.decimalPlaces)), 0) * constants.convertFieldMeep2Atomic} in au.")
        return self.measuredDipoleResponse['y'].get(str(round(t, self.decimalPlaces)), 0) * constants.convertMomentAtomic2Meep

    def chirpz(self, t):
        """
        Chirp function for the z-component of the dipole response.
        CustomSource with isIntegrated=True expects Polarization density
        https://github.com/NanoComp/meep/discussions/2809#discussioncomment-8929239
        """
        logging.debug(f"chirpz being called at {round(t * constants.convertTimeMeep2Atomic, self.decimalPlaces)} au. Emitting {self.measuredDipoleResponse['z'].get(str(round(t, self.decimalPlaces)), 0) * constants.convertFieldMeep2Atomic} in au.")
        return self.measuredDipoleResponse['z'].get(str(round(t, self.decimalPlaces)), 0) * constants.convertMomentAtomic2Meep

    def getElectricField(self, sim):
        """
        Retrieves the electric field values at the molecule's position during the simulation.

        Args:
            sim (mp.Simulation): The Meep simulation object.
        """
        logging.info(f"Getting Electric Field at the molecule at time {round(sim.meep_time() * constants.convertTimeMeep2Atomic, 4)} au")
        eField = {component: [] for component in self.xyz}
        for componentName in self.xyz:
            field = np.mean(sim.get_array(component=self.char_to_field[componentName],
                                          center=self.positionMolecule,
                                          size=mp.Vector3(1E-20, 1E-20, 1E-20)))
            eField[componentName] = field * constants.convertFieldMeep2Atomic

        if self.moleculeBool:
            self.callBohr(sim, eField)

        timestamp = str(round((sim.meep_time() + self.dt_meep) * constants.convertTimeMeep2Atomic, self.decimalPlaces))
        updateCSV(self.eField_path, timestamp, eField['x'], eField['y'], eField['z'])

    def callBohr(self, sim, eField):
        """
        Calls Bohr calculations if the electric field exceeds the response cutoff.

        Args:
            sim (mp.Simulation): The Meep simulation object.
        """
        if sim.timestep() > 2:
            if any(abs(eField[component]) >= self.eFieldCutOff for component in self.xyz):
                logging.info(f"Calling Bohr at time {round(sim.meep_time() * constants.convertTimeMeep2Atomic, 4)} au")

                eArr = [eField['x'],eField['y'],eField['z']]
                logging.debug(f'Electric field given to Bohr: {eArr} in au')
                
                # TODO:::::::::
                ind_dipole = driver_rttddft.run(self.timeStepBohr, eArr, self.method, self.coords, self.wfn, self.D_mo_0)
                # ::::::::::::::

                logging.debug(f"Bohr calculation results: {ind_dipole} in au")
                
                for componentName in self.xyz:
                    for offset in [0.5 * self.dt_meep, self.dt_meep]:
                        timestamp = str(round(sim.meep_time() + offset, self.decimalPlaces))
                        self.measuredDipoleResponse[componentName][timestamp] = ind_dipole[self.mapDirectionToDigit[componentName]]

            timestamp = str(round((sim.meep_time() + self.dt_meep) * constants.convertTimeMeep2Atomic, self.decimalPlaces))
            updateCSV(self.pField_path, timestamp, ind_dipole[0], ind_dipole[1], ind_dipole[2])

    def run(self):
        """
        Runs the Meep simulation and generates a GIF of the electric field evolution.
        """
        logging.info("Meep simulation started.")
        if self.hdf5Bool:
            from gif import clear_directory
            clear_directory(self.imageDirName)
            self.sim.use_output_directory(self.imageDirName)

        try:
            run_functions = [mp.at_every(self.dt_meep, self.getElectricField)]
            if self.hdf5Bool:
                run_functions.append(mp.at_every(self.timestepsBetween * self.dt_meep, mp.output_png(mp.Ez, f"-X 10 -Y 10 -m {self.intensityMin} -M {self.intensityMax} -z {self.frameCenter} -Zc dkbluered")))

            self.sim.run(*run_functions, until=self.t_end_meep * self.dt_meep)
            logging.info("Simulation completed successfully!")
        except Exception as e:
            logging.error(f"Simulation failed with error: {e}", exc_info=True)
        finally:
            if self.hdf5Bool:
                from gif import make_gif
                make_gif(self.imageDirName)
