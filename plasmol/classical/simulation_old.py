# classical/simulation.py
import logging
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from plasmol import constants
from plasmol.utils.csv import updateCSV
from plasmol.quantum.propagators import *
from plasmol.quantum.propagation import propagation

class SIMULATION:
    def __init__(self, params, molecule=None):
        self.params = params
        self.dt_meep = self.params.dt_au / constants.convertTimeMeep2Atomic
        self.t_end_meep = self.params.t_end_au / constants.convertTimeMeep2Atomic

        # Define simulation parameters
        self.cellLength = self.params.simulation_params['cellLength']
        self.pmlThickness = self.params.simulation_params['pmlThickness']
        self.resolution = self.params.simulation_params['resolution']
        self.eFieldCutOff = self.params.simulation_params['eFieldCutOff']

        logging.debug(f"Initializing simulation with cellLength: {self.cellLength}, resolution: {self.resolution}")
        
        self.moleculeBool = True if self.params.molecule_position else False
        self.positionMolecule = mp.Vector3(
            self.params.molecule_position['center'][0],
            self.params.molecule_position['center'][1],
            self.params.molecule_position['center'][2]) if self.moleculeBool else None
        
        if self.moleculeBool:
            self.molecule = molecule
            if self.params.propagator == "step":
                self.propagate = propagate_step
            elif self.params.propagator == "magnus2":
                self.propagate = propagate_magnus2
            elif self.params.propagator == "rk4":
                self.propagate = propagate_rk4
        
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
        
        self.source = self.params.source
        if self.source is not None:
            self.sourcesList.append(self.source.source)

        self.pmlList = [mp.PML(thickness=self.pmlThickness)]
        self.symmetry = self.params.symmetry
        self.nanoparticle = [self.params.nanoparticle] if self.params.nanoparticle is not None else []
        self.default_material = mp.Medium(index=self.params.simulation_params['surroundingMaterialIndex'])

        self.sim = mp.Simulation(
            resolution=self.resolution,
            cell_size=self.cellVolume,
            boundary_layers=self.pmlList,
            sources=self.sourcesList,
            symmetries=self.symmetry,
            geometry=self.nanoparticle,
            default_material=self.default_material
        )

        # Determine the number of decimal places for time steps
        halfTimeStepString = str(self.params.dt / 2)
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

    def callPropagation(self, sim):
        """
        Calls Quantum calculations if the electric field exceeds the response cutoff.

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

        if any(abs(eField[component]) >= self.eFieldCutOff for component in self.xyz):
            logging.info(f"Calling propagator at time {round(sim.meep_time() * constants.convertTimeMeep2Atomic, 4)} au")

            eArr = [eField['x'],eField['y'],eField['z']]
            logging.debug(f'Electric field given to propagator: {eArr} in au')
            
            ind_dipole = propagation(self.params, self.molecule, eArr, self.propagate)

            logging.debug(f"Propagation calculation results: {ind_dipole} in au")
            
            for componentName in self.xyz:
                for offset in [0.5 * self.dt_meep, self.dt_meep]:
                    timestamp = str(round(sim.meep_time() + offset, self.decimalPlaces))
                    self.measuredDipoleResponse[componentName][timestamp] = ind_dipole[self.mapDirectionToDigit[componentName]]

            timestamp = str(round((sim.meep_time() + self.dt_meep) * constants.convertTimeMeep2Atomic, self.decimalPlaces))
            updateCSV(self.params.pField_path, timestamp, ind_dipole[0], ind_dipole[1], ind_dipole[2])

        timestamp = str(round((sim.meep_time() + self.dt_meep) * constants.convertTimeMeep2Atomic, self.decimalPlaces))
        updateCSV(self.params.eField_path, timestamp, eField['x'], eField['y'], eField['z'])

    # ------------------------------------ #
    #              Additional              #
    #      custom tracking functions       #
    #          can be defined here         #
    #     similar to callPropagation()     #
    #   and added to the simulation loop   #
    #          as commented below          #
    #                                      #
    #     i.e. getElectricField() does     #
    #   the same as callPropagation() but  #
    #    without calling the propagation   #
    # ------------------------------------ #
    # def getElectricField(self, sim):
    #     """
    #     Retrieves the electric field values at the molecule's position during the simulation.

    #     Args:
    #         sim (mp.Simulation): The Meep simulation object.
    #     """
    #     logging.info(f"Getting Electric Field at the molecule at time {round(sim.meep_time() * constants.convertTimeMeep2Atomic, 4)} au")
    #     eField = {component: [] for component in self.xyz}
    #     for componentName in self.xyz:
    #         field = np.mean(sim.get_array(component=self.char_to_field[componentName],
    #                                       center=self.positionMolecule,
    #                                       size=mp.Vector3(1E-20, 1E-20, 1E-20)))
    #         eField[componentName] = field * constants.convertFieldMeep2Atomic

    #     timestamp = str(round((sim.meep_time() + self.dt_meep) * constants.convertTimeMeep2Atomic, self.decimalPlaces))
    #     updateCSV(self.params.eField_path, timestamp, eField['x'], eField['y'], eField['z'])



    def run(self):
        """
        Runs the Meep simulation and generates a GIF of the electric field evolution.
        """
        logging.info("Meep simulation started.")
        if self.hdf5Bool:
            from plasmol.utils.gif import clear_directory
            clear_directory(self.imageDirName)
            self.sim.use_output_directory(self.imageDirName)

        try:
            run_functions = []
            if self.hdf5Bool:
                run_functions.append(mp.at_every(self.timestepsBetween * self.dt_meep, mp.output_png(mp.Ez, f"-X 10 -Y 10 -m {self.intensityMin} -M {self.intensityMax} -z {self.frameCenter} -Zc dkbluered")))
            if self.moleculeBool:
                run_functions.append(mp.at_every(self.dt_meep, self.callPropagation))
            # ------------------------------------ #
            #              Additional              #
            #      custom tracking functions       #
            #           can be added here          #
            # ------------------------------------ #

            self.sim.run(*run_functions, until=self.t_end_meep)
            # show3Dmap(self.sim)
            # show2Dmap(self.sim)

            logging.info("Simulation completed successfully!")
        except Exception as e:
            logging.error(f"Simulation failed with error: {e}", exc_info=True)
        finally:
            if self.hdf5Bool:
                from plasmol.utils.gif import make_gif
                make_gif(self.imageDirName)
                import os 
                os.chdir('../')

# ------------------------------------ #
#         Example custom block         #
#         to show 3D map of NP         #
# ------------------------------------ #
def show3Dmap(sim):
    import plotly.graph_objects as go
    eps_data = sim.get_array(center=mp.Vector3(), size=self.cellVolume, component=mp.Dielectric)
    nx, ny, nz = eps_data.shape
    x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]
    iso_value = 4
    fig = go.Figure(data=go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=eps_data.flatten(),
        isomin=iso_value,
        isomax=iso_value,
        colorscale=[[0, 'gold'], [1, 'gold']],
        showscale=False,
        opacity=0.8,
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    fig.show()

# ------------------------------------ #
#         Example custom block         #
#         to show 2D map of NP         #
# ------------------------------------ #
def show2Dmap(sim):
    import plotly.graph_objects as go
    import numpy as np
    eps_data = sim.get_array(center=mp.Vector3(), size=sim.cell_size, component=mp.Dielectric)
    nx, ny, nz = eps_data.shape
    z_mid = nz // 2
    eps_slice = eps_data[:, :, z_mid]
    iso_value = 4
    # Binary mask: 1 where >= iso_value (dielectric), 0 otherwise
    z_plot = np.where(eps_slice >= iso_value, 1, 0)
    fig = go.Figure(data=go.Heatmap(
        z=z_plot,
        colorscale=[[0, 'white'], [1, 'black']],
        showscale=False
    ))
    fig.update_layout(
        title='2D Slice of Dielectric (XY plane at Z midpoint)',
        xaxis_title='X',
        yaxis_title='Y',
        yaxis=dict(scaleanchor='x')  # Make aspect ratio square
    )
    fig.show()