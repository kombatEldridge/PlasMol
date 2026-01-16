# classical/simulation.py
import logging
import meep as mp
import numpy as np
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
        sim_params = self.params.simulation
        self.cellLength = sim_params['cellLength']
        self.pmlThickness = sim_params['pmlThickness']
        self.resolution = sim_params['resolution']
        self.eFieldCutOff = sim_params['eFieldCutOff']

        logging.debug(f"Initializing simulation with cellLength: {self.cellLength}, resolution: {self.resolution}")

        self.moleculeBool = bool(self.params.molecule_position)
        self.positionMolecule = mp.Vector3(*self.params.molecule_position['center']) if self.moleculeBool else None

        if self.moleculeBool:
            self.molecule = molecule
            propagator_map = {
                "step": propagate_step,
                "magnus2": propagate_magnus2,
                "rk4": propagate_rk4
            }
            self.propagate = propagator_map.get(self.params.propagator, propagate_rk4)  # Default to rk4 if invalid

        self.hdf5Bool = bool(params.hdf5)
        if self.hdf5Bool:
            hdf5_params = params.hdf5
            self.imageDirName = hdf5_params['imageDirName']
            self.timestepsBetween = hdf5_params['timestepsBetween']
            self.intensityMin = hdf5_params['intensityMin']
            self.intensityMax = hdf5_params['intensityMax']

        # Simulation runtime variables
        self.xyz = ['x', 'y', 'z']
        self.measuredDipoleResponse = {comp: defaultdict(lambda: 0) for comp in self.xyz}  # Use lambda:0 for default scalar
        self.mapDirectionToDigit = {'x': 0, 'y': 1, 'z': 2}
        self.char_to_field = {'x': mp.Ex, 'y': mp.Ey, 'z': mp.Ez}
        self.cellVolume = mp.Vector3(self.cellLength, self.cellLength, self.cellLength)
        self.frameCenter = self.cellLength * self.resolution / 2

        # Determine decimal places for time steps (simplified: use fixed precision if possible, but keep for now)
        halfTimeStepString = str(self.params.dt / 2)
        self.decimalPlaces = len(halfTimeStepString.split('.')[1]) if '.' in halfTimeStepString else 0

        self.sourcesList = []
        if self.moleculeBool:
            for comp, field in zip(self.xyz, [mp.Ex, mp.Ey, mp.Ez]):
                src_func = lambda t, c=comp: self._get_dipole_response(c, t)
                self.sourcesList.append(
                    mp.Source(
                        mp.CustomSource(src_func=src_func, is_integrated=True),
                        center=self.positionMolecule,
                        component=field
                    )
                )
            logging.debug("Emitter for the molecule added to simulation")

        if self.params.source is not None:
            self.sourcesList.append(self.params.source.source)

        self.pmlList = [mp.PML(thickness=self.pmlThickness)]
        self.symmetry = self.params.symmetry
        self.nanoparticle = [self.params.nanoparticle] if self.params.nanoparticle else []
        self.default_material = mp.Medium(index=sim_params['surroundingMaterialIndex'])

        self.sim = mp.Simulation(
            resolution=self.resolution,
            cell_size=self.cellVolume,
            boundary_layers=self.pmlList,
            sources=self.sourcesList,
            symmetries=self.symmetry,
            geometry=self.nanoparticle,
            default_material=self.default_material
        )

    def _get_dipole_response(self, component, t):
        """
        Helper to get dipole response for a component at time t.
        Replaces chirpx, chirpy, chirpz.
        """
        timestamp = str(round(t, self.decimalPlaces))
        value = self.measuredDipoleResponse[component].get(timestamp, 0) * constants.convertMomentAtomic2Meep
        logging.debug(f"Getting dipole for {component} at {round(t * constants.convertTimeMeep2Atomic, self.decimalPlaces)} au. Emitting {value / constants.convertMomentAtomic2Meep * constants.convertFieldMeep2Atomic} in au.")
        return value

    def _get_electric_field(self, sim):
        """
        Extracts electric field at molecule position.
        """
        logging.info(f"Getting Electric Field at the molecule at time {round(sim.meep_time() * constants.convertTimeMeep2Atomic, 4)} au")
        eField = {}
        for comp in self.xyz:
            field = np.mean(sim.get_array(
                component=self.char_to_field[comp],
                center=self.positionMolecule,
                size=mp.Vector3(1E-20, 1E-20, 1E-20)
            ))
            eField[comp] = field * constants.convertFieldMeep2Atomic
        return eField

    def callPropagation(self, sim):
        """
        Calls Quantum calculations if the electric field exceeds the response cutoff.
        """
        eField = self._get_electric_field(sim)

        if any(abs(eField[comp]) >= self.eFieldCutOff for comp in self.xyz):
            logging.info(f"Calling propagator at time {round(sim.meep_time() * constants.convertTimeMeep2Atomic, 4)} au")
            eArr = [eField[c] for c in self.xyz]
            logging.debug(f'Electric field given to propagator: {eArr} in au')

            ind_dipole = propagation(self.params, self.molecule, eArr, self.propagate)
            logging.debug(f"Propagation calculation results: {ind_dipole} in au")

            for comp, digit in self.mapDirectionToDigit.items():
                for offset in [0.5 * self.dt_meep, self.dt_meep]:
                    timestamp = str(round(sim.meep_time() + offset, self.decimalPlaces))
                    self.measuredDipoleResponse[comp][timestamp] = ind_dipole[digit]

            timestamp = str(round((sim.meep_time() + self.dt_meep) * constants.convertTimeMeep2Atomic, self.decimalPlaces))
            updateCSV(self.params.pField_path, timestamp, *ind_dipole)

        timestamp = str(round((sim.meep_time() + self.dt_meep) * constants.convertTimeMeep2Atomic, self.decimalPlaces))
        updateCSV(self.params.eField_path, timestamp, eField['x'], eField['y'], eField['z'])

    def run(self):
        """
        Runs the Meep simulation and generates a GIF of the electric field evolution if configured.
        """
        logging.info("Meep simulation started.")
        original_dir = os.getcwd()  # Save original directory

        try:
            if self.hdf5Bool:
                from plasmol.utils.gif import clear_directory
                clear_directory(self.imageDirName)
                self.sim.use_output_directory(self.imageDirName)
                os.chdir(self.imageDirName)  # Change to output dir if needed for make_gif

            run_functions = []
            if self.hdf5Bool:
                png_args = f"-X 10 -Y 10 -m {self.intensityMin} -M {self.intensityMax} -z {self.frameCenter} -Zc dkbluered"
                run_functions.append(mp.at_every(self.timestepsBetween * self.dt_meep, mp.output_png(mp.Ez, png_args)))

            if self.moleculeBool:
                run_functions.append(mp.at_every(self.dt_meep, self.callPropagation))

            # Add additional custom tracking functions here if needed

            self.sim.run(*run_functions, until=self.t_end_meep)
            # self.show3Dmap()
            # self.show2Dmap()

            logging.info("Simulation completed successfully!")
        except Exception as e:
            logging.error(f"Simulation failed with error: {e}", exc_info=True)
        finally:
            if self.hdf5Bool:
                from plasmol.utils.gif import make_gif
                make_gif(self.imageDirName)  # Assumes make_gif handles paths correctly; adjust if needed
            os.chdir(original_dir)  # Always return to original dir

    # ------------------------------------ #
    #         Example custom methods       #
    #         to show maps of NP           #
    # ------------------------------------ #
    def show3Dmap(self):
        import plotly.graph_objects as go
        eps_data = self.sim.get_array(center=mp.Vector3(), size=self.cellVolume, component=mp.Dielectric)
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

    def show2Dmap(self):
        import plotly.graph_objects as go
        eps_data = self.sim.get_array(center=mp.Vector3(), size=self.sim.cell_size, component=mp.Dielectric)
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