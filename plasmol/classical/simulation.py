# classical/simulation.py
import logging
import meep as mp
import numpy as np
import os
import inspect
from collections import defaultdict

from plasmol import constants
from plasmol.utils.csv import updateCSV
from plasmol.quantum.propagators import *
from plasmol.quantum.propagation import propagation

class SIMULATION:
    def __init__(self, params, molecule=None):
        # set all key values that are in params as key values for self
        for key, value in params.__dict__.items():
            setattr(self, key, value)

        self.plasmon_resolution = round(0.5 / (self.dt / constants.convertTimeMeep2Atomic))
        self.dt_meep = self.dt / constants.convertTimeMeep2Atomic
        self.t_end_meep = self.t_end / constants.convertTimeMeep2Atomic

        logging.debug(f"Initializing simulation with cellLength: {self.plasmon_cell_length}, resolution: {self.plasmon_resolution}")

        # TODO: see if I can't fit this into params._attribute_formation()
        if self.has_molecule:
            self.molecule = molecule
            propagator_map = {
                "step": propagate_step,
                "magnus2": propagate_magnus2,
                "rk4": propagate_rk4
            }
            self.propagate = propagator_map.get(self.propagator, propagate_rk4)  # Default to rk4 if invalid
            sig = inspect.signature(self.propagate)
            exclude_args = {'molecule', 'field'}
            self.propagation_params = {name: getattr(self, name) for name in sig.parameters if name not in exclude_args}

        # Simulation runtime variables
        self.xyz = ['x', 'y', 'z']
        self.measured_dipole_response = {comp: defaultdict(lambda: 0) for comp in self.xyz}  # Use lambda:0 for default scalar
        self.map_direction_to_digit = {'x': 0, 'y': 1, 'z': 2}
        self.char_to_field = {'x': mp.Ex, 'y': mp.Ey, 'z': mp.Ez}
        self.frame_center = self.plasmon_cell_length * self.plasmon_resolution / 2

        # Determine decimal places for time steps (simplified: use fixed precision if possible, but keep for now)
        half_time_step_string = str(self.dt_meep / 2) # Should this be dt or dt_meep
        self.decimal_places = len(half_time_step_string.split('.')[1]) if '.' in half_time_step_string else 0

        self.sources_list = []
        if self.has_molecule:
            for comp, field in zip(self.xyz, [mp.Ex, mp.Ey, mp.Ez]):
                src_func = lambda t, c=comp: self._get_dipole_response(c, t)
                self.sources_list.append(
                    mp.Source(
                        mp.CustomSource(src_func=src_func, is_integrated=True),
                        center=self.molecule_position,
                        component=field
                    )
                )
            logging.debug("Emitter for the molecule added to simulation")

        if self.has_plasmon_source:
            self.sources_list.append(self.plasmon_source_object)

        self.pmlList = [mp.PML(thickness=self.plasmon_pml_thickness)]
        self.symmetry = self.symmetry
        self.nanoparticle = [self.nanoparticle] if self.nanoparticle else []
        self.default_material = mp.Medium(index=self.plasmon_surrounding_material_index)

        self.simulation = mp.Simulation(
            resolution=self.plasmon_resolution,
            cell_size=self.cell_volume,
            boundary_layers=self.pmlList,
            sources=self.sources_list,
            symmetries=self.symmetry,
            geometry=self.nanoparticle,
            default_material=self.default_material
        )

    def _get_dipole_response(self, component, t):
        """
        Helper to get dipole response for a component at time t.
        """
        timestamp = str(round(t, self.decimal_places))
        value = self.measured_dipole_response[component].get(timestamp, 0) * constants.convertMomentAtomic2Meep
        logging.debug(f"Getting dipole for {component} at {round(t * constants.convertTimeMeep2Atomic, self.decimal_places)} au. Emitting {value / constants.convertMomentAtomic2Meep * constants.convertFieldMeep2Atomic} in au.")
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
                center=self.molecule_position,
                size=mp.Vector3(1E-20, 1E-20, 1E-20)
            ))
            eField[comp] = field * constants.convertFieldMeep2Atomic
        return eField

    def _call_propagation(self, sim):
        """
        Calls Quantum calculations if the electric field exceeds the response cutoff.
        """
        eField = self._get_electric_field(sim)

        if any(abs(eField[comp]) >= self.plasmon_tolerance_efield for comp in self.xyz):
            logging.info(f"Calling propagator at time {round(sim.meep_time() * constants.convertTimeMeep2Atomic, 4)} au")
            eArr = [eField[c] for c in self.xyz]
            logging.debug(f'Electric field given to propagator: {eArr} in au')

            ind_dipole = propagation(self.propagation_params, self.molecule, eArr, self.propagate)
            logging.debug(f"Propagation calculation results: {ind_dipole} in au")

            for comp, digit in self.map_direction_to_digit.items():
                for offset in [0.5 * self.dt_meep, self.dt_meep]:
                    timestamp = str(round(sim.meep_time() + offset, self.decimal_places))
                    self.measured_dipole_response[comp][timestamp] = ind_dipole[digit]

            timestamp = str(round((sim.meep_time() + self.dt_meep) * constants.convertTimeMeep2Atomic, self.decimal_places))
            updateCSV(self.pField_path, timestamp, *ind_dipole)

        timestamp = str(round((sim.meep_time() + self.dt_meep) * constants.convertTimeMeep2Atomic, self.decimal_places))
        updateCSV(self.eField_path, timestamp, eField['x'], eField['y'], eField['z'])

    def run(self):
        """
        Runs the Meep simulation and generates a GIF of the electric field evolution if configured.
        """
        logging.info("Meep simulation started.")
        cwd = os.getcwd()

        try:
            run_functions = []
            if self.has_images:
                from plasmol.utils.gif import clear_directory
                clear_directory(self.images_dir_name)
                self.simulation.use_output_directory(self.images_dir_name)
                self.images_args = self.images_args
                self.images_args += f"-z {self.frame_center}"
                run_functions.append(mp.at_every(self.images_timesteps_between * self.dt_meep, mp.output_png(mp.Ez, self.images_args)))

            if self.has_molecule:
                run_functions.append(mp.at_every(self.dt_meep, self._call_propagation))

            # ------------------------------------ #
            #              Additional              #
            #      custom tracking functions       #
            #           can be added here          #
            # ------------------------------------ #
            # run_functions.append(...)

            self.simulation.run(*run_functions, until=self.t_end_meep)

            logging.info("Simulation completed successfully!")
        except Exception as e:
            logging.error(f"Simulation failed with error: {e}", exc_info=True)
        finally:
            # ------------------------------------ #
            #              Additional              #
            #      custom visualization calls      #
            #           can be added here          #
            # ------------------------------------ #
            # self.show3Dmap()
            # self.show2Dmap()

            # TODO: Ensure gifs work
            if self.has_images:
                if self.images_make_gif:
                    from plasmol.utils.gif import make_gif
                    make_gif(self.images_dir_name)
            os.chdir(cwd)

    # ------------------------------------ #
    #         Example custom methods       #
    #         to show maps of NP           #
    # ------------------------------------ #
    def show3Dmap(self):
        import plotly.graph_objects as go
        eps_data = self.simulation.get_array(center=mp.Vector3(), size=self.cell_volume, component=mp.Dielectric)
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
        eps_data = self.simulation.get_array(center=mp.Vector3(), size=self.simulation.cell_size, component=mp.Dielectric)
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