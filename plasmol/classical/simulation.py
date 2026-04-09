import os
import logging
import meep as mp
import numpy as np

from plasmol import constants
from plasmol.utils.csv import update_csv
from plasmol.quantum.propagation import propagation


class SIMULATION:
    def __init__(self, params):
        # set all key values that are in params as key values for self
        for key, value in params.__dict__.items():
            setattr(self, key, value)

        self.plasmon_resolution = round(0.5 / (self.dt / constants.convertTimeMeep2Atomic))
        self.dt_meep = self.dt / constants.convertTimeMeep2Atomic
        self.t_end_meep = self.t_end / constants.convertTimeMeep2Atomic

        logging.debug(f"Initializing simulation with cellLength: {self.plasmon_cell_length}, resolution: {self.plasmon_resolution}")

        # Key = integer step index (step = round(t / dt_meep)), Value = induced dipole (au)
        self.measured_dipole_response = {component: {} for component in self.xyz}

        self.map_direction_to_digit = {'x': 0, 'y': 1, 'z': 2}
        self.char_to_field = {'x': mp.Ex, 'y': mp.Ey, 'z': mp.Ez}
        self.frame_center = self.plasmon_cell_length * self.plasmon_resolution / 2

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
            self.sources_list.append(self.plasmon_source_object.source)

        self.pmlList = [mp.PML(thickness=self.plasmon_pml_thickness)]
        self.nanoparticle = [self.nanoparticle] if self.nanoparticle else []
        self.default_material = mp.Medium(index=self.plasmon_surrounding_material_index)

        self.simulation = mp.Simulation(
            resolution=self.plasmon_resolution,
            cell_size=self.cell_volume,
            boundary_layers=self.pmlList,
            sources=self.sources_list,
            symmetries=self.plasmon_symmetries,
            geometry=self.nanoparticle,
            default_material=self.default_material
        )

    def _get_step(self, t_meep: float) -> int:
        """Convert Meep time (in Meep units) to nearest integer step index."""
        return round(t_meep / self.dt_meep)

    def _get_dipole_response(self, component: str, t_meep: float):
        """
        Helper to get dipole response for a component at time t (in Meep units).
        Uses integer step index for perfect numerical stability.
        """
        step = self._get_step(t_meep)
        value_au = self.measured_dipole_response[component].get(step, 0.0)
        value_meep = value_au * constants.convertMomentAtomic2Meep

        logging.debug(f"Getting dipole for {component} at step {step} "
                      f"(t={t_meep*constants.convertTimeMeep2Atomic:.6f} au). "
                      f"Emitting {value_au:.6e} au")
        return value_meep

    def _get_electric_field(self, sim):
        """
        Extracts electric field at molecule position.
        """
        t_au = sim.meep_time() * constants.convertTimeMeep2Atomic
        logging.info(f"Getting Electric Field at the molecule at time {round(t_au, 4)} au")

        field_e = {}
        for comp in self.xyz:
            field = np.mean(sim.get_array(
                component=self.char_to_field[comp],
                center=self.molecule_position,
                size=mp.Vector3(1E-20, 1E-20, 1E-20)
            ))
            field_e[comp] = field * constants.convertFieldMeep2Atomic
        return field_e

    def _call_propagation(self, sim):
        """
        Calls Quantum calculations if the electric field exceeds the response cutoff.
        Stores induced dipole using integer step indices.
        """
        field_e = self._get_electric_field(sim)
        current_t_meep = sim.meep_time()
        current_step = self._get_step(current_t_meep)

        if any(abs(field_e[comp]) >= self.plasmon_tolerance_field_e for comp in self.xyz):
            logging.info(f"Calling propagator at time {round(current_t_meep * constants.convertTimeMeep2Atomic, 4)} au (step {current_step})")

            eArr = [field_e[c] for c in self.xyz]
            logging.debug(f'Electric field given to propagator: {eArr} in au')

            ind_dipole = propagation(
                params=self.molecule_propagator_params,
                molecule=self.molecule,
                exc=eArr,
                propagator=self.molecule_propagator
            )
            logging.debug(f"Propagation calculation results: {ind_dipole} in au")

            # Store the new dipole for the next step(s) that Meep's CustomSource may query
            next_step = current_step + 1
            for comp, digit in self.map_direction_to_digit.items():
                self.measured_dipole_response[comp][next_step] = ind_dipole[digit]
                # Extra safety for any internal half-step queries
                self.measured_dipole_response[comp][next_step + 1] = ind_dipole[digit]

            # Write polarization (induced dipole) to CSV in atomic units
            timestamp_au = (current_t_meep + self.dt_meep) * constants.convertTimeMeep2Atomic
            update_csv(self.field_p_filepath, timestamp_au, *ind_dipole)

        # Always write electric field to CSV in atomic units
        timestamp_au = (current_t_meep + self.dt_meep) * constants.convertTimeMeep2Atomic
        update_csv(self.field_e_filepath, timestamp_au, field_e['x'], field_e['y'], field_e['z'])

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
                self.images_args += f"-z {self.frame_center}"
                run_functions.append(mp.at_every(self.images_timesteps_between * self.dt_meep, mp.output_png(mp.Ez, self.images_args)))

            if self.has_molecule:
                run_functions.append(mp.at_every(self.dt_meep, self._call_propagation))

            self.simulation.run(*run_functions, until=self.t_end_meep)

            logging.info("Simulation completed successfully!")
        except Exception as e:
            logging.error(f"Simulation failed with error: {e}", exc_info=True)
        finally:
            if self.has_images and getattr(self, 'images_make_gif', True):
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
            yaxis=dict(scaleanchor='x')
        )
        fig.show()