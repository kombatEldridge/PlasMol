"""Full sequential formation (faithful port of PARAMS._attribute_formation)."""
import logging
import inspect
import numpy as np
import meep as mp

from plasmol.drivers import get_driver
from plasmol.quantum.propagators import *
from plasmol.quantum.sources import QUANTUMSOURCE
from plasmol.classical.sources import MEEPSOURCE
from plasmol.utils.params_helpers.common import construct_geometry, load_meep_material

logger = logging.getLogger("main")


def form_all(params):
    """Build derived attributes and objects needed to run the simulation.

    Turn validated parameters into runtime values (e.g. Meep/quantum objects,
    path strings, flags) used by the rest of the simulation.
    """

    self = params
    """
    This function is meant to form the attributes so they are ready to 
    be used by the rest of the codebase.
    """
    self.driver_str = getattr(self, 'driver_str', None)
    if self.has_custom:
        if self.driver_str is None:
            raise ValueError(f"Additional parameters specified but no driver name provided. Please specify a driver name.")
        logging.debug(f"Custom driver specified: {self.driver_str}")
    elif 'molecule' in self.simulation_types and 'plasmon' in self.simulation_types:
        self.driver_str = 'plasmol'
    elif 'molecule' in self.simulation_types:
        self.driver_str = 'quantum'
    elif 'plasmon' in self.simulation_types:
        self.driver_str = 'classical'

    self.driver = get_driver(self.driver_str)

    dt_str = f"{self.dt:.10f}".rstrip('0')
    self.time_rounding_decimals = len(dt_str.split('.')[-1]) if '.' in dt_str else 0

    if self.has_plasmon:
        if hasattr(self, 'plasmon_cell_volume'):
            self.cell_volume = mp.Vector3(*self.plasmon_cell_volume)
        else:
            self.cell_volume = mp.Vector3(self.plasmon_cell_length, self.plasmon_cell_length, self.plasmon_cell_length)
        if hasattr(self, 'plasmon_symmetries'):
            symmetries_list = []
            self.plasmon_symmetries_text = self.plasmon_symmetries
            dir_map = {'X': mp.X, 'Y': mp.Y, 'Z': mp.Z}
            for i in range(0, len(self.plasmon_symmetries), 2):
                axis = self.plasmon_symmetries[i].upper()
                phase = int(self.plasmon_symmetries[i + 1])
                symmetries_list.append(mp.Mirror(dir_map[axis], phase=phase))
            self.plasmon_symmetries = symmetries_list if symmetries_list else None

        from plasmol.classical.meep_verbosity import meep_io_context

        with meep_io_context(self.verbose):
            if self.has_plasmon_source:
                self.plasmon_source_object = MEEPSOURCE(
                    source_type=self.plasmon_source_type.lower().strip(),
                    source_center=self.plasmon_source_center,
                    source_size=self.plasmon_source_size,
                    component=self.plasmon_source_component.lower().strip(),
                    amplitude=self.plasmon_source_amplitude,
                    is_integrated=self.plasmon_source_is_integrated,
                    **{k: v for k, v in getattr(self, 'plasmon_source_additional_parameters', {}).items()}
                )

            if self.has_nanoparticle:
                self.nanoparticle_material = load_meep_material(self.nanoparticle_material)
                self.nanoparticle = mp.Sphere(
                    radius=self.nanoparticle_radius,
                    center=mp.Vector3(*self.nanoparticle_center),
                    material=self.nanoparticle_material
                )

        if self.has_images:
            self.images_args = ""
            for str in self.images_additional_parameters:
                self.images_args += f" {str}"

        if self.has_molecule:
            self.plasmol_molecule_position = mp.Vector3(*self.plasmol_molecule_position) 

    if self.has_molecule:
        self.molecule_atoms, self.molecule_coords, self.molecule_geometry_units = construct_geometry(self, self.molecule_geometry, self.molecule_geometry_units.lower())
        delattr(self, 'molecule_geometry')

        propagator_map = {
            "step": propagate_step,
            "magnus2": propagate_magnus2,
            "rk4": propagate_rk4
        }
        self.molecule_propagator = propagator_map[self.molecule_propagator_str]
        sig = inspect.signature(self.molecule_propagator)
        exclude_args = {'molecule', 'exc'}
        self.molecule_propagator_params = {name: getattr(self, name) for name in sig.parameters if name not in exclude_args}
        self.molecule_propagator_params['has_core_hole'] = True if getattr(self, 'has_core_hole', False) else False

        if not self.has_plasmon:
            time_values = np.arange(0, self.t_end + self.dt, self.dt)
            self.times = np.round(np.linspace(0, time_values[-1], int(len(time_values))), decimals=self.time_rounding_decimals)
            if not self.has_fourier:
                self.molecule_source_field = QUANTUMSOURCE(self).field
        else:
            # Meep field CSVs stamp (meep_time + dt) after at_beginning + at_every,
            # i.e. dt, 2*dt, ..., (n_steps+1)*dt with n_steps = round(t_end/dt).
            n_steps = max(1, round(self.t_end / self.dt))
            self.times = np.round(
                self.dt * np.arange(1, n_steps + 2),
                decimals=self.time_rounding_decimals,
            )

        if self.has_fourier:
            for dir in {"x", "y", "z"}:
                attr = f"field_e_{dir}_filepath"
                value = f"{dir}_dir/{self.field_e_filepath}"
                setattr(self, attr, value)
                attr = f"field_p_{dir}_filepath"
                value = f"{dir}_dir/{self.field_p_filepath}"
                setattr(self, attr, value)
                attr = f"spectra_e_{dir}_vs_p_{dir}_filepath"
                value = f"{dir}_dir/{self.spectra_e_vs_p_filepath}"
                setattr(self, attr, value)
