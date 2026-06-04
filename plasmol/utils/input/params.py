import os
import re 
import sys
import math
import json
import logging
import inspect
import meep as mp 
import numpy as np
from pathlib import Path
from pyscf.dft import libxc
from rich.table import Table
from typing import Tuple, Union
from rich.console import Console

from plasmol import constants
from plasmol.drivers import *
from plasmol.quantum.propagators import *
from plasmol.utils.input.struct import param_defs
from plasmol.quantum.electric_field import ELECTRICFIELD
from plasmol.classical.sources import MEEPSOURCE, walk_through_src_funcs

logger = logging.getLogger("main")

class PARAMS:
    def __init__(self, args):
        if hasattr(args, 'checkpoint') and args.checkpoint is not None:
            if os.path.exists(args.checkpoint):
                from plasmol.utils.checkpoint import resume_from_checkpoint
                logger.info(f"Checkpoint file {args.checkpoint} found.")
                params = resume_from_checkpoint(args)
                for attr, value in params.__dict__.items():
                    setattr(self, attr, value)
                return
            else:
                raise ValueError(f"Checkpoint file {args.checkpoint} not found, but resume from checkpoint flag ('-c') given.")
        else:
            self.resumed_from_checkpoint = False
        
        self.preparams = self._parse_input_file(args)
        self.input_file_path = self.preparams["args"]["input"]
        self.simulation_types = self.preparams["simulation_types"]
        self.xyz = ['x', 'y', 'z']
        self.custom_parameters = {}

        def _type_name(t):
            names = {
                bool: 'boolean',
                str: 'string',
                int: 'integer',
                float: 'float',
                list: 'list',
                dict: 'dictionary',
            }
            return names.get(t, t.__name__)

        # Initialize all section booleans to False
        boolean_names = set()
        for _, _, is_section_dict, bname, _, _, _, _, _ in param_defs:
            if is_section_dict and bname is not None:
                boolean_names.add(bname)
        for bname in boolean_names:
            setattr(self, bname, False)

        # Populate attributes from param_defs
        for attr, path, is_section_dict, boolean_name, default_value, section_condition, data_type, _, _ in param_defs:
            if section_condition is not None:
                if isinstance(section_condition, str):
                    if section_condition not in self.simulation_types:
                        continue
                elif isinstance(section_condition, list):
                    if not all(c in self.simulation_types for c in section_condition):
                        continue
            
            value = self._get_nested_value(self.preparams, path)
            
            if value is not None:
                if not isinstance(value, data_type):
                    raise ValueError(f"Invalid type for {attr}: expected {_type_name(data_type)}, got {_type_name(type(value))}.") 

            if path[0] == 'additional_parameters' and value is not None:
                self.custom_parameters[attr] = value
                self.has_custom = True

            if is_section_dict:
                has_section = value is not None
                setattr(self, boolean_name, has_section)
                if has_section:
                    setattr(self, attr, value)
            else:
                if value is not None:
                    setattr(self, attr, value)
                else:
                    # Apply default if the section is active (or no section boolean)
                    if boolean_name is None or getattr(self, boolean_name, False):
                        if default_value is not None:
                            logger.warning(f"Using default value for {attr}: {default_value}")
                            setattr(self, attr, default_value)

        self._attribute_checks()
        self._attribute_formation()
        delattr(self, 'preparams')
        logger.info("All parameters successfully parsed and validated.")

        self.verbose = getattr(args, 'verbose', 1)
        self.log = getattr(args, 'log', None)

    def _attribute_checks(self):
        """Perform checks and instantiations based on attributes."""
        # Settings (always required)
        if not hasattr(self, 'dt'):
            raise ValueError("Missing required parameter: 'dt' in settings.")
        if not hasattr(self, 't_end'):
            raise ValueError("Missing required parameter: 't_end' in settings.")
        if self.dt <= 0:
            raise RuntimeError("'dt' must be a positive value.")
        if self.t_end <= 0:
            raise RuntimeError("'t_end' must be a positive value.")
        if self.dt > self.t_end:
            raise RuntimeError("'dt' cannot be larger than 't_end'.")
        n_steps = round(self.t_end / self.dt)
        if abs(n_steps * self.dt - self.t_end) > 1e-9:
            raise RuntimeError("'t_end' must be a multiple of 'dt'")

        # Plasmon params
        if self.has_plasmon:
            # Plasmon simulation params
            if self.has_simulation:
                if self.plasmon_tolerance_field_e <= 0:
                    raise ValueError("'plasmon_tolerance_field_e' must be a positive value.")
                if hasattr(self, 'plasmon_cell_volume') and hasattr(self, 'plasmon_cell_length'):
                    logger.debug("'cell_volume' is specified; overriding 'cell_length'.")
                elif hasattr(self, 'plasmon_cell_length') and self.plasmon_cell_length <= 0:
                    raise ValueError("'plasmon_cell_length' must be a positive value.")
                if self.plasmon_pml_thickness <= 0:
                    raise ValueError("'plasmon_pml_thickness' must be a positive value.")
                if not hasattr(self, 'plasmon_symmetries'):
                    logger.warning("No 'symmetries' specified for plasmon simulation; operating without symmetries will cause longer simulation times.")
                    self.plasmon_symmetries = []
                else:
                    if len(self.plasmon_symmetries) % 2 != 0:
                        raise ValueError(f"Invalid plasmon symmetry '{self.plasmon_symmetries}'; list length must be even (pairs of axis and value)")
                    for i, sym in enumerate(self.plasmon_symmetries):
                        if i % 2 == 0:
                            if not (isinstance(sym, str) and sym.lower() in self.xyz):
                                raise ValueError(f"Invalid plasmon symmetry '{self.plasmon_symmetries}'; even indices must be 'x', 'y', or 'z' (case-insensitive)")
                        else:
                            if not (isinstance(sym, int) and sym in [1, -1]):
                                raise ValueError(f"Invalid plasmon symmetry '{self.plasmon_symmetries}'; odd indices must be 1 or -1 (integers)")
                if self.plasmon_surrounding_material_index < 1.0:
                    raise ValueError("'surrounding_material_index' must be >= 1.0 (vacuum).")
                elif self.plasmon_surrounding_material_index == 1.0:
                    logger.debug("For 'surrounding_material_index' value of 1.0 (vacuum) is being used.")
                elif self.plasmon_surrounding_material_index > 5.0:
                    logger.warning("'surrounding_material_index' is unusually high; please verify this value.")

            # Plasmon source params
            if self.has_plasmon_source:
                for attr in ['plasmon_source_type', 'plasmon_source_center', 'plasmon_source_size', 'plasmon_source_component', 'plasmon_source_additional_parameters']:
                    if not hasattr(self, attr):
                        pretty = attr.removeprefix("plasmon_source_")
                        raise ValueError(f"Source requires '{pretty}' attribute.")
                for loc in self.plasmon_source_center:
                    if not isinstance(loc, (int, float)):
                        raise ValueError(f"Invalid plasmon source center '{loc}'; must be a number.")
                for loc in self.plasmon_source_size:
                    if not isinstance(loc, (int, float)):
                        raise ValueError(f"Invalid plasmon source size '{loc}'; must be a number.")
                if self.plasmon_source_component not in self.xyz:
                    raise ValueError(f"Invalid plasmon source component '{self.plasmon_source_component}'; must be 'x', 'y', or 'z'.")
                if getattr(self, "plasmon_source_additional_parameters", None) is not None:
                    if 'frequency' not in self.plasmon_source_additional_parameters and 'wavelength' not in self.plasmon_source_additional_parameters and not self.plasmon_source_type == 'custom':
                        raise ValueError(f"Either 'frequency' or 'wavelength' must be provided in 'plasmon_source_additional_parameters'.")
                elif self.plasmon_source_frequency is None and self.plasmon_source_wavelength is None and self.plasmon_source_type != 'custom':
                    raise ValueError(f"Either 'frequency' or 'wavelength' must be provided for {self.plasmon_source_type} source.")

                if self.plasmon_source_type == 'custom':
                    if not hasattr(self, 'plasmon_source_additional_parameters') or 'src_func' not in self.plasmon_source_additional_parameters:
                        raise ValueError(f"Custom source requires 'src_func' in 'plasmon_source_additional_parameters' attribute.")
                    try:
                        walk_through_src_funcs(self.plasmon_source_additional_parameters['src_func'])
                    except ValueError as e:
                        raise ValueError(f"Error occurred while processing custom source function: {e}")

            else:
                logger.info('No source chosen for simulation. Continuing without it.')

            # Nanoparticle params
            if self.has_nanoparticle:
                for attr in ['nanoparticle_material', 'nanoparticle_radius', 'nanoparticle_center']:
                    if not hasattr(self, attr):
                        pretty = attr.removeprefix("nanoparticle_")
                        raise ValueError(f"Nanoparticle requires '{pretty}' attribute.")
                for loc in self.nanoparticle_center:
                    if not isinstance(loc, (int, float)):
                        raise ValueError(f"Invalid nanoparticle center '{loc}'; must be a number.")

            # Images params
            if self.has_images:
                if hasattr(self, 'images_additional_parameters'):
                    for loc in self.images_additional_parameters:
                        if not isinstance(loc, str):
                            raise ValueError(f"Invalid image additional parameter '{loc}'; must be a string.")
                if not hasattr(self, 'images_timesteps_between'):
                    raise ValueError("Images requires 'timesteps_between' attribute.")
            else:
                logger.info('No picture output chosen for simulation. Continuing without it.')

            # Molecule position param
            if self.has_molecule_position:
                if not self.has_molecule:
                    raise ValueError(f"Molecule position properly specified in 'plasmon' section but without 'molecule' section present in input file.")
                for loc in self.plasmol_molecule_position:
                    if not isinstance(loc, (int, float)):
                        raise ValueError(f"Invalid molecule position '{loc}'; must be a number.")
                if len(self.plasmol_molecule_position) != 3:
                    raise ValueError("Molecule position must be an array of three numbers [x, y, z].")
                if self.has_nanoparticle:
                    self.plasmon_resolution = round(self.plasmon_courant / (self.dt / constants.convertTimeMeep2Atomic))
                    self.plasmon_pixel_length = 1/self.plasmon_resolution
                    p = np.asarray(self.plasmol_molecule_position, dtype=float)
                    c = np.asarray(self.nanoparticle_center, dtype=float)
                    if p.shape != (3,) or c.shape != (3,):
                        raise ValueError("Point and center must be 3D coordinates (x, y, z)")
                    d = np.linalg.norm(p - c)
                    distance = abs(d - self.nanoparticle_radius)
                    print(d)
                    if self.plasmon_pixel_length > distance:
                        raise ValueError(f"Molecule position is too close to nanoparticle surface (dist = {distance:.6f} μm). Minimum distance required: {self.plasmon_pixel_length:.6f} μm.")
                    if self.nanoparticle_radius > d:
                        raise ValueError(f"Molecule position is inside the nanoparticle (dist from NP center = {d:.6f} μm). Minimum distance required: {self.nanoparticle_radius + self.plasmon_pixel_length:.6f} μm.")

        # Molecule params
        if self.has_molecule:
            if self.has_plasmon:
                if not hasattr(self, 'plasmol_molecule_position'):
                    raise RuntimeError("No 'plasmol_molecule_position' object found in 'plasmon' section, but quantum (molecule) is present. Please specify the 'plasmol_molecule_position' parameters in the 'plasmon' section.")
            if self.has_comparison:
                if hasattr(self, 'molecule_basis') or hasattr(self, 'molecule_xc'):
                    logger.info("Comparison modifier selected; ignoring singular basis set and xc.")
                else:
                    # Setting singular basis and xc so it can pass the following checks. These will be ignored anyway.
                    self.molecule_basis = '6-31g'
                    self.molecule_xc = 'pbe0'
            for attr in ['molecule_geometry', 'molecule_geometry_units', 'molecule_basis', 'molecule_charge', 'molecule_spin']:
                if not hasattr(self, attr) or getattr(self, attr) == []:
                    pretty = attr.removeprefix("molecule_")
                    raise ValueError(f"Molecule requires '{pretty}' attribute.")
            lrc_parameters = [self.molecule_lrc_parameter] if hasattr(self, 'molecule_lrc_parameter') else []
            self._check_xc(self.molecule_xc, *lrc_parameters)
            if type(self.molecule_geometry) == str:
                path = Path(self.molecule_geometry)
                if not path.exists():
                    raise ValueError(f"Geometry file not found: {path}")
                if not path.suffix.lower() == '.xyz':
                    raise ValueError("String input must be a path to a .xyz file.")
            else:
                for loc in self.molecule_geometry:
                    if not isinstance(loc, dict):
                        raise ValueError(f"Invalid molecule position '{loc}'; must be a dictionary (ex. {'atom': 'O', 'coord': [0.0, 0.0, -0.1302052882]}).")
            if hasattr(self, 'molecule_propagator_str'):
                self.molecule_propagator_str = self.molecule_propagator_str.lower()
                if self.molecule_propagator_str not in ['step', 'rk4', 'magnus2']:
                    raise ValueError(f"Unsupported propagator: {self.molecule_propagator_str}. Acceptable: step, rk4, magnus2.")
            if not self.molecule_geometry_units in ['angstrom', 'bohr']:
                raise ValueError(f"Invalid 'molecule_geometry_units': '{self.molecule_geometry_units}'. Must be 'angstrom' or 'bohr'.")

            # Source params (molecule section)
            if self.has_plasmon_source and self.has_molecule_source:
                raise ValueError("Source found in both plasmon and molecule sections. Please specify only one.")
            elif self.has_molecule_source:
                for attr in ['molecule_source_intensity', 'molecule_source_peak_time', 'molecule_source_width_steps']:
                    if not hasattr(self, attr):
                        pretty = attr.removeprefix("molecule_source_")
                        raise ValueError(f"Molecule source requires '{pretty}' attribute.")
                if self.has_fourier:
                    for attr in ['molecule_source_type', 'molecule_source_component']:
                        if hasattr(self, attr):
                            pretty = attr.removeprefix("molecule_source_")
                            logger.warning(f"Molecule source '{pretty}' attribute being ignored because Fourier modifier is enabled.")
                    setattr(self, 'molecule_source_type', 'kick')
                    setattr(self, 'molecule_source_component', 'z')
                if hasattr(self, 'molecule_source_peak_time'):
                    value = getattr(self, 'molecule_source_peak_time')
                    if value is not None and value < 0:
                        raise ValueError(f"Molecule source peak_time must be a positive value.")
                for attr in ['molecule_source_intensity', 'molecule_source_width_steps']:
                    if hasattr(self, attr):
                        value = getattr(self, attr)
                        if value is not None and value <= 0:
                            pretty = attr.removeprefix("molecule_source_")
                            raise ValueError(f"Molecule source '{pretty}' must be a positive value.")
                if not hasattr(self, "molecule_source_additional_parameters") and self.molecule_source_type == 'pulse':
                    pretty = attr.removeprefix("molecule_source_")
                    raise ValueError(f"Source requires '{pretty}' attribute.")

                if self.molecule_source_type.lower() == 'pulse':
                    msap = getattr(self, 'molecule_source_additional_parameters', {})
                    if 'wavelength' not in msap and 'frequency' not in msap:
                        raise ValueError("Molecule source of type 'pulse' requires 'wavelength' or 'frequency' attribute.")
                if self.molecule_source_type.lower() not in ['pulse', 'kick']:
                    raise ValueError(f"Molecule source must be of type 'pulse' or 'kick' and not '{self.molecule_source_type}'.")
                if self.molecule_source_component not in self.xyz:
                    raise ValueError(f"Molecule source component must be one of {self.xyz}.")
                if hasattr(self, 'molecule_source_additional_parameters') and self.molecule_source_additional_parameters is not None:
                    for attr in self.molecule_source_additional_parameters:
                        value = self.molecule_source_additional_parameters.get(attr)
                        if value is not None and value <= 0:
                            raise ValueError(f"Molecule source '{pretty}' must be a positive value.")
                    if 'frequency' not in self.molecule_source_additional_parameters and 'wavelength' not in self.molecule_source_additional_parameters:
                        raise ValueError(f"Either 'frequency' or 'wavelength' must be provided in 'molecule_source_additional_parameters'.")

            # Fourier params
            if self.has_fourier:
                if self.has_plasmon:
                    raise ValueError(f"Fourier modifier cannot be used with plasmon modifier.")
                logger.info("Fourier modifier selected; running three simulations for Fourier analysis along each axis.")
                if not hasattr(self, 'fourier_spectrum_filepath') or getattr(self, 'fourier_spectrum_filepath') in ['']:
                    raise ValueError(f"Fourier modifier requires spectrum_filepath attribute.")
                if self.fourier_min_ev < 0:
                    raise ValueError(f"Fourier modifier 'min_ev' must be a non-negative value.")
                if self.fourier_max_ev < 0:
                    raise ValueError(f"Fourier modifier 'max_ev' must be a non-negative value.")
                if self.fourier_max_ev <= self.fourier_min_ev:
                    raise ValueError(f"Fourier modifier 'max_ev' must be greater than 'min_ev'.")
                if not hasattr(self, 'fourier_gamma'):
                    if not self.has_broadening:
                        raise ValueError(f"Fourier modifier requires gamma attribute.")
                    else:
                        self.fourier_gamma = 0
                if self.fourier_gamma < 0:
                    raise ValueError("Fourier modifier 'gamma' must be a non-negative value.")
                if hasattr(self, 'fourier_field_p_damping_gamma'):
                    self.fourier_damp = True
                    logger.info("Damping modifier selected; preparing to apply damping to time-domain signals. See documentation for details.")
                    if self.fourier_field_p_damping_gamma <= 0:
                        raise ValueError("Damping 'gamma' must be a positive value.")
                else:
                    self.fourier_damp = False

            # Lopata Broadening params
            if self.has_broadening:
                logger.info("Broadening modifier selected; applying Lopata broadening to spectra.")
                if not hasattr(self, 'broadening_type'):
                    raise ValueError("Broadening modifier requires 'type' attribute (either 'static' or 'dynamic').")
                if self.broadening_type.lower() not in ['static', 'dynamic']:
                    raise ValueError("Broadening 'type' must be 'static' or 'dynamic'.")
                if hasattr(self, 'broadening_gam0') and self.broadening_gam0 <= 0:
                    raise ValueError("Broadening 'gam0' must be a positive value.")
                if hasattr(self, 'broadening_xi')  and self.broadening_xi < 0:
                    raise ValueError("Broadening 'xi' must be a non-negative value.")
                if hasattr(self, 'broadening_eps0') and self.broadening_eps0 < 0:
                        raise ValueError("Broadening 'eps0' must be a non-negative value.") 
                if hasattr(self, 'broadening_clamp') and self.broadening_clamp <= 0:
                    raise ValueError("Broadening 'clamp' must be a positive value.")

            # Comparison mode params
            if self.has_comparison:
                logger.info("Comparison modifier selected; preparing to run additional simulations for comparison to molecule results.")
                if self.has_plasmon:
                    raise ValueError("Comparison mode is not supported with plasmon simulations. Please run with only molecule simulations.")
                if self.has_fourier:
                    raise ValueError("Comparison mode is not supported with fourier simulations. Please run with only molecule simulations.")
                if not hasattr(self, 'comparison_bases') or not hasattr(self, 'comparison_xcs'):
                    raise ValueError("Comparison mode requires both 'bases' and 'xcs' lists. See documentation for details.")
                for loc in self.comparison_bases:
                    if not isinstance(loc, str):
                        raise ValueError(f"Invalid comparison basis '{loc}'; must be a string.")
                for loc in self.comparison_xcs:
                    if not isinstance(loc, str):
                        raise ValueError(f"Invalid comparison xcs '{loc}'; must be a string.")
                # TODO: Implement LRC to comparison
                if hasattr(self, 'comparison_lrc_parameters'):
                    if not isinstance(self.comparison_lrc_parameters, dict):
                        raise ValueError("Comparison 'lrc_parameters' must be a dictionary.")
                    for loc in self.comparison_lrc_parameters:
                        if not isinstance(self.comparison_lrc_parameters[loc], (int, float)):
                            raise ValueError(f"Invalid comparison 'lrc_parameters' for '{loc}'; must be a number.")
                        if loc not in self.comparison_xcs:
                            raise ValueError(f"Comparison 'lrc_parameters' for '{loc}' is not in the list of xcs.")
                for xc in self.comparison_xcs:
                    omega = self.comparison_lrc_parameters.get(xc, None)
                    self._check_xc(xc, omega)
                for loc in ['comparison_num_virtual', 'comparison_num_occupied', 'comparison_index_min', 'comparison_index_max']:
                    if hasattr(self, loc):
                        if getattr(self, loc) < 1: 
                            pretty = loc.removeprefix("molecule_source_")
                            raise ValueError(f"Comparison '{pretty}' must be at least 1.")

        # Checkpointing params
        if self.has_checkpoint:
            logger.info("Checkpointing selected; preparing to save and load checkpoints during simulation.")
            if self.has_plasmon:
                raise ValueError("Checkpointing not supported on simulations including plasmons.")
            if not hasattr(self, 'checkpoint_filepath') or self.checkpoint_filepath in ['']:
                raise ValueError("Checkpointing requires 'filepath' attribute for checkpoint file.")
            if not hasattr(self, 'checkpoint_frequency_steps') and not hasattr(self, 'checkpoint_frequency_time'):
                raise ValueError("Checkpointing requires 'frequency_steps' or 'frequency_time' attribute for snapshot frequency.")
            if hasattr(self, 'checkpoint_frequency_steps') and hasattr(self, 'checkpoint_frequency_time'):
                raise ValueError("Checkpointing requires either 'frequency_steps' or 'frequency_time' attribute, not both.")
            if hasattr(self, 'checkpoint_frequency_steps') and self.checkpoint_frequency_steps <= 0:
                raise ValueError("Checkpointing 'frequency_steps' must be a positive value.")
            if hasattr(self, 'checkpoint_frequency_time') and self.checkpoint_frequency_time <= 0:
                raise ValueError("Checkpointing 'frequency_time' must be a positive value.")
            if hasattr(self, 'checkpoint_frequency_time') and self.checkpoint_frequency_time > self.t_end:
                logger.warning(f"Checkpointing 'frequency_time' ({self.checkpoint_frequency_time}) is greater than simulation end time ({self.t_end}). Will only save checkpoint at simulation end.")
            if hasattr(self, 'checkpoint_frequency_time'):
                n_steps = round(self.checkpoint_frequency_time / self.dt)
                reconstructed = n_steps * self.dt
                if not math.isclose(reconstructed, self.checkpoint_frequency_time, rel_tol=1e-9, abs_tol=1e-12):
                    remainder = self.checkpoint_frequency_time % self.dt
                    raise ValueError(
                        f"Checkpointing 'frequency_time' ({self.checkpoint_frequency_time}) must be a multiple of "
                        f"the time step ({self.dt}), but got remainder = {remainder}"
                    )
                self.checkpoint_frequency_steps = n_steps
            if not self.has_molecule:
                raise ValueError("Checkpointing is only supported with molecule simulations.")

        # Files
        for file in ['field_e_filepath', 'field_p_filepath']:
            if hasattr(self, file):
                value = getattr(self, file)
                if not isinstance(value, str) or value in ['']:
                    raise ValueError(f"Filepath for '{file}' must be a non-empty string.")
                
        # Misc/Add'l Parameters
        if self.has_custom:
            logger.info(f"Additional parameters specified: {list(self.custom_parameters.keys())}.")

    def _attribute_formation(self):
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
                dir_map = {'X': mp.X, 'Y': mp.Y, 'Z': mp.Z}
                for i in range(0, len(self.plasmon_symmetries), 2):
                    axis = self.plasmon_symmetries[i].upper()
                    phase = int(self.plasmon_symmetries[i + 1])
                    symmetries_list.append(mp.Mirror(dir_map[axis], phase=phase))
                self.plasmon_symmetries = symmetries_list if symmetries_list else None

            if self.has_plasmon_source:
                self.plasmon_source_object = MEEPSOURCE(
                    source_type=self.plasmon_source_type.lower().strip(),
                    source_center=self.plasmon_source_center,
                    source_size=self.plasmon_source_size,
                    component=self.plasmon_source_component.lower().strip(),
                    amplitude=self.plasmon_source_amplitude,
                    is_integrated=self.plasmon_source_is_integrated,
                    **{k: v for k, v in self.plasmon_source_additional_parameters.items()}
                )

            if self.has_nanoparticle:
                self.nanoparticle_material = self._load_meep_material(self.nanoparticle_material)
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
            self.molecule_atoms, self.molecule_coords, self.molecule_geometry_units = self._construct_geometry(self.molecule_geometry, self.molecule_geometry_units.lower())
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
            
            if not self.has_plasmon:
                time_values = np.arange(0, self.t_end + self.dt, self.dt)
                self.times = np.linspace(0, time_values[-1], int(len(time_values)))
                self.molecule_source_field = ELECTRICFIELD(self).field

            if self.has_broadening:
                del self.broadening_dict["type"]

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
            
    def _get_nested_value(self, d, path):
        """Helper to get nested dict value safely."""
        cur = d
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return None
        return cur

    def _load_meep_material(self, material_str):
        import importlib
        materials = importlib.import_module("meep.materials")
        try:
            return getattr(materials, material_str)
        except AttributeError as e:
            raise ImportError(
                f"Material '{material_str}' not found in meep.materials. "
                f"Check spelling/case or available materials."
            ) from e

    def _check_xc(self, func_name: str, omega: float = None):
        try:
            orig_omega, _, _ = libxc.rsh_coeff(func_name)
            if omega is not None and orig_omega == 0:
                raise ValueError(f"Functional '{func_name}' is not a range-separated hybrid (RSH) so lrc_parameter will be ignored.")
            return True
        except Exception as e:
            raise ValueError(f"Error checking xc functional '{func_name}': {e}")

    def _construct_geometry(self, geometry, units):
        """
        Post-process molecule geometry:
        - Accepts either:
            1. List of dicts: [{"atom": "O", "coord": [x,y,z]}, ...]
            2. String path to a .xyz file
        - Validates input
        - Converts to Bohr units
        - Builds the exact coords string expected by the simulator
        """
        atoms = []
        coords_bohr = {}

        if isinstance(geometry, str):
            path = Path(geometry)
            
            # Parse XYZ file
            with open(path) as f:
                lines = [line.strip() for line in f if line.strip()]

            # First line: total number of atoms (optional)
            # Second line: molecule name or comment (optional)
            # All other lines: element symbol or atomic number, x, y, and z coordinates, separated by spaces, tabs, or commas
            start_line = None
            num_atoms = None
            for current_line, line in enumerate(lines):
                items = line.split()
                if len(items) < 4:
                    if len(items) == 1 and items[0].isdigit():
                        num_atoms = int(items[0])
                    continue
                else:
                    for item in items:
                        item = item.replace('.', '').replace(',', '')
                        if not item.isdigit():
                            continue
                    start_line = current_line
            
            if start_line is None:
                raise ValueError("Invalid XYZ file format: no valid atom lines found.")
            if num_atoms is None:
                num_atoms = 0
                for i in range(start_line, len(lines)):
                    items = lines[i].split()
                    if len(items) == 4:
                        num_atoms += 1
            
            geometry = []
            for i in range(2, 2 + num_atoms):
                parts = lines[i].split()
                atom = parts[0]
                coord = [float(x) for x in parts[1:4]]
                geometry.append({"atom": atom, "coord": coord})

        if not isinstance(geometry, list):
            raise ValueError("geometry must be a list of dicts or a path to a .xyz file.")

        for idx, entry in enumerate(geometry, start=1):
            if not isinstance(entry, dict) or 'atom' not in entry or 'coord' not in entry:
                raise ValueError("Each geometry entry must be a dict with 'atom' (str) and 'coord' (list of 3 floats).")

            atom = entry['atom']
            coord = entry['coord']

            if len(coord) != 3:
                raise ValueError(f"Coords for atom {atom} must have exactly 3 numbers.")

            atoms.append(atom)
            label = f"{atom}{idx}"
            coords_bohr[label] = np.array(coord, dtype=float)

        # Convert to Bohr if input was in Ångstroms
        if units.lower().startswith('angstrom'):
            factor = 1.8897259886
            coords_bohr = {label: xyz * factor for label, xyz in coords_bohr.items()}
            units = "bohr"

        # Build the exact string format PySCF wants
        coords_str = ""
        for i, atom in enumerate(atoms):
            x, y, z = coords_bohr[f"{atom}{i+1}"]
            coords_str += f" {atom} {x} {y} {z}"
            if i < len(atoms) - 1:
                coords_str += ";"

        return atoms, coords_str.strip(), units

    def _parse_input_file(self, args):
        """
        Prepares parameters from the input JSON file and CLI args.

        Loads the JSON file, performs necessary post-processing (e.g., for quantum geometry),
        and determines the simulation type based on present sections.

        Args:
            args: Command-line arguments containing 'input' (path to JSON file).

        Returns:
            dict: Preparams dictionary with 'settings', 'plasmon', 'molecule', 'args', and 'simulations'.

        Raises:
            RuntimeError: If required parameters are missing or invalid configuration.
        """
        input_path = args.input
        with open(input_path, 'r') as f:
            # Removes comments
            content = ''.join(re.sub(r"(#|--|%|//)(.*)$", '', line) for line in f if not line.strip().startswith(('#', '--', '%', '//')))
            params = json.loads(content)

        # Extract main sections; they are optional except settings
        settings_params = params.get('settings', {})
        plasmon_params = params.get('plasmon')
        molecule_params = params.get('molecule')
        files_params = params.get('files')
        addl_params = params.get('additional_parameters')

        # ---- Determine simulation type + validation ----
        simulation_types = []
        if molecule_params:
            simulation_types.append('molecule')
        if plasmon_params:
            simulation_types.append('plasmon')

        if not simulation_types:
            raise RuntimeError("The minimum required parameters were not given. Please check guidelines for information on minimal requirements.")

        # Logging for single-simulation cases (same behaviour as before)
        if len(simulation_types) == 1:
            if simulation_types[0] == 'molecule':
                logger.info("Only 'molecule' parameters given. Running RT-TDDFT simulation only.")
            else:
                logger.info("Only 'plasmon' parameters given. Running MEEP simulation only.")

        # ---- Build preparams ----
        cli_args = {k: v for k, v in vars(args).items() if v is not None}

        preparams = {
            "settings": settings_params,
            "simulation_types": simulation_types,
            "args": cli_args,
        }
        if plasmon_params:
            preparams["plasmon"] = plasmon_params
        if molecule_params:
            preparams["molecule"] = molecule_params
        if files_params:
            preparams["files"] = files_params
        if addl_params:
            preparams["additional_parameters"] = addl_params

        return preparams

    def __getstate__(self):
        """Return state for pickling (e.g. multiprocessing in fdtd_response driver).
        Removes unpicklable Meep/SWIG objects so the rest of params can be safely pickled.
        """
        state = self.__dict__.copy()
        for attr in ['plasmon_source_object', 'nanoparticle']:
            state.pop(attr, None)
        return state

    def __setstate__(self, state):
        """Restore state after unpickling. Meep objects are intentionally absent;
        the calling code (e.g. fdtd_response) is responsible for recreating them.
        """
        self.__dict__.update(state)
    
    @classmethod
    def describe_parameters(cls):
        """Print beautiful table of ALL input parameters (used by --describe)."""
        console = Console()
        table = Table(
            title="PlasMol — All Parameters",
            show_lines=True,
            title_style="bold magenta"
        )
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta", justify="center")
        table.add_column("Default", style="yellow", justify="center")
        table.add_column("Description", style="green")
        table.add_column("Units", style="rosy_brown")

        def _type_name(t):
            # Reuse / extend your existing helper
            names = {
                bool: 'boolean',
                str: 'string',
                int: 'integer',
                float: 'float',
                list: 'list',
                dict: 'dictionary',
            }
            if isinstance(t, tuple):                     # (int, float) -> "float"
                return " or ".join(names.get(x, x.__name__) for x in t)
            return names.get(t, getattr(t, '__name__', str(t)))

        for entry in param_defs:
            (attr, _, is_section_dict, _, default_value,
             _, data_type, description, units) = entry

            if is_section_dict:          # skip container dicts (plasmon_dict, etc.)
                continue

            type_str = _type_name(data_type)

            # Default display (matches your example)
            default_str = "—" if default_value is None else str(default_value)
            units_str = "—" if units is None else str(units)

            table.add_row(
                attr,
                type_str,
                default_str,
                description,
                units_str or "—"
            )

        console.print(table)