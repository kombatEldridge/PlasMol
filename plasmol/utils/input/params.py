# utils/input/params.py
import logging
import math
import sys
import numpy as np
import re 
import json 

from plasmol import constants

logger = logging.getLogger("main")
class PARAMS:
    """
    Container for simulation parameters given from input files and cli inputs.
    """
    def __init__(self, args):
        self.preparams = parseInputFile(args)
        self.restart = self.preparams["args"]["restart"]
        self.do_nothing = self.preparams["args"]["do_nothing"]
        self.simulation_types = self.preparams["simulation_types"]

        # Define a large, extensible list of parameter definitions.
        # Each entry is a tuple: (attribute_name, path_as_list, default_value, section_condition)
        # - attribute_name: str, the name to set as self.<attribute_name>
        # - path_as_list: list of str, the nested keys to access in parsed (e.g., ['settings', 'dt'])
        # - default_value: any, value if not found (use None for required params)
        # - section_condition: str or None, only set if self.type in ['plasmon', 'molecule'] matches or None for always
        param_defs = [
            # Settings (always required)
            ('dt', ['settings', 'dt'], None, None),
            ('t_end', ['settings', 't_end'], None, None),

            # Plasmon params
            ('plasmon', ['plasmon'], None, 'plasmon'),
            ('plasmon_tolerance_efield', ['plasmon', 'simulation', "tolerance_efield"], 1e-12, 'plasmon'),
            ('plasmon_cell_length', ['plasmon', 'simulation', "cell_length"], 0.1, 'plasmon'),
            ('plasmon_cell_volume', ['plasmon', 'simulation', "cell_volume"], None, 'plasmon'),
            ('plasmon_pml_thickness', ['plasmon', 'simulation', "pml_thickness"], 0.01, 'plasmon'),
            ('plasmon_symmetries', ['plasmon', 'simulation', 'symmetries'], None, 'plasmon'),
            ('plasmon_surrounding_material_index', ['plasmon', 'simulation', "surrounding_material_index"], 1.33, 'plasmon'),

            # Plasmon source params
            ('plasmon_source', ['plasmon', 'source'], None, 'plasmon'),
            ('plasmon_source_type', ['plasmon', 'source', 'type'], None, 'plasmon'),
            ('plasmon_source_center', ['plasmon', 'source', 'center'], None, 'plasmon'),
            ('plasmon_source_size', ['plasmon', 'source', 'size'], None, 'plasmon'),
            ('plasmon_source_component', ['plasmon', 'source', 'component'], None, 'plasmon'),
            ('plasmon_source_amplitude', ['plasmon', 'source', 'amplitude'], 1, 'plasmon'),
            ('plasmon_source_is_integrated', ['plasmon', 'source', 'is_integrated'], True, 'plasmon'),
            ('plasmon_source_additional_parameters', ['plasmon', 'source', 'additional_parameters'], None, 'plasmon'),

            # Nanoparticle params
            ('nanoparticle', ['plasmon', 'nanoparticle'], None, 'plasmon'),
            ('nanoparticle_material', ['plasmon', 'nanoparticle', 'material'], None, 'plasmon'),
            ('nanoparticle_radius', ['plasmon', 'nanoparticle', 'radius'], None, 'plasmon'),
            ('nanoparticle_center', ['plasmon', 'nanoparticle', 'center'], None, 'plasmon'),

            # Images params
            ('images', ['plasmon', 'images'], None, 'plasmon'),
            ('images_timesteps_between', ['plasmon', 'images', 'timesteps_between'], None, 'plasmon'),
            ('images_additional_parameters', ['plasmon', 'images', 'additional_parameters'], None, 'plasmon'),
            ('images_dir_name', ['plasmon', 'images', 'dir_name'], None, 'plasmon'),

            # Molecule position param
            ('plasmol_molecule_position', ['plasmon', 'molecule_position'], [0,0,0], ['plasmon', 'molecule']),

            # Molecule params
            ('molecule', ['molecule'], None, 'molecule'),
            ('molecule_geometry', ['molecule', 'geometry'], None, 'molecule'),
            ('molecule_geometry_units', ['molecule', 'geometry_units'], None, 'molecule'),
            ('molecule_basis', ['molecule', 'basis'], None, 'molecule'),
            ('molecule_charge', ['molecule', 'charge'], None, 'molecule'),
            ('molecule_spin', ['molecule', 'spin'], None, 'molecule'),
            ('molecule_xc', ['molecule', 'xc'], None, 'molecule'),
            ('molecule_lrc_parameter', ['molecule', 'lrc_parameter'], None, 'molecule'),
            ('molecule_custom_xc', ['molecule', 'custom_xc'], None, 'molecule'),
            ('molecule_propagator', ['molecule', 'propagator', "type"], None, 'molecule'),
            ('molecule_pc_convergence', ['molecule', 'propagator', "pc_convergence"], 1e-12, 'molecule'),
            ('molecule_max_iterations', ['molecule', 'propagator', "max_iterations"], 200, 'molecule'),
            ('molecule_hermiticity_tolerance', ['molecule', 'hermiticity_tolerance'], 1e-12, 'molecule'),

            # Source params (molecule section)
            ('molecule_source', ['molecule', 'source'], None, 'molecule'),
            ('molecule_source_type', ['molecule', 'source', 'type'], None, 'molecule'),
            ('molecule_source_intensity_au', ['molecule', 'source', 'intensity_au'], None, 'molecule'),
            ('molecule_source_peak_time_au', ['molecule', 'source', 'peak_time_au'], None, 'molecule'),
            ('molecule_source_width_steps', ['molecule', 'source', 'width_steps'], None, 'molecule'),
            ('molecule_source_wavelength_nm', ['molecule', 'source', 'wavelength_nm'], None, 'molecule'),
            ('molecule_source_dir', ['molecule', 'source', 'dir'], None, 'molecule'),

            # Fourier params runs three sims at once, one per axis
            ('fourier', ['molecule', 'modifiers', 'fourier'], None, 'molecule'),
            ('fourier_gamma', ['molecule', 'modifiers', 'fourier', 'gamma'], None, 'molecule'),
            ('fourier_pfield_filepath', ['molecule', 'modifiers', 'fourier', 'fourier_pfield_filepath'], None, 'molecule'),
            ('fourier_spectrum_filepath', ['molecule', 'modifiers', 'fourier', 'fourier_spectrum_filepath'], None, 'molecule'),

            # Lopata Broadening params
            ('broadening', ['molecule', 'modifiers', 'broadening'], None, 'molecule'),
            ('broadening_type', ['molecule', 'modifiers', 'broadening', "type"], None, 'molecule'),
            ('broadening_gam0', ['molecule', 'modifiers', 'broadening', "gam0"], None, 'molecule'),
            ('broadening_xi', ['molecule', 'modifiers', 'broadening', "xi"], None, 'molecule'),
            ('broadening_eps0', ['molecule', 'modifiers', 'broadening', "eps0"], None, 'molecule'),
            ('broadening_clamp', ['molecule', 'modifiers', 'broadening', "clamp"], None, 'molecule'),

            # Comparison mode params
            ('comparison', ['molecule', 'modifiers', 'comparison'], None, 'molecule'),
            ('comparison_bases', ['molecule', 'modifiers', 'comparison', 'bases'], None, 'molecule'),
            ('comparison_xcs', ['molecule', 'modifiers', 'comparison', 'xcs'], None, 'molecule'),
            ('comparison_num_virtual', ['molecule', 'modifiers', 'comparison', 'num_virtual'], 3, 'molecule'),
            ('comparison_num_occupied', ['molecule', 'modifiers', 'comparison', 'num_occupied'], 3, 'molecule'),
            ('comparison_y_min', ['molecule', 'modifiers', 'comparison', 'y_min'], -1, 'molecule'),
            ('comparison_y_max', ['molecule', 'modifiers', 'comparison', 'y_max'], 1, 'molecule'),

            # Dampening mode params
            ('dampening', ['molecule', 'modifiers', 'dampening'], None, 'molecule'),
            ('dampening_gamma', ['molecule', 'modifiers', 'dampening', 'gamma'], None, 'molecule'),

            # Checkpointing params
            ('checkpoint', ['molecule', 'files', 'checkpoint'], None, 'molecule'),
            ('checkpoint_filepath', ['molecule', 'files', 'checkpoint', 'filepath'], None, 'molecule'),
            ('checkpoint_snapshot_frequency', ['molecule', 'files', 'checkpoint', 'frequency'], None, 'molecule'),

            # Files
            ('field_e_filepath', ['molecule', 'files', 'field_e_filepath'], 'eField.csv', 'molecule'),
            ('field_p_filepath', ['molecule', 'files', 'field_p_filepath'], 'pField.csv', 'molecule'),
            ('field_e_vs_p_filepath', ['molecule', 'files', 'field_e_vs_p_filepath'], None, 'molecule'),
        ]

        # Populate attributes from param_defs
        for attr, path, default, condition in param_defs:
            if condition:
                ok = (condition in self.simulation_types) if isinstance(condition, str) else all(c in self.simulation_types for c in condition)
                if not ok:
                    continue
            value = self._get_nested_value(self.preparams, path, default)
            if value is not None:
                setattr(self, attr, value)

        self._attribute_checks()
        print("All parameters successfully parsed and validated.")
        for key, value in vars(self).items():
            print(f"\t\t{key}: {value}")
        sys.exit(0)
        # self._attribute_formation()
        delattr(self, 'parsed')  # Clean up


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
        if math.isclose(self.t_end % self.dt, 0.0, abs_tol=1e-10):
            raise RuntimeError("'t_end' must be a multiple of 'dt'.")

        # Plasmon params
        if hasattr(self, 'plasmon'):
            if self.plasmon_tolerance_efield <= 0:
                raise ValueError("'plasmon_tolerance_efield' must be a positive value.")
            
            if self.plasmon_cell_length <= 0:
                raise ValueError("'plasmon_cell_length' must be a positive value.")
            
            # TODO: Add overriding to _attribute_formation
            if hasattr(self, 'plasmon_cell_volume'):
                logger.debug("'cell_volume' is specified; overriding 'cell_length'.")

            if self.plasmon_pml_thickness <= 0:
                raise ValueError("'plasmon_pml_thickness' must be a positive value.")
            
            if not hasattr(self, 'plasmon_symmetries'):
                logger.warning("No 'symmetries' specified for plasmon simulation; operating without symmetries will cause longer simulation times.")

            if self.plasmon_surrounding_material_index < 1.0:
                raise ValueError("'surrounding_material_index' must be >= 1.0 (vacuum).")
            elif self.plasmon_surrounding_material_index == 1.0:
                logger.debug("For 'surrounding_material_index' value of 1.0 (vacuum) is being used.")
            elif self.plasmon_surrounding_material_index == 1.33:
                logger.debug("For 'surrounding_material_index' typical value of 1.33 for water is being used.")
            elif self.plasmon_surrounding_material_index > 5.0:
                logger.warning("'surrounding_material_index' is unusually high; please verify this value.")

        # Plasmon source params
        if hasattr(self, 'plasmon_source'):
            for attr in ['plasmon_source_type', 'plasmon_source_center', 'plasmon_source_size', 'plasmon_source_component', 'plasmon_source_additional_parameters']:
                if not hasattr(self, attr):
                    pretty = attr.removeprefix("plasmon_source_")
                    raise ValueError(f"Source requires '{pretty}' attribute .")
        else:
            logger.info('No source chosen for simulation. Continuing without it.')

        # Nanoparticle params
        if hasattr(self, 'nanoparticle'):
            for attr in ['nanoparticle_material', 'nanoparticle_radius', 'nanoparticle_center']:
                if not hasattr(self, attr):
                    pretty = attr.removeprefix("nanoparticle_")
                    raise ValueError(f"Nanoparticle requires '{pretty}' attribute.")
        
        # Images params
        # TODO: _attribute_formation for 
            # png_args = f"-S 10 -m {self.intensityMin} -M {self.intensityMax} -z {self.frameCenter} -Zc dkbluered"
            # add -z {self.frameCenter} afterword in simulation.py
        if hasattr(self, 'images'):
            if not hasattr(self, 'images_timesteps_between'):
                raise ValueError("Images requires 'timesteps_between' attribute.")
        else:
            logger.info('No picture output chosen for simulation. Continuing without it.')

        # Molecule position param
        if hasattr(self, 'plasmol_molecule_position'):
            if not isinstance(self.plasmol_molecule_position, (list, tuple)):
                raise ValueError("Molecule position must be a list or tuple.")
            elif len(self.plasmol_molecule_position) != 3:
                raise ValueError("Molecule position must be an array of three numbers [x, y, z].")
            elif any(not isinstance(coord, (int, float)) for coord in self.plasmol_molecule_position):
                    raise ValueError("Molecule position coordinates must be numbers.")
            elif all(coord == 0 for coord in self.plasmol_molecule_position):
                    logger.warning("Molecule position is set to [0, 0, 0]; ensure this is intended.")
        
        # Molecule params
        if hasattr(self, 'molecule'):
            for attr in ['molecule_geometry', 'molecule_geometry_units', 'molecule_basis', 'molecule_charge', 'molecule_spin', 'molecule_xc']:
                if not hasattr(self, attr):
                    pretty = attr.removeprefix("molecule_")
                    raise ValueError(f"Molecule requires '{pretty}' attribute.")

            # TODO: create constants.SUPPORTED_LRC_XC_FUNCTIONALS
            if hasattr(self, 'molecule_lrc_parameter'):
                if self.molecule_xc.lower() not in constants.SUPPORTED_LRC_XC_FUNCTIONALS:
                    raise ValueError(f"LRC parameter provided but xc functional '{self.molecule_xc}' is not long-range corrected.")
            
            # TODO: create self_check_custom_xc in utils/input/checks.py
            if hasattr(self, 'molecule_custom_xc'):
                self_check_custom_xc(self.molecule_custom_xc)
            
            if hasattr(self, 'molecule_propagator'):
                self.molecule_propagator = self.molecule_propagator.lower()
                if self.molecule_propagator not in ['step', 'rk4', 'magnus2']:
                    raise ValueError(f"Unsupported propagator: {self.molecule_propagator}. Acceptable: step, rk4, magnus2.")
            
            if not isinstance(self.molecule_geometry, list):
                raise ValueError("The 'molecule' section must contain a 'geometry' list of atoms and coords. See documentation for details on how to format this.")

            if not self.molecule_geometry_units in ['angstrom', 'bohr']:
                raise ValueError(f"Invalid 'molecule_geometry_units': '{self.molecule_geometry_units}'. Must be 'angstrom' or 'bohr'.")

            self.molecule_atoms, self.molecule_coords, self.molecule_geometry_units = postprocess_molecule(
                self.molecule_geometry,
                self.molecule_geometry_units.lower()
            )

            delattr(self, 'molecule_geometry')

        if hasattr(self, 'plasmon_source') and hasattr(self, 'molecule_source'):
            logger.warning("Source found in molecule section, but this is a full PlasMol run. Ignoring molecule source; using plasmon source.")
            if hasattr(self, 'molecule_source'): delattr(self, 'molecule_source')
            if hasattr(self, 'molecule_source_type'): delattr(self, 'molecule_source_type')
            if hasattr(self, 'molecule_source_intensity_au'): delattr(self, 'molecule_source_intensity_au')
            if hasattr(self, 'molecule_source_peak_time_au'): delattr(self, 'molecule_source_peak_time_au')
            if hasattr(self, 'molecule_source_width_steps'): delattr(self, 'molecule_source_width_steps')
            if hasattr(self, 'molecule_source_wavelength_nm'): delattr(self, 'molecule_source_wavelength_nm')
            if hasattr(self, 'molecule_source_dir'): delattr(self, 'molecule_source_dir')
        elif hasattr(self, 'molecule_source'):
            for attr in ['molecule_source_type', 'molecule_source_intensity_au', 'molecule_source_peak_time_au', 'molecule_source_width_steps', 'molecule_source_wavelength_nm', 'molecule_source_dir']:
                if not hasattr(self, attr):
                    pretty = attr.removeprefix("molecule_source_")
                    raise ValueError(f"Molecule source requires '{pretty}' attribute.")
            for attr in ['molecule_source_intensity_au', 'molecule_source_peak_time_au', 'molecule_source_width_steps', 'molecule_source_wavelength_nm']:
                if hasattr(self, attr):
                    value = getattr(self, attr)
                    if value is not None and value <= 0:
                        pretty = attr.removeprefix("molecule_source_")
                        raise ValueError(f"Molecule source '{pretty}' must be a positive value.")
            if self.molecule_source_type.lower() == 'pulse' and not hasattr(self, 'molecule_source_wavelength_nm'):
                raise ValueError("Molecule source of type 'pulse' requires 'wavelength_nm' attribute.")
            if self.molecule_source_type.lower() == 'pulse' or self.molecule_source_type.lower() == 'kick':
                raise ValueError("Molecule source must be of type 'pulse' or 'kick'.")

        # Fourier params
        if hasattr(self, 'fourier'):
            logger.info("Fourier modifier selected; running three simulations for Fourier analysis along each axis.")
            for attr in ['fourier_gamma', 'fourier_pfield_filepath', 'fourier_spectrum_filepath']:
                if not hasattr(self, attr) or getattr(self, attr) in ['']:
                    pretty = attr.removeprefix("fourier_")
                    raise ValueError(f"Fourier modifier requires '{pretty}' attribute.")
            if self.fourier_gamma <= 0:
                raise ValueError("Fourier modifier 'gamma' must be a positive value.")
            
        # Lopata Broadening params
        if hasattr(self, 'broadening'):
            logger.info("Broadening modifier selected; applying Lopata broadening to spectra.")
            for attr in ['broadening_type', 'broadening_gam0', 'broadening_xi', 'broadening_eps0', 'broadening_clamp']:
                    if not hasattr(self, attr) or getattr(self, attr) in ['']:
                        pretty = attr.removeprefix("broadening_")
                        raise ValueError(f"Broadening modifier requires '{pretty}' attribute.")
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
        if hasattr(self, 'comparison'):
            logger.info("Comparison modifier selected; preparing to run additional simulations for comparison to quantum results.")
            if hasattr(self, 'comparison_bases') and hasattr(self, 'comparison_xcs'):
                raise ValueError("Comparison mode requires both 'bases' and 'xcs' lists. See documentation for details.")
            if self.comparison_num_virtual < 1:
                raise ValueError("Comparison 'num_virtual' must be at least 1.")
            if self.comparison_num_occupied < 1:
                raise ValueError("Comparison 'num_occupied' must be at least 1.")

        # Dampening mode params
        if hasattr(self, 'dampening'):
            logger.info("Dampening modifier selected; preparing to apply dampening to time-domain signals. See documentation for details.")
            if not hasattr(self, 'dampening_gamma'):
                raise ValueError("Dampening modifier requires 'gamma' attribute.")
            if self.dampening_gamma <= 0:
                raise ValueError("Dampening 'gamma' must be a positive value.")

        # Checkpointing params
        if hasattr(self, 'checkpoint'):
            logger.info("Checkpointing selected; preparing to save and load checkpoints during simulation.")
            if not hasattr(self, 'checkpoint_filepath') or self.checkpoint_filepath in ['']:
                raise ValueError("Checkpointing requires 'filepath' attribute for checkpoint file.")
            if not hasattr(self, 'checkpoint_snapshot_frequency'):
                raise ValueError("Checkpointing requires 'frequency' attribute for snapshot frequency.")
            if self.checkpoint_snapshot_frequency <= 0:
                raise ValueError("Checkpointing 'frequency' must be a positive value.")

        # Files
        for file in ['field_e_filepath', 'field_p_filepath']:
            if hasattr(self, file):
                value = getattr(self, file)
                if not isinstance(value, str) or value in ['']:
                    pretty = file.removeprefix("field_").removesuffix("_filepath")
                    raise ValueError(f"Filepath for '{pretty}' must be a non-empty string.")



    # def _attribute_formation(self):
        #     self.source = sources.MEEPSOURCE(
        #         source_type=self.plasmon_source_type,
        #         sourceCenter=self.plasmon_source_center,
        #         sourceSize=self.plasmon_source_size,
        #         component=self.plasmon_source_component,
        #         amplitude=self.plasmon_source_amplitude,
        #         is_integrated=self.plasmon_source_is_integrated,
        #         **{k: v for k, v in self.plasmon_source_additional_parameters.items()}
        #     )


        #     self.molecule_source = electric_field.ELECTRICFIELD(
        #         source_type=self.molecule_source_type,
        #         intensity_au=self.molecule_source_intensity_au,
        #         peak_time_au=self.molecule_source_peak_time_au,
        #         width_steps=self.molecule_source_width_steps,
        #         width_au=self.molecule_source_width_au,
        #         wavelength_nm=self.molecule_source_wavelength_nm,
        #         wavelength_au=self.molecule_source_wavelength_au,
        #         direction=self.molecule_source_dir
        #     )




        #     if 'imageDirName' not in self.hdf5:
        #         self.hdf5['imageDirName'] = f"plasmon-{datetime.now().strftime('%m%d%Y_%H%M%S')}"
        #         logger.info(f"Directory for images: {os.path.abspath(self.hdf5['imageDirName'])}")


        # # Symmetries
        # if hasattr(self, 'symmetries'):
        #     symmetries_list = []
        #     sym = self.symmetries
        #     i = 0
        #     while i < len(sym):
        #         if sym[i] in ['X', 'Y', 'Z']:
        #             if i + 1 < len(sym):
        #                 try:
        #                     phase = int(sym[i + 1])
        #                 except ValueError:
        #                     raise ValueError(f"Symmetry '{sym[i]}' not followed by valid integer.")
        #                 dir_map = {'X': mp.X, 'Y': mp.Y, 'Z': mp.Z}
        #                 symmetries_list.append(mp.Mirror(dir_map[sym[i]], phase=phase))
        #                 i += 2
        #             else:
        #                 raise ValueError(f"Symmetry '{sym[i]}' has no value following it.")
        #         else:
        #             i += 1
        #     self.symmetry = symmetries_list if symmetries_list else None

        # # Set resolution based on dt
        # # might can be moved to simulation.py
        # if hasattr(self, 'plasmon') and self.plasmon:
        #     self.resolution = round(0.5 / (self.dt / constants.convertTimeMeep2Atomic))

        # # Nanoparticle params
        # if hasattr(self, 'nanoparticle') and self.nanoparticle:
        #     self.nanoparticle = mp.Sphere(
        #         radius=self.radius,
        #         center=mp.Vector3(*self.nanoparticle_dict.get('center', [0, 0, 0])),
        #         material=material
        #     )
        # else:
        #     logger.info('No object chosen for simulation. Continuing without it.')
        #     self.nanoparticle = None

        # if hasattr(self, 'material'):
        #     self.material = self._load_meep_material(self.material)




    def _get_nested_value(self, d, path, default=None):
        """Helper to get nested dict value safely."""
        cur = d
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return default

        return cur

    def _load_meep_material(material_str):
        import importlib
        materials = importlib.import_module("meep.materials")
        try:
            return getattr(materials, material_str)
        except AttributeError as e:
            raise ImportError(
                f"Material '{material_str}' not found in meep.materials. "
                f"Check spelling/case or available materials."
            ) from e


def postprocess_molecule(geometry, units):
    """
    Post-process molecule geometry:
    - Validate input
    - Convert to Bohr units
    - Build the exact coords string expected by the simulator
    """
    atoms = []
    coords_bohr = {}

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

    # Convert to Bohr
    if units.startswith('angstrom'):
        factor = 1.8897259886
        coords_bohr = {label: xyz * factor for label, xyz in coords_bohr.items()}
        units = "bohr"

    # Build the exact string format the simulator wants
    coords_str = ""
    for i, atom in enumerate(atoms):
        x, y, z = coords_bohr[f"{atom}{i+1}"]
        coords_str += f" {atom} {x} {y} {z}"
        if i < len(atoms) - 1:
            coords_str += ";"

    return atoms, coords_str.strip(), units


def parseInputFile(args):
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

    # ---- Determine simulation type + validation ----
    simulation_types = []
    if molecule_params:
        simulation_types.append('molecule')
    if plasmon_params:
        simulation_types.append('plasmon')

    if not simulation_types:
        raise RuntimeError(
            "The minimum required parameters were not given. "
            "Please check guidelines for information on minimal requirements."
        )

    # Plasmon-specific validation (required whenever plasmon section exists)
    if plasmon_params:
        if 'simulation' not in plasmon_params:
            raise RuntimeError(
                "No 'simulation' object found in 'plasmon' section. "
                "Please specify the 'simulation' in the 'plasmon' section."
            )
        if molecule_params and 'molecule_position' not in plasmon_params:
            raise RuntimeError(
                "No 'molecule_position' object found in 'plasmon' section, "
                "but quantum (molecule) is present. "
                "Please specify the 'molecule_position' parameters in the 'plasmon' section."
            )

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

    return preparams
