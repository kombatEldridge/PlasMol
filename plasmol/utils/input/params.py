# utils/input/params.py
import logging
import numpy as np
import meep as mp
from datetime import datetime
import os

from plasmol.classical import sources
from plasmol import constants

logger = logging.getLogger("main")

class PARAMS:
    """
    Container for simulation parameters given from input files and cli inputs.
    """
    def __init__(self, parsed):
        self.parsed = parsed
        self.type = self.parsed["simulation_type"]
        self.restart = self.parsed["args"].get("restart", False)  # Default to False if not provided
        self.do_nothing = self.parsed["args"].get("do_nothing", False)  # Default to False

        # Define a large, extensible list of parameter definitions.
        # Each entry is a tuple: (attribute_name, path_as_list, default_value, section_condition)
        # - attribute_name: str, the name to set as self.<attribute_name>
        # - path_as_list: list of str, the nested keys to access in parsed (e.g., ['settings', 'dt'])
        # - default_value: any, value if not found (use None for required params)
        # - section_condition: str or None, only set if self.type in ['PlasMol', 'Plasmon', 'Molecule'] matches or None for always
        # Add new params here as needed; required params without default will raise error if missing.
        param_defs = [
            # Settings (always required)
            ('dt', ['settings', 'dt'], None, None),
            ('t_end', ['settings', 't_end'], None, None),

            # Plasmon params
            ('plasmon', ['plasmon'], None, 'plasmon'),
            ('tolerance_efield', ['plasmon', 'simulation', "tolerance_efield"], 1e-12, 'plasmon'),
            ('cell_length', ['plasmon', 'simulation', "cell_length"], None, 'plasmon'),
            ('pml_thickness', ['plasmon', 'simulation', "pml_thickness"], None, 'plasmon'),
            ('symmetries', ['plasmon', 'simulation', 'symmetries'], None, 'plasmon'),
            ('surrounding_material_index', ['plasmon', 'simulation', "surrounding_material_index"], 1.33, 'plasmon'),

            # Plasmon source params
            ('plasmon_source', ['plasmon', 'source'], False, 'plasmon'),
            ('plasmon_source_type', ['plasmon', 'source', 'source_type'], False, 'plasmon'),
            ('plasmon_source_center', ['plasmon', 'source', 'source_center'], False, 'plasmon'),
            ('plasmon_source_size', ['plasmon', 'source', 'source_size'], False, 'plasmon'),
            ('plasmon_source_component', ['plasmon', 'source', 'component'], False, 'plasmon'),
            ('plasmon_source_amplitude', ['plasmon', 'source', 'amplitude'], False, 'plasmon'),
            ('plasmon_source_source_dict', ['plasmon', 'source', 'additional_parameters'], False, 'plasmon'),

            # Nanoparticle params
            ('nanoparticle', ['plasmon', 'nanoparticle'], False, 'plasmon'),
            ('material', ['plasmon', 'nanoparticle', 'material'], False, 'plasmon'),
            ('radius', ['plasmon', 'nanoparticle', 'radius'], False, 'plasmon'),
            ('center', ['plasmon', 'nanoparticle', 'center'], False, 'plasmon'),

            # HDF5 params
            ('images', ['plasmon', 'images'], False, 'plasmon'),
            ('timesteps_between', ['plasmon', 'images', 'timesteps_between'], None, 'plasmon'),
            ('intensity_min', ['plasmon', 'images', 'intensity_min'], None, 'plasmon'),
            ('intensity_max', ['plasmon', 'images', 'intensity_max'], None, 'plasmon'),
            ('image_dir_name', ['plasmon', 'images', 'image_dir_name'], None, 'plasmon'),

            # Molecule position for PlasMol
            ('molecule_position', ['plasmon', 'molecule_position'], None, 'plasmon'),

            # Molecule params
            ('molecule', ['molecule'], False, 'molecule'),
            ('coords', ['molecule', 'geometry', 'coords'], None, 'molecule'),
            ('atoms', ['molecule', 'geometry', 'atoms'], None, 'molecule'),
            ('basis', ['molecule', 'basis'], None, 'molecule'),
            ('charge', ['molecule', 'charge'], None, 'molecule'),
            ('spin', ['molecule', 'spin'], None, 'molecule'),
            ('xc', ['molecule', 'xc'], None, 'molecule'),
            ('lrc_parameter', ['molecule', 'lrc_parameter'], None, 'molecule'),
            ('custom_xc', ['molecule', 'custom_xc'], None, 'molecule'),
            ('propagator', ['molecule', 'propagator', "type"], None, 'molecule'),
            ('pc_convergence', ['molecule', 'propagator', "pc_convergence"], 1e-12, 'molecule'),
            ('max_iterations', ['molecule', 'propagator', "max_iterations"], 200, 'molecule'),
            ('hermiticity_tolerance', ['molecule', 'hermiticity_tolerance'], 1e-12, 'molecule'),

            # Source params (molecule section)
            ('molecule_source', ['molecule', 'source'], False, 'molecule'),
            ('molecule_source_type', ['molecule', 'source', 'type'], None, 'molecule'),
            ('molecule_source_intensity_au', ['molecule', 'source', 'intensity_au'], None, 'molecule'),
            ('molecule_source_peak_time_au', ['molecule', 'source', 'peak_time_au'], None, 'molecule'),
            ('molecule_source_width_steps', ['molecule', 'source', 'width_steps'], None, 'molecule'),
            ('molecule_source_width_au', ['molecule', 'source', 'width_au'], None, 'molecule'),
            ('molecule_source_wavelength_nm', ['molecule', 'source', 'wavelength_nm'], None, 'molecule'),
            ('molecule_source_wavelength_au', ['molecule', 'source', 'wavelength_au'], None, 'molecule'),
            ('molecule_source_dir', ['molecule', 'source', 'dir'], None, 'molecule'),

            # Fourier params runs three sims at once, one per axis
            ('fourier', ['molecule', 'modifiers', 'fourier'], False, 'molecule'),
            ('fourier_gamma', ['molecule', 'modifiers', 'fourier', 'gamma'], None, 'molecule'),
            ('filepath_pfield_fourier', ['molecule', 'modifiers', 'fourier', 'filepath_pfield_fourier'], None, 'molecule'),
            ('filepath_spectrum_fourier', ['molecule', 'modifiers', 'fourier', 'filepath_spectrum_fourier'], None, 'molecule'),

            # Lopata Broadening params
            ('broadening', ['molecule', 'modifiers', 'broadening'], False, 'molecule'),
            ('broadening_type', ['molecule', 'modifiers', 'broadening', "type"], None, 'molecule'),
            ('broadening_gam0', ['molecule', 'modifiers', 'broadening', "gam0"], None, 'molecule'),
            ('broadening_xi', ['molecule', 'modifiers', 'broadening', "xi"], None, 'molecule'),
            ('broadening_eps0', ['molecule', 'modifiers', 'broadening', "eps0"], None, 'molecule'),
            ('broadening_clamp', ['molecule', 'modifiers', 'broadening', "clamp"], None, 'molecule'),

            # Comparison mode params
            ('comparison', ['molecule', 'modifiers', 'comparison'], False, 'molecule'),
            ('comparison_bases', ['molecule', 'modifiers', 'comparison', 'bases'], [], 'molecule'),
            ('comparison_xcs', ['molecule', 'modifiers', 'comparison', 'xcs'], [], 'molecule'),
            ('comparison_num_virtual', ['molecule', 'modifiers', 'comparison', 'num_virtual'], 3, 'molecule'),
            ('comparison_num_occupied', ['molecule', 'modifiers', 'comparison', 'num_occupied'], 3, 'molecule'),
            ('comparison_y_min', ['molecule', 'modifiers', 'comparison', 'y_min'], -1, 'molecule'),
            ('comparison_y_max', ['molecule', 'modifiers', 'comparison', 'y_max'], 1, 'molecule'),

            # Comparison mode params
            ('dampening', ['molecule', 'modifiers', 'dampening'], False, 'molecule'),
            ('dampening_gamma', ['molecule', 'modifiers', 'dampening', 'gamma'], None, 'molecule'),

            # Checkpointing params
            ('checkpoint', ['molecule', 'files', 'checkpoint'], False, 'molecule'),
            ('filepath_checkpoint', ['molecule', 'files', 'checkpoint', 'filepath_checkpoint'], None, 'molecule'),
            ('snapshot_frequency', ['molecule', 'files', 'checkpoint', 'snapshot_frequency'], None, 'molecule'),

            # Files
            ('filepath_efield', ['molecule', 'files', 'filepath_efield'], 'eField.csv', 'molecule'),
            ('filepath_pfield', ['molecule', 'files', 'filepath_pfield'], 'pField.csv', 'molecule'),
            ('filepath_efield_vs_pfield', ['molecule', 'files', 'filepath_efield_vs_pfield'], None, 'molecule'),
            ('filepath_fourier_spectrum', ['molecule', 'files', 'filepath_fourier_spectrum'], None, 'molecule'),
        ]

        # Populate attributes from param_defs
        for attr, path, default, condition in param_defs:
            if condition and condition not in self.type.lower():
                continue  # Skip if condition not met (e.g., 'molecule' params only if molecule present)
            value = self._get_nested_value(self.parsed, path, default)
            if value is not None:
                setattr(self, attr, value)

        if hasattr(self, 'propagator'):
            self.propagator = self.propagator.lower()
            if self.propagator not in ['step', 'rk4', 'magnus2']:
                raise ValueError(f"Unsupported propagator: {self.propagator}. Acceptable: step, rk4, magnus2.")

        if hasattr(self, 'comparison'):
            if not self.comparison_bases or not self.comparison_xcs:
                raise ValueError("Comparison mode requires both 'bases' and 'xcs' lists.")
            
            if hasattr(self, 'shape'):
                logger.warning("Comparison mode ignores source; no time propagation.")
                delattr(self, 'shape')

        if hasattr(self, 'shape') and self.type != 'Molecule':
            logger.warning("Source found in molecule section, but full PlasMol available. Ignoring molecule source; use plasmon section.")
            delattr(self, 'shape')  # And remove other molecule source attrs if set
            if hasattr(self, 'peak_time_au'): delattr(self, 'peak_time_au')
            if hasattr(self, 'width_steps'): delattr(self, 'width_steps')
            if hasattr(self, 'intensity_au'): delattr(self, 'intensity_au')
            if hasattr(self, 'wavelength_nm'): delattr(self, 'wavelength_nm')
            if hasattr(self, 'dir'): delattr(self, 'dir')

        # plasmon-specific instantiations (if plasmon present)
        if 'plasmon' in self.type.lower():
            self._instantiate_plasmon_objects()

        delattr(self, 'parsed')  # Clean up

    def _get_nested_value(self, d, path, default=None):
        """Helper to get nested dict value safely."""
        for key in path:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                return default
        return d

    def _instantiate_plasmon_objects(self):
        """Instantiate plasmon objects like source, nanoparticle, etc."""
        # Source
        if hasattr(self, 'source_dict') and self.source_dict:
            source_type = self.source_dict.get('sourceType')
            if not source_type:
                raise ValueError("Source requires 'sourceType'.")
            self.source = sources.MEEPSOURCE(
                source_type=source_type,
                sourceCenter=self.source_dict.get('sourceCenter', [0, 0, 0]),
                sourceSize=self.source_dict.get('sourceSize', [0, 0, 0]),
                frequency=self.source_dict.get('frequency'),
                wavelength=self.source_dict.get('wavelength'),
                component=self.source_dict.get('component', 'z'),
                amplitude=self.source_dict.get('amplitude', 1.0),
                is_integrated=self.source_dict.get('is_integrated', True),
                **{k: v for k, v in self.source_dict.items() if k not in [
                    'sourceType', 'sourceCenter', 'sourceSize', 'frequency', 'wavelength',
                    'component', 'amplitude', 'is_integrated']}
            )
        else:
            logger.info('No source chosen for simulation. Continuing without it.')
            self.source = None

        # Nanoparticle
        if hasattr(self, 'nanoparticle_dict') and self.nanoparticle_dict:
            material_str = self.nanoparticle_dict.get('material')
            if material_str == 'Au':
                from meep.materials import Au_JC_visible as Au
                material = Au
            elif material_str == 'Ag':
                from meep.materials import Ag
                material = Ag
            else:
                raise ValueError(f"Unsupported material type: {material_str}")
            self.nanoparticle = mp.Sphere(
                radius=self.nanoparticle_dict.get('radius', 0),
                center=mp.Vector3(*self.nanoparticle_dict.get('center', [0, 0, 0])),
                material=material
            )
        else:
            logger.info('No object chosen for simulation. Continuing without it.')
            self.nanoparticle = None

        # Symmetries
        if hasattr(self, 'symmetries') and self.symmetries:
            symmetries_list = []
            sym = self.symmetries
            i = 0
            while i < len(sym):
                if sym[i] in ['X', 'Y', 'Z']:
                    if i + 1 < len(sym):
                        try:
                            phase = int(sym[i + 1])
                        except ValueError:
                            raise ValueError(f"Symmetry '{sym[i]}' not followed by valid integer.")
                        dir_map = {'X': mp.X, 'Y': mp.Y, 'Z': mp.Z}
                        symmetries_list.append(mp.Mirror(dir_map[sym[i]], phase=phase))
                        i += 2
                    else:
                        raise ValueError(f"Symmetry '{sym[i]}' has no value following it.")
                else:
                    i += 1
            self.symmetry = symmetries_list if symmetries_list else None
        else:
            logger.info('No symmetries chosen for simulation. Continuing without them.')
            self.symmetry = None

        # HDF5
        if hasattr(self, 'hdf5') and self.hdf5:
            required_keys = ['timestepsBetween', 'intensityMin', 'intensityMax']
            if any(key not in self.hdf5 for key in required_keys):
                raise ValueError(f"HDF5 requires {', '.join(required_keys)}.")
            if 'imageDirName' not in self.hdf5:
                self.hdf5['imageDirName'] = f"plasmon-{datetime.now().strftime('%m%d%Y_%H%M%S')}"
                logger.info(f"Directory for images: {os.path.abspath(self.hdf5['imageDirName'])}")
        else:
            logger.info('No picture output chosen for simulation. Continuing without it.')
            self.hdf5 = None

        # Simulation params (required for plasmon)
        if not hasattr(self, 'simulation') or not self.simulation:
            raise RuntimeError('No simulation parameters chosen for plasmon simulation.')
        
        # Molecule position (required for PlasMol)
        if self.type == 'PlasMol' and not hasattr(self, 'molecule_position'):
            raise RuntimeError('Molecule position required for PlasMol simulation.')

        # Adjust resolution based on dt
        if 'resolution' in self.simulation:
            dt_alt = (0.5 / self.simulation['resolution']) * constants.convertTimeMeep2Atomic
            if not np.isclose(dt_alt, self.dt):
                new_res = round(0.5 / (self.dt / constants.convertTimeMeep2Atomic))
                logger.info(f"Resolution does not match dt. Using new resolution: {new_res}")
                self.simulation['resolution'] = new_res
        else:
            new_res = round(0.5 / (self.dt / constants.convertTimeMeep2Atomic))
            self.simulation['resolution'] = new_res