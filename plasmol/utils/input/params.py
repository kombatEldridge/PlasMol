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
    def __init__(self, preparams):
        self.preparams = preparams
        self.type = self.preparams["simulation_type"]
        self.restart = self.preparams["args"].get("restart", False)  # Default to False if not provided
        self.do_nothing = self.preparams["args"].get("do_nothing", False)  # Default to False

        # Define a large, extensible list of parameter definitions.
        # Each entry is a tuple: (attribute_name, path_as_list, default_value, section_condition)
        # - attribute_name: str, the name to set as self.<attribute_name>
        # - path_as_list: list of str, the nested keys to access in preparams (e.g., ['settings', 'dt'])
        # - default_value: any, value if not found (use None for required params)
        # - section_condition: str or None, only set if self.type in ['PlasMol', 'Quantum', 'Classical'] matches or None for always
        # Add new params here as needed; required params without default will raise error if missing.
        param_defs = [
            # Settings (always required)
            ('dt', ['settings', 'dt'], None, None),
            ('t_end', ['settings', 't_end'], None, None),
            ('eField_path', ['settings', 'eField_path'], None, None),

            # Quantum params (for 'Quantum' or 'PlasMol')
            ('pField_path', ['quantum', 'files', 'pField_path'], None, 'quantum'),
            ('pField_Transform_path', ['quantum', 'files', 'pField_Transform_path'], None, 'quantum'),
            ('eField_vs_pField_path', ['quantum', 'files', 'eField_vs_pField_path'], None, 'quantum'),
            ('eV_spectrum_path', ['quantum', 'files', 'eV_spectrum_path'], None, 'quantum'),
            ('checkpoint_path', ['quantum', 'files', 'checkpoint', 'path'], None, 'quantum'),
            ('checkpoint_freq', ['quantum', 'files', 'checkpoint', 'frequency'], None, 'quantum'),
            ('molecule_coords', ['quantum', 'rttddft', 'geometry', 'molecule_coords'], None, 'quantum'),
            ('molecule_atoms', ['quantum', 'rttddft', 'geometry', 'atoms'], None, 'quantum'),
            ('atoms', ['quantum', 'rttddft', 'geometry', 'atoms'], None, 'quantum'),  # Alias for molecule_atoms
            ('basis', ['quantum', 'rttddft', 'basis'], None, 'quantum'),
            ('charge', ['quantum', 'rttddft', 'charge'], None, 'quantum'),
            ('spin', ['quantum', 'rttddft', 'spin'], None, 'quantum'),
            ('xc', ['quantum', 'rttddft', 'xc'], None, 'quantum'),
            ('mu', ['quantum', 'rttddft', 'mu'], None, 'quantum'),
            ('propagator', ['quantum', 'rttddft', 'propagator'], None, 'quantum'),  # Will be lowercased later
            ('check_tolerance', ['quantum', 'rttddft', 'check_tolerance'], None, 'quantum'),
            ('transform', ['quantum', 'rttddft', 'transform'], False, 'quantum'),  # Bool, default False
            ('fourier_gamma', ['quantum', 'rttddft', 'fourier_gamma'], None, 'quantum'),
            ('damping', ['quantum', 'rttddft', 'damping'], None, 'quantum'),  # Processed later
            ('mu_damping', ['quantum', 'rttddft', 'mu_damping'], 0, 'quantum'),
            ('bases', ['quantum', 'comparison', 'bases'], [], 'quantum'),
            ('xcs', ['quantum', 'comparison', 'xcs'], [], 'quantum'),
            ('num_virtual', ['quantum', 'comparison', 'num_virtual'], None, 'quantum'),
            ('num_occupied', ['quantum', 'comparison', 'num_occupied'], None, 'quantum'),  # Often same as num_virtual
            ('y_min', ['quantum', 'comparison', 'y_min'], None, 'quantum'),
            ('y_max', ['quantum', 'comparison', 'y_max'], None, 'quantum'),

            # Quantum source (optional, with warning if in PlasMol)
            ('shape', ['quantum', 'source', 'shape'], None, 'quantum'),
            ('peak_time_au', ['quantum', 'source', 'peak_time_au'], None, 'quantum'),
            ('width_steps', ['quantum', 'source', 'width_steps'], None, 'quantum'),
            ('intensity_au', ['quantum', 'source', 'intensity_au'], None, 'quantum'),
            ('wavelength_nm', ['quantum', 'source', 'wavelength_nm'], None, 'quantum'),
            ('dir', ['quantum', 'source', 'dir'], None, 'quantum'),

            # Classical params (for 'Classical' or 'PlasMol'; objects instantiated separately)
            ('simulation', ['classical', 'simulation'], None, 'classical'),
            ('molecule_position', ['classical', 'molecule'], None, 'classical'),
            ('source_dict', ['classical', 'source'], None, 'classical'),  # Raw dict for instantiation
            ('nanoparticle_dict', ['classical', 'object'], None, 'classical'),  # Raw dict for instantiation
            ('symmetries', ['classical', 'simulation', 'symmetries'], None, 'classical'),  # List for processing
            ('hdf5', ['classical', 'hdf5'], None, 'classical'),  # Dict, processed later
        ]

        # Populate attributes from param_defs
        for attr, path, default, condition in param_defs:
            if condition and condition not in self.type.lower():
                continue  # Skip if condition not met (e.g., 'quantum' params only if quantum present)
            value = self._get_nested_value(self.preparams, path, default)
            if value is None and default is None:
                raise RuntimeError(f"Required parameter '{attr}' (path: {'.'.join(path)}) is missing.")
            setattr(self, attr, value)

        # Post-processing for specific params
        if hasattr(self, 'propagator'):
            self.propagator = self.propagator.lower()
            if self.propagator == 'magnus2':
                self.maxiter = self._get_nested_value(self.preparams, ['quantum', 'rttddft', 'maxiter'], None)
                self.pc_convergence = self._get_nested_value(self.preparams, ['quantum', 'rttddft', 'pc_convergence'], None)
            elif self.propagator not in ['step', 'rk4', 'magnus2']:
                raise ValueError(f"Unsupported propagator: {self.propagator}. Acceptable: step, rk4, magnus2.")

        if hasattr(self, 'damping') and self.damping is not None:
            damping_dict = self._get_nested_value(self.preparams, ['quantum', 'rttddft', 'damping'], {})
            if isinstance(damping_dict, dict) and 'dynamic' in damping_dict:
                self.damping = 'dynamic'
            else:
                logger.warning("No damping type specified, defaulting to static.")
                self.damping = 'static'
            self.gam0 = damping_dict.get('gam0')
            self.xi = damping_dict.get('xi')
            self.eps0 = damping_dict.get('eps0')
            self.clamp = damping_dict.get('clamp')

        if hasattr(self, 'bases') and (self.bases or self.xcs):
            if not self.bases or not self.xcs:
                raise ValueError("Comparison mode requires both 'bases' and 'xcs' lists.")
            self.num_occupied = self.num_virtual  # Assuming same as num_virtual per original
            if hasattr(self, 'shape'):
                logger.warning("Comparison mode ignores source; no time propagation.")
                delattr(self, 'shape')

        if hasattr(self, 'shape') and self.type != 'Quantum':
            logger.warning("Source found in quantum section, but full PlasMol available. Ignoring quantum source; use classical section.")
            delattr(self, 'shape')  # And remove other quantum source attrs if set
            if hasattr(self, 'peak_time_au'): delattr(self, 'peak_time_au')
            if hasattr(self, 'width_steps'): delattr(self, 'width_steps')
            if hasattr(self, 'intensity_au'): delattr(self, 'intensity_au')
            if hasattr(self, 'wavelength_nm'): delattr(self, 'wavelength_nm')
            if hasattr(self, 'dir'): delattr(self, 'dir')

        # Classical-specific instantiations (if classical present)
        if 'classical' in self.type.lower():
            self._instantiate_classical_objects()

        delattr(self, 'preparams')  # Clean up

    def _get_nested_value(self, d, path, default=None):
        """Helper to get nested dict value safely."""
        for key in path:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                return default
        return d

    def _instantiate_classical_objects(self):
        """Instantiate classical objects like source, nanoparticle, etc."""
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
                self.hdf5['imageDirName'] = f"classical-{datetime.now().strftime('%m%d%Y_%H%M%S')}"
                logger.info(f"Directory for images: {os.path.abspath(self.hdf5['imageDirName'])}")
        else:
            logger.info('No picture output chosen for simulation. Continuing without it.')
            self.hdf5 = None

        # Simulation params (required for classical)
        if not hasattr(self, 'simulation') or not self.simulation:
            raise RuntimeError('No simulation parameters chosen for classical simulation.')
        
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