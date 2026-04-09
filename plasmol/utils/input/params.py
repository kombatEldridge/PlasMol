import logging
import math
import inspect
import sys
import numpy as np
import re
import json
import os
import meep as mp
from pyscf.dft import libxc

from plasmol.classical.sources import MEEPSOURCE
from plasmol.quantum.electric_field import ELECTRICFIELD
from plasmol.quantum.propagators import *
from plasmol.utils.input.models import PlasMolInput   # NEW: Pydantic v2 models

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

        self._load_and_validate_input(args)

        # Store CLI logging parameters for use in child processes etc.
        self.verbose = getattr(args, 'verbose', 1)
        self.log = getattr(args, 'log', None)

        self._post_process()
        logger.info("All parameters successfully parsed and validated with Pydantic v2.")

    def _load_and_validate_input(self, args):
        """Load JSON (removing comments) and validate with Pydantic v2 models."""
        input_path = args.input
        with open(input_path, 'r') as f:
            # Removes comments (keeps the same behavior you had before)
            content = ''.join(re.sub(r"(#|--|%|//)(.*)$", '', line) for line in f if not line.strip().startswith(('#', '--', '%', '//')))
            raw = json.loads(content)

        self.input_model: PlasMolInput = PlasMolInput.model_validate(raw)

        # Expose everything as flat attributes (keeps the rest of your codebase unchanged)
        for k, v in self.input_model.model_dump().items():
            setattr(self, k, v)

        # Store a few convenience attributes that were derived before
        self.input_file_path = input_path
        self.simulation_types = []
        if self.molecule is not None:
            self.simulation_types.append('molecule')
        if self.plasmon is not None:
            self.simulation_types.append('plasmon')
        self.xyz = ['x', 'y', 'z']

    def _post_process(self):
        """
        This function is meant to form the attributes so they are ready to
        be used by the rest of the codebase.
        (Most validation is now handled by Pydantic)
        """
        self.run_molecule_simulation = False
        self.run_plasmon_simulation = False
        if 'molecule' in self.simulation_types and 'plasmon' in self.simulation_types:
            pass
        elif 'molecule' in self.simulation_types:
            self.run_molecule_simulation = True
        elif 'plasmon' in self.simulation_types:
            self.run_plasmon_simulation = True

        dt_str = f"{self.dt:.10f}".rstrip('0')
        self.time_rounding_decimals = len(dt_str.split('.')[-1]) if '.' in dt_str else 0

        if self.plasmon is not None:  # has_plasmon
            if hasattr(self.plasmon.simulation, 'cell_volume') and self.plasmon.simulation.cell_volume:
                self.cell_volume = mp.Vector3(*self.plasmon.simulation.cell_volume)
            else:
                self.cell_volume = mp.Vector3(self.plasmon.simulation.cell_length,
                                              self.plasmon.simulation.cell_length,
                                              self.plasmon.simulation.cell_length)

            if hasattr(self.plasmon.simulation, 'symmetries') and self.plasmon.simulation.symmetries:
                symmetries_list = []
                dir_map = {'X': mp.X, 'Y': mp.Y, 'Z': mp.Z}
                for i in range(0, len(self.plasmon.simulation.symmetries), 2):
                    axis = self.plasmon.simulation.symmetries[i].upper()
                    phase = int(self.plasmon.simulation.symmetries[i + 1])
                    symmetries_list.append(mp.Mirror(dir_map[axis], phase=phase))
                self.plasmon_symmetries = symmetries_list if symmetries_list else None
            else:
                self.plasmon_symmetries = None

            if self.plasmon.source is not None:  # has_plasmon_source
                self.plasmon_source_object = MEEPSOURCE(
                    source_type=self.plasmon.source.type.lower().strip(),
                    source_center=self.plasmon.source.center,
                    source_size=self.plasmon.source.size,
                    component=self.plasmon.source.component.lower().strip(),
                    amplitude=self.plasmon.source.amplitude,
                    is_integrated=self.plasmon.source.is_integrated,
                    **self.plasmon.source.additional_parameters
                )

            if self.plasmon.nanoparticle is not None:  # has_nanoparticle
                self.nanoparticle_material = self._load_meep_material(self.plasmon.nanoparticle.material)
                self.nanoparticle = mp.Sphere(
                    radius=self.plasmon.nanoparticle.radius,
                    center=mp.Vector3(*self.plasmon.nanoparticle.center),
                    material=self.nanoparticle_material
                )

            if self.plasmon.images is not None:  # has_images
                self.images_args = ""
                for s in self.plasmon.images.additional_parameters:
                    self.images_args += f" {s}"

            if self.plasmon.molecule_position is not None:  # has_molecule_position
                self.molecule_position = mp.Vector3(*self.plasmon.molecule_position)

        if self.molecule is not None:  # has_molecule
            self.molecule_atoms, self.molecule_coords, self.molecule_geometry_units = self._construct_geometry(
                self.molecule.geometry, self.molecule.geometry_units.lower())

            propagator_map = {
                "step": propagate_step,
                "magnus2": propagate_magnus2,
                "rk4": propagate_rk4
            }
            self.molecule_propagator_str = self.molecule.propagator.type.lower()
            self.molecule_propagator = propagator_map[self.molecule_propagator_str]
            sig = inspect.signature(self.molecule_propagator)
            exclude_args = {'molecule', 'exc'}
            self.molecule_propagator_params = {name: getattr(self.molecule.propagator, name)
                                               for name in sig.parameters if name not in exclude_args}

            time_values = np.arange(0, self.dt + self.t_end, self.dt)  # fixed off-by-one
            self.times = np.linspace(0, time_values[-1], int(len(time_values)))
            self.molecule_source_field = ELECTRICFIELD(self).field

            if self.molecule.modifiers and self.molecule.modifiers.broadening:
                self.broadening_dict = self.molecule.modifiers.broadening.model_dump()
                del self.broadening_dict["type"]

            if self.molecule.modifiers and self.molecule.modifiers.fourier:
                for dir_ in {"x", "y", "z"}:
                    attr = f"field_e_{dir_}_filepath"
                    value = f"{dir_}_dir/{self.molecule.files.field_e_filepath}"
                    setattr(self, attr, value)
                    attr = f"field_p_{dir_}_filepath"
                    value = f"{dir_}_dir/{self.molecule.files.field_p_filepath}"
                    setattr(self, attr, value)
                    attr = f"spectra_e_{dir_}_vs_p_{dir_}_filepath"
                    value = f"{dir_}_dir/{self.molecule.files.spectra_e_vs_p_filepath or 'spectra.png'}"
                    setattr(self, attr, value)

    # ----------------------------------------------------------------------
    # The helper methods below are unchanged from your original file
    # ----------------------------------------------------------------------

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