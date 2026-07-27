import os
import sys
import logging
from rich.table import Table
from rich.console import Console

from plasmol.utils.struct import param_defs
from plasmol.utils.params_helpers import CHECK_PIPELINE, FORM_PIPELINE, check_spatial_symmetries
from plasmol.utils.params_helpers.common import (
    get_nested_value,
    check_xc,
    load_meep_material,
    resolve_geometry_path,
    construct_geometry,
)
from plasmol.utils.params_helpers.parse_input import parse_input_file

logger = logging.getLogger("main")

# additional_parameters keys that must not mark the run as a custom-driver job
PLASMON_RUN_ADDITIONAL_PARAMETERS = frozenset({
    'decay_stop',
    'decay_threshold',
    'checkpoint_filename_used',
})


class PARAMS:
    def __init__(self, args):
        # Phase 1: CLI only provides -c / --checkpoint → restore files and exit.
        if args.input is None:
            self._restore_checkpoint_files(args)
            sys.exit(0)

        self.preparams = parse_input_file(args)
        self.verbose = getattr(args, 'verbose', 1)
        self.log = getattr(args, 'log', None)
        self.input_file_path = self.preparams["args"]["input"]
        self.simulation_types = self.preparams["simulation_types"]
        self.xyz = ['x', 'y', 'z']
        self.additional_parameters = {}

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

        default_values_used = []

        # Populate attributes from param_defs
        for attr, path, is_section_dict, boolean_name, default_value, section_condition, data_type, _, _ in param_defs:
            if section_condition is not None:
                if isinstance(section_condition, str):
                    if section_condition not in self.simulation_types:
                        continue
                elif isinstance(section_condition, list):
                    if not all(c in self.simulation_types for c in section_condition):
                        continue
            
            value = get_nested_value(self.preparams, path)
            
            if value is not None:
                if not isinstance(value, data_type):
                    raise ValueError(f"Invalid type for {attr}: expected {_type_name(data_type)}, got {_type_name(type(value))}.") 

            if path[0] == 'additional_parameters' and value is not None:
                self.additional_parameters[attr] = value
                if attr not in PLASMON_RUN_ADDITIONAL_PARAMETERS:
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
                            default_values_used.append((attr, default_value))
                            setattr(self, attr, default_value)

        if default_values_used:
            logger.debug("The following variables are using default values because none were specified:")
            for attr, default_value in default_values_used:
                logger.debug(f"    {attr}: {default_value}")

        if getattr(self, 'driver_str', None) == 'np_abs_cross_sec':
            self.has_np_abs_cross_sec = True
            for attr, _, _, boolean_name, default_value, _, _, _, _ in param_defs:
                if boolean_name == 'has_np_abs_cross_sec' and default_value is not None and not hasattr(self, attr):
                    setattr(self, attr, default_value)
                    default_values_used.append((attr, default_value))
        else:
            self.has_np_abs_cross_sec = False

        if getattr(self, 'driver_str', None) == 'core_hole':
            self.has_core_hole = True
            for attr, _, _, boolean_name, default_value, _, _, _, _ in param_defs:
                if boolean_name == 'has_core_hole' and default_value is not None and not hasattr(self, attr):
                    setattr(self, attr, default_value)
                    default_values_used.append((attr, default_value))
        else:
            self.has_core_hole = False
            
        self._attribute_checks()
        self._attribute_formation()
        self._test_symmetry()
        self._checkpoint_check(args)
        logger.info("All parameters successfully parsed and validated.")
        delattr(self, 'preparams')


    def _attribute_checks(self):
        """Run section checks (params_helpers/has_*.check)."""
        for fn in CHECK_PIPELINE:
            fn(self)

    def _attribute_formation(self):
        """Build derived objects (params_helpers FORM_PIPELINE)."""
        for fn in FORM_PIPELINE:
            fn(self)

    def _test_symmetry(self):
        """Validate Meep mirror symmetries after formation."""
        check_spatial_symmetries(self)

    def _get_nested_value(self, d, path):
        return get_nested_value(d, path)

    def _load_meep_material(self, material_str):
        return load_meep_material(material_str)

    def _check_xc(self, func_name: str, omega: float = None):
        return check_xc(self, func_name, omega)

    def _resolve_geometry_path(self, geometry: str):
        return resolve_geometry_path(self, geometry)

    def _construct_geometry(self, geometry, units):
        return construct_geometry(self, geometry, units)

    def _restore_checkpoint_files(self, args):
        """Phase 1: restore CSVs / input JSON / xyz from a CLI checkpoint path, then exit."""
        if not getattr(args, 'checkpoint', None):
            raise ValueError("No input file and no --checkpoint given.")
        if not os.path.exists(args.checkpoint):
            raise ValueError(
                f"Checkpoint file {args.checkpoint} not found, but resume from checkpoint flag ('-c') given."
            )
        from plasmol.utils.checkpoint import restore_files_from_checkpoint
        logger.info(f"Checkpoint file {args.checkpoint} found; restoring files only.")
        result = restore_files_from_checkpoint(args.checkpoint)
        logger.info("Simulation exiting so you can inspect / edit the restored input file.")
        logger.info(
            f"To continue, re-run with the restored input ('{result['restored_input_filepath']}')"
        )
        logger.info(
            "Parameters safe to edit before resuming include: dt, t_end, "
            "checkpoint_frequency_time/steps, checkpoint_filepath, "
            "fourier_max_ev/min_ev/gamma/npz_filepath/spectrum_filepath/field_e_ref_filepath/reference_only, "
            "spectra_e_vs_p_filepath, verbose."
        )
        logger.info("===== Directory is now setup to resume from checkpoint =====")

    def _checkpoint_check(self, args):
        """
        Phase 2: if the input declares checkpoint_filename_used, load non-file
        propagator state from that checkpoint. Otherwise mark a fresh run.
        """
        if getattr(args, 'checkpoint', None) and not getattr(self, 'checkpoint_filename_used', None):
            logger.warning(
                "CLI --checkpoint is ignored when an input file is provided without "
                "additional_parameters.checkpoint_filename_used. "
                "Run with only -c / --checkpoint first to restore files."
            )

        ckpt = getattr(self, 'checkpoint_filename_used', None)
        if ckpt:
            from plasmol.utils.checkpoint import load_state_from_checkpoint
            load_state_from_checkpoint(self, ckpt)
            # Ensure the runtime resume flag is set for drivers / molecule.
            self.resumed_from_checkpoint = True
            if not isinstance(getattr(self, 'additional_parameters', None), dict):
                self.additional_parameters = {}
            self.additional_parameters['checkpoint_filename_used'] = ckpt
            logger.info(
                f"===== Resuming simulation from checkpoint '{ckpt}' ====="
            )
        else:
            self.resumed_from_checkpoint = False

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

            if attr.endswith('_dict'):          # skip container dicts (plasmon_dict, etc.)
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

