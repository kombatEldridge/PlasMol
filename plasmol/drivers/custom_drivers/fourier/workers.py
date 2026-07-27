# Parallel worker entry points for quantum / plasmol / vacuum-reference jobs.
import logging

import meep as mp

from plasmol.classical.sources import MEEPSOURCE
from plasmol.classical.meep_verbosity import meep_io_context
from plasmol.drivers.plasmol import run as run_plasmol
from plasmol.drivers.quantum import run as run_quantum
from plasmol.utils.csv import init_csv
from plasmol.utils.logging import setup_logging

logger = logging.getLogger("main")


class PrefixFilter(logging.Filter):
    def __init__(self, direction):
        super().__init__()
        self.prefix = f"[{direction}-dir]"

    def filter(self, record):
        record.msg = f"{self.prefix} {record.msg}"
        return True


def build_meep_source(params):
    """Create the Meep incident-source object on params (not picklable across processes)."""
    with meep_io_context(getattr(params, 'verbose', 1)):
        params.plasmon_source_object = MEEPSOURCE(
            source_type=getattr(params, 'plasmon_source_type').lower().strip(),
            source_center=getattr(params, 'plasmon_source_center'),
            source_size=getattr(params, 'plasmon_source_size'),
            component=getattr(params, 'plasmon_source_component'),
            is_integrated=getattr(params, 'plasmon_source_is_integrated'),
            **{k: v for k, v in getattr(params, 'plasmon_source_additional_parameters', {}).items()}
        )


def build_plasmol_meep_objects(params):
    """Create Meep source/nanoparticle objects on params (not picklable across processes)."""
    build_meep_source(params)
    with meep_io_context(getattr(params, 'verbose', 1)):
        if getattr(params, 'has_nanoparticle', False):
            mat_name = params.nanoparticle_dict["material"]
            params.nanoparticle_material = params._load_meep_material(mat_name)
            params.nanoparticle = mp.Sphere(
                radius=getattr(params, 'nanoparticle_radius'),
                center=mp.Vector3(*getattr(params, 'nanoparticle_center')),
                material=params.nanoparticle_material
            )


def run_quantum_with_prefix(params_copy):
    setup_logging(
        getattr(params_copy, 'verbose', 1),
        getattr(params_copy, 'log', None)
    )
    f = PrefixFilter(params_copy.molecule_source_component)
    logging.getLogger("main").addFilter(f)
    logging.getLogger().addFilter(f)
    try:
        run_quantum(params_copy)
    finally:
        logging.getLogger("main").removeFilter(f)
        logging.getLogger().removeFilter(f)


def run_plasmol_with_prefix(params_copy):
    setup_logging(
        getattr(params_copy, 'verbose', 1),
        getattr(params_copy, 'log', None)
    )
    build_plasmol_meep_objects(params_copy)
    f = PrefixFilter(params_copy.plasmon_source_component)
    logging.getLogger("main").addFilter(f)
    logging.getLogger().addFilter(f)
    try:
        run_plasmol(params_copy)
    finally:
        logging.getLogger("main").removeFilter(f)
        logging.getLogger().removeFilter(f)


def run_reference_with_prefix(params_copy):
    """Vacuum reference: Meep only, no NP / no molecule; write E_inc at molecule site."""
    from plasmol.classical.simulation import SIMULATION

    setup_logging(
        getattr(params_copy, 'verbose', 1),
        getattr(params_copy, 'log', None)
    )
    build_meep_source(params_copy)
    f = PrefixFilter(f"ref-{params_copy.plasmon_source_component}")
    logging.getLogger("main").addFilter(f)
    logging.getLogger().addFilter(f)
    try:
        init_csv(
            params_copy.field_e_filepath,
            "Reference (vacuum, no NP/molecule) electric field intensity in atomic units",
        )
        SIMULATION(params_copy).run()
        logging.info(
            f"Vacuum reference E-field written to {params_copy.field_e_filepath}"
        )
    finally:
        logging.getLogger("main").removeFilter(f)
        logging.getLogger().removeFilter(f)
