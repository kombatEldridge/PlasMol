"""params_helpers: section check/form modules named by has_* gates."""
from plasmol.utils.params_helpers import (
    has_settings,
    has_custom,
    has_plasmon,
    has_simulation,
    has_plasmon_source,
    has_nanoparticle,
    has_images,
    has_molecule_position,
    has_molecule,
    has_molecule_source,
    has_fourier,
    has_cap,
    has_comparison,
    has_checkpoint,
    has_files,
    has_np_abs_cross_sec,
    has_core_hole,
    has_scatter_response_fxn,
)
from plasmol.utils.params_helpers import form_all as form_all_mod
from plasmol.utils.params_helpers.has_simulation import check_spatial_symmetries

CHECK_PIPELINE = [
    has_settings.check,
    has_plasmon.check,
    has_simulation.check,
    has_plasmon_source.check,
    has_nanoparticle.check,
    has_images.check,
    has_molecule_position.check,
    has_molecule.check,
    has_molecule_source.check,
    has_fourier.check,
    has_cap.check,
    has_comparison.check,
    has_checkpoint.check,
    has_files.check,
    has_np_abs_cross_sec.check,
    has_core_hole.check,
    has_scatter_response_fxn.check,
]

# Single sequential formation port (driver + Meep objects + molecule + fourier paths).
FORM_PIPELINE = [
    form_all_mod.form_all,
]

__all__ = [
    "CHECK_PIPELINE",
    "FORM_PIPELINE",
    "check_spatial_symmetries",
]
