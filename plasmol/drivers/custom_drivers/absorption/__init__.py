"""
Fourier absorption-spectrum driver package.

Public entry point: ``run(params)``.

Submodules
----------
- ``driver``        – orchestration
- ``workers``       – parallel Meep / RT-TDDFT workers
- ``setup``         – per-direction params copies
- ``polarization``  – parallel / perpendicular NP–mol modes
- ``source_face``   – plane-wave k ⊥ E geometry
- ``io_fields``     – CSV fold / damp / reference merge
- ``spectrum``      – FFT, deconvolution, absorption, plots
- ``postprocess``   – post-run spectrum assembly
"""

from plasmol.drivers.custom_drivers.fourier.driver import run

# Re-export helpers used by tests and external callers.
from plasmol.drivers.custom_drivers.fourier.io_fields import (
    apply_damping,
    fold,
    fold_single,
    load_dipole_from_csv,
    load_reference_e_tensor,
    merge_reference_e_fields,
    validate_reference_times,
    write_single_reference_e_field,
)
from plasmol.drivers.custom_drivers.fourier.polarization import (
    build_parallel_abs_spec_runs,
    build_perpendicular_abs_spec_runs,
    np_mol_axis_vector,
    resolve_parallel_component,
    resolve_perpendicular_component,
)
from plasmol.drivers.custom_drivers.fourier.source_face import (
    ensure_transverse_plane_wave_source,
    source_face_normal_index,
)
from plasmol.drivers.custom_drivers.fourier.spectrum import (
    absorption,
    absorption_single,
    fourier,
    orient_spectrum_sign,
)
from plasmol.drivers.custom_drivers.fourier.postprocess import (
    fourier_post_process,
    fourier_post_process_single,
)

__all__ = [
    "run",
    "fourier",
    "absorption",
    "absorption_single",
    "orient_spectrum_sign",
    "fold",
    "fold_single",
    "apply_damping",
    "load_dipole_from_csv",
    "load_reference_e_tensor",
    "merge_reference_e_fields",
    "validate_reference_times",
    "write_single_reference_e_field",
    "np_mol_axis_vector",
    "resolve_parallel_component",
    "resolve_perpendicular_component",
    "build_parallel_abs_spec_runs",
    "build_perpendicular_abs_spec_runs",
    "ensure_transverse_plane_wave_source",
    "source_face_normal_index",
    "fourier_post_process",
    "fourier_post_process_single",
]
