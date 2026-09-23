# Parameter-copy builders for directional absorption jobs.
import copy
import os

from plasmol.quantum.sources import QUANTUMSOURCE

# Hybrid NP+molecule (parallel / perpendicular / single) writes one polarization
# into this directory instead of ``x_dir`` / ``y_dir`` / ``z_dir``.
FIELDS_DIR = "fields"


def _in_fields(path, default):
    return os.path.join(FIELDS_DIR, os.path.basename(path if path else default))


def _relocate_core_hole_occ(params_copy, dest_dir):
    """Write the hole-occupation CSV next to that copy's field files."""
    if not getattr(params_copy, 'has_core_hole', False):
        return
    src = getattr(params_copy, 'core_hole_mo_occ_filepath', None)
    if not src:
        return
    params_copy.core_hole_mo_occ_filepath = os.path.join(
        dest_dir, os.path.basename(src)
    )


def make_plasmol_direction_copy(params, component, flat=False):
    """One production plasmol params copy for a single source polarization.

    Source-face rearrange (k ⊥ E) is deferred to the Meep worker so the
    log lines carry the ``[x-dir]`` / ``[y-dir]`` / ``[z-dir]`` prefix.

    When ``flat`` is True (NP + molecule: parallel / perpendicular / single),
    CSVs go in ``fields/``. Molecule-only three-kick and hybrid ``full`` (no NP)
    still use ``{component}_dir/``.
    """
    params_copy = copy.deepcopy(params)
    params_copy.plasmon_source_component = component
    if flat:
        params_copy.dir_path = FIELDS_DIR
        os.makedirs(FIELDS_DIR, exist_ok=True)
        params_copy.field_e_filepath = _in_fields(
            getattr(params_copy, 'field_e_filepath', None), 'field_e.csv'
        )
        params_copy.field_p_filepath = _in_fields(
            getattr(params_copy, 'field_p_filepath', None), 'field_p.csv'
        )
        params_copy.spectra_e_vs_p_filepath = _in_fields(
            getattr(params_copy, 'spectra_e_vs_p_filepath', None), 'output.png'
        )
    else:
        params_copy.dir_path = f"{component}_dir"
        params_copy.field_e_filepath = getattr(params_copy, f'field_e_{component}_filepath')
        params_copy.field_p_filepath = getattr(params_copy, f'field_p_{component}_filepath')
        params_copy.spectra_e_vs_p_filepath = getattr(
            params_copy, f'spectra_e_{component}_vs_p_{component}_filepath'
        )
        os.makedirs(params_copy.dir_path, exist_ok=True)
    _relocate_core_hole_occ(params_copy, params_copy.dir_path)
    return params_copy


def make_reference_direction_copy(params, component, flat=False):
    """One vacuum reference params copy for a single source polarization.

    Source-face rearrange (k ⊥ E) is deferred to the Meep worker so the
    log lines carry the ``[ref-x-dir]`` (etc.) prefix.

    When ``flat`` is True (NP + molecule), the raw vacuum E CSV is
    ``fields/field_e_ref.csv``.
    """
    if not getattr(params, 'has_molecule_position', False):
        raise ValueError(
            "Meep absorption reference runs require plasmon.molecule.position "
            "(location at which to sample the vacuum incident field)."
        )
    params_copy = copy.deepcopy(params)
    params_copy.plasmon_source_component = component
    if flat:
        params_copy.dir_path = FIELDS_DIR
        os.makedirs(FIELDS_DIR, exist_ok=True)
        params_copy.field_e_filepath = os.path.join(FIELDS_DIR, "field_e_ref.csv")
    else:
        params_copy.dir_path = f"{component}_dir"
        params_copy.field_e_filepath = f"{component}_dir/field_e_ref.csv"
        os.makedirs(params_copy.dir_path, exist_ok=True)
    params_copy.has_nanoparticle = False
    params_copy.has_molecule = False
    params_copy.record_field_only = True
    params_copy.has_images = False
    params_copy.has_checkpoint = False
    params_copy.probe_points = None
    if hasattr(params_copy, 'nanoparticle'):
        params_copy.nanoparticle = None
    return params_copy


def set_up_params_copy_plasmol(params):
    return [make_plasmol_direction_copy(params, d) for d in params.xyz]


def set_up_params_copy_reference(params):
    """
    Vacuum reference runs for Meep Fourier deconvolution.

    Same cell, source, and sample location as the production job, but with
    no nanoparticle and no quantum molecule. Writes E_inc to
    ``{x,y,z}_dir/field_e_ref.csv``.
    """
    return [make_reference_direction_copy(params, d) for d in params.xyz]


def set_up_params_copy_molecule(params):
    params_copies = []
    for d in params.xyz:
        params_copy = copy.deepcopy(params)
        params_copy.molecule_source_component = d
        params_copy.molecule_source_field = QUANTUMSOURCE(params_copy).field
        params_copy.dir_path = f"{d}_dir"
        params_copy.field_e_filepath = getattr(params_copy, f'field_e_{d}_filepath')
        params_copy.field_p_filepath = getattr(params_copy, f'field_p_{d}_filepath')
        params_copy.spectra_e_vs_p_filepath = getattr(
            params_copy, f'spectra_e_{d}_vs_p_{d}_filepath'
        )
        if not params.resumed_from_checkpoint:
            os.makedirs(params_copy.dir_path, exist_ok=True)
        _relocate_core_hole_occ(params_copy, params_copy.dir_path)
        params_copies.append(params_copy)
    return params_copies
