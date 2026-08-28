# Parameter-copy builders for directional absorption jobs.
import copy
import os

from plasmol.quantum.sources import QUANTUMSOURCE


def make_plasmol_direction_copy(params, component, flat=False):
    """One production plasmol params copy for a single source polarization.

    Source-face rearrange (k ⊥ E) is deferred to the Meep worker so the
    log lines carry the ``[x-dir]`` / ``[y-dir]`` / ``[z-dir]`` prefix.

    When ``flat`` is True (parallel / perpendicular), CSVs stay in the job
    directory. Full x+y+z mode still uses ``{component}_dir/``.
    """
    params_copy = copy.deepcopy(params)
    params_copy.plasmon_source_component = component
    if flat:
        params_copy.dir_path = ""
        params_copy.field_e_filepath = getattr(params_copy, 'field_e_filepath', 'field_e.csv')
        params_copy.field_p_filepath = getattr(params_copy, 'field_p_filepath', 'field_p.csv')
        params_copy.spectra_e_vs_p_filepath = getattr(
            params_copy, 'spectra_e_vs_p_filepath', 'output.png'
        )
    else:
        params_copy.dir_path = f"{component}_dir"
        params_copy.field_e_filepath = getattr(params_copy, f'field_e_{component}_filepath')
        params_copy.field_p_filepath = getattr(params_copy, f'field_p_{component}_filepath')
        params_copy.spectra_e_vs_p_filepath = getattr(
            params_copy, f'spectra_e_{component}_vs_p_{component}_filepath'
        )
        os.makedirs(params_copy.dir_path, exist_ok=True)
    return params_copy


def make_reference_direction_copy(params, component, flat=False):
    """One vacuum reference params copy for a single source polarization.

    Source-face rearrange (k ⊥ E) is deferred to the Meep worker so the
    log lines carry the ``[ref-x-dir]`` (etc.) prefix.

    When ``flat`` is True (parallel / perpendicular), the raw vacuum E CSV
    is written in the job directory rather than ``{component}_dir/``.
    """
    if not getattr(params, 'has_molecule_position', False):
        raise ValueError(
            "Meep absorption reference runs require plasmon.molecule.position "
            "(location at which to sample the vacuum incident field)."
        )
    params_copy = copy.deepcopy(params)
    params_copy.plasmon_source_component = component
    if flat:
        params_copy.dir_path = ""
        params_copy.field_e_filepath = "field_e_ref.csv"
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
        params_copies.append(params_copy)
    return params_copies
