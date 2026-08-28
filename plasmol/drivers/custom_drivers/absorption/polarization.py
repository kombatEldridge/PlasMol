# NP–molecule axis and parallel / perpendicular polarization modes.
import logging

import numpy as np

from plasmol.drivers.custom_drivers.fourier._util import as_xyz_array
from plasmol.drivers.custom_drivers.fourier.setup import (
    make_plasmol_direction_copy,
    make_reference_direction_copy,
)

logger = logging.getLogger("main")


def np_mol_axis_vector(params):
    """
    Vector from nanoparticle center to molecule sample point (μm).

    If no nanoparticle is present, the NP center is taken as the origin.
    """
    mol = as_xyz_array(getattr(params, 'plasmol_molecule_position', None))
    if getattr(params, 'has_nanoparticle', False) and getattr(params, 'nanoparticle_center', None) is not None:
        center = as_xyz_array(params.nanoparticle_center)
    else:
        center = np.zeros(3, dtype=float)
    axis = mol - center
    norm = float(np.linalg.norm(axis))
    if norm < 1e-15:
        raise ValueError(
            "NP–molecule axis is zero-length (molecule sits at the nanoparticle "
            "center / origin). Displace plasmon.molecule.position for "
            "parallel/perpendicular Fourier modes."
        )
    return axis, norm


def resolve_parallel_component(params):
    """Cartesian Meep source component nearest to the NP→molecule axis."""
    axis, norm = np_mol_axis_vector(params)
    unit = axis / norm
    idx = int(np.argmax(np.abs(unit)))
    component = params.xyz[idx]
    alignment = abs(unit[idx])
    if alignment < 0.9:
        logger.warning(
            f"NP–molecule axis {axis} (μm) is not well aligned with a Cartesian "
            f"axis (max |û·ê|={alignment:.3f}). Using parallel component '{component}'. "
            f"Place the molecule on a Cartesian axis for a cleaner ∥ spectrum."
        )
    else:
        logger.info(
            f"Parallel polarization: NP–molecule axis {axis} (μm) → "
            f"E component '{component}' (|û·ê|={alignment:.3f})."
        )
    return component


def resolve_perpendicular_component(params):
    """
    Cartesian Meep source component perpendicular to the NP→molecule axis.

    Preference: explicit fourier_perp_component, else most orthogonal Cartesian.
    """
    axis, norm = np_mol_axis_vector(params)
    unit = axis / norm
    user = getattr(params, 'fourier_perp_component', None)
    if user:
        user = user.lower().strip()
        if user not in params.xyz:
            raise ValueError(f"Invalid fourier_perp_component '{user}'.")
        uidx = params.xyz.index(user)
        if abs(unit[uidx]) > 0.5:
            raise ValueError(
                f"Requested perp_component '{user}' is not perpendicular to the "
                f"NP–molecule axis {axis} (μm) (|û·ê|={abs(unit[uidx]):.3f}). "
                f"Choose a different component or omit perp_component for auto selection."
            )
        logger.info(
            f"Perpendicular polarization: using user perp_component '{user}' "
            f"(NP–molecule axis {axis} μm)."
        )
        return user

    scores = [(abs(unit[i]), i) for i in range(3)]
    preference = {1: 0, 2: 1, 0: 2}
    scores.sort(key=lambda t: (t[0], preference[t[1]]))
    idx = scores[0][1]
    component = params.xyz[idx]
    logger.info(
        f"Perpendicular polarization: NP–molecule axis {axis} (μm) → "
        f"E component '{component}' (|û·ê|={abs(unit[idx]):.3f})."
    )
    return component


def build_parallel_abs_spec_runs(params):
    """
    Build production (+ optional vacuum reference) params for a parallel abs spectrum.

    Returns (params_copies, ref_copies, component).
    """
    component = resolve_parallel_component(params)
    params.fourier_active_component = component
    params_copies = [make_plasmol_direction_copy(params, component)]
    ref_copies = []
    if not getattr(params, 'fourier_use_existing_e_field_ref', False):
        ref_copies = [make_reference_direction_copy(params, component)]
    logger.info(
        f"Parallel abs spectrum: 1 production run + {len(ref_copies)} vacuum "
        f"reference run(s) with E || '{component}'."
    )
    return params_copies, ref_copies, component


def build_perpendicular_abs_spec_runs(params):
    """
    Build production (+ optional vacuum reference) params for a perpendicular abs spectrum.

    Returns (params_copies, ref_copies, component).
    """
    component = resolve_perpendicular_component(params)
    params.fourier_active_component = component
    params_copies = [make_plasmol_direction_copy(params, component)]
    ref_copies = []
    if not getattr(params, 'fourier_use_existing_e_field_ref', False):
        ref_copies = [make_reference_direction_copy(params, component)]
    logger.info(
        f"Perpendicular abs spectrum: 1 production run + {len(ref_copies)} vacuum "
        f"reference run(s) with E || '{component}' (⊥ NP–mol axis)."
    )
    return params_copies, ref_copies, component
