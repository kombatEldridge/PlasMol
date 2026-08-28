# Plane-wave source face helpers: ensure k ⊥ E.
import logging

import numpy as np

from plasmol.drivers.custom_drivers.fourier._util import as_xyz_array

logger = logging.getLogger("main")


def source_face_normal_index(size, atol=1e-12):
    """
    Infer plane-wave face normal (propagation) index from Meep source size.

    For a planar source, exactly one size component is zero; that axis is the
    face normal (and preferred k direction). Returns None if the size is not
    unambiguously planar.
    """
    size = np.asarray(size, dtype=float).reshape(-1)
    if size.size != 3:
        raise ValueError(f"Source size must be length 3; got {size.shape}.")
    zero = np.flatnonzero(np.abs(size) <= atol)
    if zero.size == 1:
        return int(zero[0])
    if zero.size == 0:
        imin = int(np.argmin(np.abs(size)))
        smax = float(np.max(np.abs(size)))
        if smax > 0 and abs(size[imin]) < 0.05 * smax:
            return imin
    return None


def pick_propagation_axis(e_idx, candidates, center, current_k_idx=None, prefer_k=None):
    """
    Choose propagation axis index from ``candidates`` (two axes ⊥ E).

    Preference order:
      1. ``prefer_k`` if it is a candidate;
      2. existing face normal if already a candidate;
      3. axis of largest |source_center| among candidates;
      4. first of candidates in order x, y, z.
    """
    cand = list(candidates)
    if prefer_k is not None:
        pref = prefer_k if isinstance(prefer_k, int) else 'xyz'.index(str(prefer_k).lower())
        if pref in cand:
            return pref
    if current_k_idx is not None and current_k_idx in cand:
        return current_k_idx
    center = np.asarray(center, dtype=float).reshape(3)
    by_offset = sorted(cand, key=lambda i: -abs(center[i]))
    if abs(center[by_offset[0]]) > 1e-15:
        return by_offset[0]
    for i in (0, 1, 2):
        if i in cand:
            return i
    return cand[0]


def rearrange_vector_for_new_normal(vec, old_k, new_k):
    """
    Move the face-normal slot from ``old_k`` to ``new_k`` by swapping components.

    No new lengths are invented: e.g. size [0, 0.2, 0.2] with old_k=x, new_k=y
    becomes [0.2, 0, 0.2]; center [-0.04, 0, 0] becomes [0, -0.04, 0].
    """
    out = np.asarray(vec, dtype=float).reshape(3).copy()
    if old_k == new_k:
        return out
    out[old_k], out[new_k] = out[new_k], out[old_k]
    return out


def ensure_transverse_plane_wave_source(params, component=None, prefer_k=None, size_atol=1e-12):
    """
    Ensure the Meep source is a planar face with normal k perpendicular to E.

    * If the user size is planar and the zero-size axis is already ⊥ to
      ``component``, leave center and size unchanged.
    * Otherwise, pick a new normal among the two axes ⊥ E and **rearrange**
      the existing size/center vectors (swap the old-normal and new-normal
      slots). Assumes a cubic cell and a planar user source such as
      ``size = [0, L, L]`` — no aperture widths are recomputed from the cell.
    """
    if component is None:
        component = getattr(params, 'plasmon_source_component', None)
    if component is None:
        raise ValueError("ensure_transverse_plane_wave_source requires a source component.")
    component = str(component).lower().strip()
    if component not in ('x', 'y', 'z'):
        raise ValueError(f"Invalid source component '{component}'.")

    e_idx = 'xyz'.index(component)
    size = as_xyz_array(getattr(params, 'plasmon_source_size', [0, 0, 0]))
    center = as_xyz_array(getattr(params, 'plasmon_source_center', [0, 0, 0]))
    k_idx = source_face_normal_index(size, atol=size_atol)

    if k_idx is not None and k_idx != e_idx:
        k_comp = 'xyz'[k_idx]
        logger.info(
            f"Plane-wave source face already transverse: E || {component}, "
            f"k || {k_comp} (size={size.tolist()}, center={center.tolist()})."
        )
        params.plasmon_source_component = component
        params.plasmon_source_size = size.tolist()
        params.plasmon_source_center = center.tolist()
        return {
            'component': component,
            'k_component': k_comp,
            'center': center.tolist(),
            'size': size.tolist(),
            'modified': False,
            'kept': True,
        }

    candidates = [i for i in range(3) if i != e_idx]
    if k_idx is not None and k_idx == e_idx:
        logger.warning(
            f"Source face normal is parallel to E || {component} (longitudinal); "
            f"rearranging plane-wave face toward candidates "
            f"{['xyz'[i] for i in candidates]}."
        )
    else:
        raise ValueError(
            f"Source size {size.tolist()} is not a clear planar face "
            f"(need one zero component, e.g. [0, L, L]). Cannot rearrange for "
            f"E || {component} without inventing aperture lengths."
        )

    new_k = pick_propagation_axis(
        e_idx, candidates, center, current_k_idx=k_idx, prefer_k=prefer_k
    )
    k_comp = 'xyz'[new_k]

    new_size = rearrange_vector_for_new_normal(size, k_idx, new_k)
    new_center = rearrange_vector_for_new_normal(center, k_idx, new_k)

    # Sanity: after rearrange, normal slot should be the zero size entry.
    if abs(new_size[new_k]) > size_atol:
        raise ValueError(
            f"After rearranging source size {size.tolist()} from k='{'xyz'[k_idx]}' "
            f"to k='{k_comp}', size[{k_comp}]={new_size[new_k]} is not zero. "
            f"Provide a planar source with one zero component."
        )

    params.plasmon_source_component = component
    params.plasmon_source_size = new_size.tolist()
    params.plasmon_source_center = new_center.tolist()
    logger.info(
        f"Rearranged plane-wave source: E || {component}, k || {k_comp}, "
        f"size {size.tolist()} → {new_size.tolist()}, "
        f"center {center.tolist()} → {new_center.tolist()}."
    )
    return {
        'component': component,
        'k_component': k_comp,
        'center': new_center.tolist(),
        'size': new_size.tolist(),
        'modified': True,
        'kept': False,
    }
