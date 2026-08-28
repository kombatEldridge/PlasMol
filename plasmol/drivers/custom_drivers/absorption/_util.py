# Shared small utilities for the Fourier driver package.
import numpy as np


def as_xyz_array(vec):
    """Convert list / ndarray / meep.Vector3 → (3,) float array."""
    if vec is None:
        raise ValueError("Expected a 3-vector, got None.")
    if hasattr(vec, 'x') and hasattr(vec, 'y') and hasattr(vec, 'z'):
        return np.array([float(vec.x), float(vec.y), float(vec.z)], dtype=float)
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"Expected a length-3 vector, got shape {arr.shape}.")
    return arr
