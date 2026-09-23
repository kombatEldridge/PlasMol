"""D1 dipole axis and the rotation that puts it on +x."""
import numpy as np
import pytest

from plasmol.quantum.geometry import _rotation_matrix
from plasmol.quantum.orientation_check import (
    axis_angle_deg,
    rotation_onto_x,
    swing_axis,
)


def test_unsigned_axis_angle_ignores_sign():
    assert axis_angle_deg([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert axis_angle_deg([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(90.0)


def test_swing_axis_follows_the_moving_component():
    time = np.linspace(0.0, 1.0, 21)
    mu = np.column_stack([np.zeros_like(time), 2.0 * time, 0.1 * np.ones_like(time)])
    axis, rms = swing_axis(mu)
    assert abs(axis[1]) == pytest.approx(1.0, abs=1e-8)
    assert rms[1] > rms[0]
    assert rms[1] > rms[2]


def test_axis_already_on_x_needs_no_rotation():
    assert rotation_onto_x([1.0, 0.0, 0.0]) is None
    assert rotation_onto_x([-1.0, 0.0, 0.0]) is None


def test_rotation_onto_x_maps_the_swing_axis():
    source = np.array([0.0, 0.6, 0.8])
    spec = rotation_onto_x(source)
    mapped = _rotation_matrix(spec["axis"], spec["angle_deg"]) @ source
    assert abs(mapped[0]) == pytest.approx(np.linalg.norm(source))
    assert mapped[1] == pytest.approx(0.0, abs=1e-6)
    assert mapped[2] == pytest.approx(0.0, abs=1e-6)
    assert spec["angle_deg"] == pytest.approx(axis_angle_deg(source, [1.0, 0.0, 0.0]))
