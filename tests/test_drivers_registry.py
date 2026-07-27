"""Driver registry smoke tests."""
import pytest
from plasmol.drivers import get_driver

EXPECTED = {
    "classical", "plasmol", "quantum", "comparison", "fourier",
    "scatter_response_fxn", "np_abs_cross_sec", "verify_source",
    "tune", "core_hole",
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_registered_driver(name):
    assert callable(get_driver(name))


def test_unknown_driver():
    with pytest.raises(ValueError, match="Unknown driver"):
        get_driver("nope")
