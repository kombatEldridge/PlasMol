# tests/test_params.py
import pytest

def test_minimal_json_parses_cleanly(minimal_params):
    """Basic sanity check on our minimal test JSON."""
    p = minimal_params

    assert p.dt == 0.5
    assert p.t_end == 5.0
    assert p.has_molecule is True
    assert p.has_plasmon is False
    assert p.run_molecule_simulation is True
    assert p.molecule_propagator_str == "rk4"
    assert p.molecule_source_type == "kick"
    assert p.molecule_source_component == "z"

    print("Minimal JSON parsed successfully!")
    print(f"   dt = {p.dt} au, t_end = {p.t_end} au")
    print(f"   Molecule: {p.molecule_atoms}  |  Basis: {p.molecule_basis}")


def test_minimal_params_has_required_attributes(minimal_params):
    """Make sure the most important objects were created."""
    p = minimal_params
    assert hasattr(p, "times")
    assert hasattr(p, "molecule_source_field")
    assert len(p.times) > 0
    assert p.molecule_source_field.shape == (len(p.times), 3)