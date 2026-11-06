# quantum/resume_molecule.py
"""Restore the molecule state a checkpoint left behind.

The checkpoint saves the orbitals after a step and labels them with the time
that step started. It does not save ``F_orth``. The following step propagates
with the Fock of that restored density and the field at the label,
``E(checkpoint_time)``, not the field one step later. Magnus keeps the saved
midpoint Fock, and the step propagator keeps the saved ``C_orth_ndt``. A
direction an absorption checkpoint never reached is built as a fresh start.
"""
import os
import logging
import numpy as np

from plasmol.utils.csv import init_csv

logger = logging.getLogger("main")


def _checkpoint_suffix(molecule):
    if not getattr(molecule, "has_absorption", False):
        return ""
    direction = getattr(molecule, "molecule_source_component", None)
    if not direction:
        return ""
    return f"_{direction}"


def _direction_has_checkpoint(molecule):
    """False for a new run, and for an absorption direction with no saved state."""
    if not molecule.resumed_from_checkpoint:
        return False
    direction = None
    if getattr(molecule, "has_absorption", False):
        direction = getattr(molecule, "molecule_source_component", None)
    skipped = getattr(molecule, "not_checkpointed_dirs", ())
    return direction not in skipped


def _field_at_checkpoint(molecule, suffix):
    """Field sample at the checkpoint label. ``molecule_source_field`` matches ``times``."""
    checkpoint_time = float(molecule.values_from_checkpoint[f"checkpoint_time{suffix}"])
    times = np.asarray(molecule.times, dtype=float)
    matches = np.flatnonzero(np.isclose(times, checkpoint_time, rtol=0.0, atol=1e-12))
    if matches.size != 1:
        raise ValueError(
            f"Checkpoint time {checkpoint_time} au is not one sample on the time grid "
            f"({times[0]} .. {times[-1]} au, {times.size} samples)."
        )
    return np.asarray(molecule.molecule_source_field[int(matches[0])])


def initialize_density_and_fock(molecule):
    """Set ``D_ao_0``, ``D_ao``, ``F_orth``, and the propagator extras.

    On resume, ``F_orth`` includes ``E(checkpoint_time)``. The midpoint Fock and
    the step-propagator coefficients are copied from the checkpoint, not rebuilt.
    A fresh run builds a field-free Fock: no step has stored a field yet.
    """
    if _direction_has_checkpoint(molecule):
        suffix = _checkpoint_suffix(molecule)
        saved = molecule.values_from_checkpoint
        molecule.D_ao_0 = saved[f"D_ao_0{suffix}"]
        molecule.mf.mo_coeff = saved[f"mo_coeff{suffix}"]
        molecule.D_ao = molecule.mf.make_rdm1(mo_occ=molecule.occ)
        # The finished step stored F[D(t+dt), E(t)] and the checkpoint labeled it t.
        # The next step propagates with that Fock, so exc is E(checkpoint_time).
        molecule.F_orth = molecule.get_F_orth(molecule.D_ao, _field_at_checkpoint(molecule, suffix))
        propagator = molecule.molecule_propagator_str
        if propagator == "magnus2":
            molecule.F_orth_n12dt = saved[f"F_orth_n12dt{suffix}"]
        elif propagator == "step":
            molecule.C_orth_ndt = saved[f"C_orth_ndt{suffix}"]
        return

    if (molecule.has_core_hole and getattr(molecule, "_core_hole_dm0", None) is not None):
        molecule.D_ao_0 = np.asarray(molecule._core_hole_dm0, dtype=np.complex128)
        molecule.D_ao = molecule.D_ao_0.copy()
    else:
        molecule.D_ao_0 = molecule.mf.make_rdm1(mo_occ=molecule.occ)
        molecule.D_ao = molecule.D_ao_0

    molecule.F_orth = molecule.get_F_orth(molecule.D_ao)
    if molecule.molecule_propagator_str == "magnus2":
        molecule.F_orth_n12dt = molecule.F_orth
    elif molecule.molecule_propagator_str == "step":
        molecule.C_orth_ndt = molecule.rotate_coeff_to_orth(molecule.mf.mo_coeff)


def open_mo_occ_csv(molecule, filepath):
    """Keep a restored occupation CSV, or start one.

    A resumed run whose CSV was not in the checkpoint starts a new file and warns.
    """
    if molecule.resumed_from_checkpoint and os.path.exists(filepath):
        logger.debug(f"Resuming core-hole MO occupation tracking from existing {filepath}")
        return

    indices = molecule.core_hole_log_indices
    header = ["Timestamps (au)"] + [f"MO index {i}" for i in indices]
    init_csv(filepath, f"Time-dependent hole occupations (neutral MO basis) for MO indices: " f"{indices}", header=header,)
    if molecule.resumed_from_checkpoint:
        logger.warning("Resumed core-hole run but no mo_occ CSV was restored from the checkpoint; started a fresh occupation file at current time.")
    else:
        logger.debug(f"Core-hole MO occupation file initialized: {filepath}")
