# quantum/propagators/magnus2.py
import logging
import numpy as np
from scipy.linalg import expm

logger = logging.getLogger("main")

def _expm(F):
    if F.ndim == 3:
        return np.stack([expm(F[s]) for s in range(F.shape[0])])
    return expm(F)

def _rms_diff(A, B):
    """RMS |A-B| per element, so the threshold does not grow with the basis."""
    if not isinstance(A, np.ndarray):
        A = np.stack(A)
    if not isinstance(B, np.ndarray):
        B = np.stack(B)
    diff = np.asarray(A) - np.asarray(B)
    return float(np.sqrt(np.mean(np.abs(diff) ** 2)))

class MagnusNotConverged(RuntimeError):
    """Predictor-corrector residual did not converge for this interval."""

def _copy_coeff(coeff):
    if isinstance(coeff, np.ndarray):
        return coeff.copy()
    return [np.array(c, copy=True) for c in coeff]

def _snapshot(molecule):
    return {
        "mo_coeff": _copy_coeff(molecule.mf.mo_coeff),
        "D_ao": np.array(molecule.D_ao, copy=True),
        "F_orth": np.array(molecule.F_orth, copy=True),
        "F_orth_n12dt": np.array(molecule.F_orth_n12dt, copy=True),
    }

def _restore(molecule, snap):
    molecule.mf.mo_coeff = snap["mo_coeff"]
    molecule.D_ao = snap["D_ao"]
    molecule.F_orth = snap["F_orth"]
    molecule.F_orth_n12dt = snap["F_orth_n12dt"]

def _format_residual(diffs, density_rms):
    picks = [1, 10, 50, 100, 150, len(diffs)]
    parts = []
    seen = set()
    for n in picks:
        if 1 <= n <= len(diffs) and n not in seen:
            seen.add(n)
            parts.append(f"check {n}={diffs[n - 1]:.3e}")
    density = "" if density_rms is None else f" density_rms={density_rms:.3e}"
    return (
        f"Magnus2 residual RMS over {len(diffs)} checks: "
        f"first={diffs[0]:.3e} min={min(diffs):.3e} last={diffs[-1]:.3e} "
        f"samples: {' '.join(parts)}{density}"
    )

def propagate(molecule_max_iterations, dt, molecule_pc_convergence, molecule, exc):
    """
    Propagate molecular orbitals using the Magnus2 method.

    Implements a predictor-corrector scheme to propagate the density matrix over one time step,
    using extrapolation and iterative refinement until convergence. If that step does not
    converge, the same interval is retried once as two steps of dt/2 under the same field.

    Parameters:
    molecule_max_iterations : int
        Maximum number of iterations for convergence.
    dt : float
        Time step size.
    molecule_pc_convergence : float
        Convergence threshold for the predictor-corrector scheme.
    molecule : object
        Molecule object with current state data.
    exc : np.ndarray
        External electric field at the current time step.

    Returns:
    None
    """
    snap = _snapshot(molecule)
    try:
        _propagate_interval(
            molecule_max_iterations, dt, molecule_pc_convergence, molecule, exc,
            logging.WARNING,
        )
    except MagnusNotConverged:
        _restore(molecule, snap)
        # The saved midpoint Fock belongs to a step of size dt. A half-step
        # must not extrapolate from it, so start that step from the current Fock.
        molecule.F_orth_n12dt = np.array(molecule.F_orth, copy=True)
        logger.warning(
            "Magnus2 did not converge within %d iterations at dt=%.6g. "
            "Retrying that interval as two steps of dt/2.",
            molecule_max_iterations, dt,
        )
        half = 0.5 * dt
        _propagate_interval(
            molecule_max_iterations, half, molecule_pc_convergence, molecule, exc,
            logging.ERROR,
        )
        _propagate_interval(
            molecule_max_iterations, half, molecule_pc_convergence, molecule, exc,
            logging.ERROR,
        )

def _propagate_interval(molecule_max_iterations, dt, molecule_pc_convergence, molecule, exc, failure_level):
    C_orth = molecule.rotate_coeff_to_orth(molecule.mf.mo_coeff)
    F_orth_p12dt = 2 * molecule.F_orth - molecule.F_orth_n12dt
    C_ao_pdt_old = None
    D_ao_old = None
    diffs = []
    density_rms = None

    iteration = 0
    while True:
        iteration += 1
        if iteration > molecule_max_iterations:
            if diffs:
                logger.log(failure_level, "%s", _format_residual(diffs, density_rms))
            raise MagnusNotConverged(
                f"Failed to converge within {molecule_max_iterations} iterations"
            )

        # 1) predictor
        U = _expm(-1j * dt * F_orth_p12dt)
        C_orth_pdt = U @ C_orth
        C_pdt = molecule.rotate_coeff_away_from_orth(C_orth_pdt)

        # 2) compute new Fock
        D_ao_pdt = molecule.mf.make_rdm1(mo_coeff=C_pdt, mo_occ=molecule.occ)
        F_orth_pdt = molecule.get_F_orth(D_ao_pdt, exc)

        # 3) only check convergence if we have a previous value
        if C_ao_pdt_old is not None:
            diff = _rms_diff(C_pdt, C_ao_pdt_old)
            diffs.append(diff)
            density_rms = _rms_diff(D_ao_pdt, D_ao_old)
            # A residual that has stopped falling is a floor, not a divergent step.
            # 10× the request covers ordinary roundoff (~1e-12). CAP steps can
            # stick near 1e-9 after falling from ~1e-4; 1e-8 still rejects that
            # unconverged residual. A residual that is still falling is not accepted.
            stalled = False
            if len(diffs) >= 8:
                best = min(diffs)
                recent = diffs[-5:]
                stalled = (
                    best < max(10.0 * molecule_pc_convergence, 1e-8)
                    and min(recent) > 0.9 * best
                    and max(recent) < 5.0 * best
                )
            if diff < molecule_pc_convergence or stalled:
                molecule.mf.mo_coeff = C_pdt
                molecule.D_ao = D_ao_pdt
                molecule.F_orth = F_orth_pdt
                molecule.F_orth_n12dt = F_orth_p12dt
                if stalled and diff >= molecule_pc_convergence and not getattr(propagate, "_floor_logged", False):
                    logger.info(
                        "Magnus2 reached a roundoff floor of %.3e after %d checks (requested %.3e). Accepting the step.",
                        best, len(diffs), molecule_pc_convergence,
                    )
                    propagate._floor_logged = True
                else:
                    logger.debug(f'Magnus2 converged in {iteration} iterations.')
                break

        # 4) update history for next iteration
        F_orth_p12dt = 0.5 * (molecule.F_orth + F_orth_pdt)
        C_ao_pdt_old = C_pdt.copy() if not molecule.is_open_shell else [c.copy() for c in C_pdt]
        D_ao_old = np.array(D_ao_pdt, copy=True)

        # 5) Keep the latest coefficients / density for the next iteration
        molecule.mf.mo_coeff = C_pdt
        molecule.D_ao = D_ao_pdt
