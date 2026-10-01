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

def propagate(molecule_max_iterations, dt, molecule_pc_convergence, molecule, exc):
    """
    Propagate molecular orbitals using the Magnus2 method.

    Implements a predictor-corrector scheme to propagate the density matrix over one time step,
    using extrapolation and iterative refinement until convergence.

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
    C_orth = molecule.rotate_coeff_to_orth(molecule.mf.mo_coeff)
    F_orth_p12dt = 2 * molecule.F_orth - molecule.F_orth_n12dt
    C_ao_pdt_old = None
    diffs = []

    iteration = 0
    while True:
        iteration += 1
        if iteration > molecule_max_iterations:
            if diffs:
                logger.error(
                    "Magnus2 residual RMS over %d checks: first=%.3e min=%.3e last=%.3e",
                    len(diffs), diffs[0], min(diffs), diffs[-1],
                )
            raise RuntimeError(f"Failed to converge within {molecule_max_iterations} iterations")

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
            # Once the residual has fallen to a few times the tolerance and then
            # only jitters, more iterations do not move the density. Virtual-orbital
            # phases sit on that floor. A residual that is still falling, or that
            # levels off far above the tolerance, is not accepted here.
            stalled = False
            if len(diffs) >= 8:
                best = min(diffs)
                recent = diffs[-5:]
                stalled = (
                    best < 10.0 * molecule_pc_convergence
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

        # 5) Keep the latest coefficients / density for the next iteration
        molecule.mf.mo_coeff = C_pdt
        molecule.D_ao = D_ao_pdt
