from __future__ import annotations

from graders import (
    bug_absorption,
    bug_dt_meep,
    bug_fourier_damp,
    bug_gamma,
    bug_intensity_au,
    bug_mf_omega,
    bug_t_end_meep,
    no_fault,
)
from graders.common import VerifyResult

VERIFIERS = {
    "no_fault": no_fault.verify,
    "bug_intensity_au": bug_intensity_au.verify,
    "bug_dt_meep": bug_dt_meep.verify,
    "bug_t_end_meep": bug_t_end_meep.verify,
    "bug_fourier_damp": bug_fourier_damp.verify,
    "bug_gamma": bug_gamma.verify,
    "bug_absorption": bug_absorption.verify,
    "bug_mf_omega": bug_mf_omega.verify,
}


def verify_lines(
    category: str, submission_pairs: list[tuple[str, str]], truth_edits: list[dict]
) -> VerifyResult:
    verifier = VERIFIERS.get(category)
    if verifier is None:
        return VerifyResult(
            lines_correct=False,
            reason=f"no verifier registered for category: {category}",
        )
    return verifier(submission_pairs, truth_edits)


__all__ = ["VERIFIERS", "VerifyResult", "verify_lines"]