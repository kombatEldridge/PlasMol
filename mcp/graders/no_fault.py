from __future__ import annotations

from graders.common import VerifyResult


def verify(
    submission_pairs: list[tuple[str, str]], truth_edits: list[dict]
) -> VerifyResult:
    ok = not submission_pairs and not truth_edits
    return VerifyResult(
        lines_correct=ok,
        reason=None if ok else "no_fault submissions must have empty line pairs",
        unmatched_submission_pairs=list(submission_pairs),
    )