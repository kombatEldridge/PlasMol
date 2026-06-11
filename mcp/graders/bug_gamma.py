from __future__ import annotations

import re

from graders.common import replace_matching_line, verify_category, VerifyResult

BROKEN_RE = re.compile(
    r"gamma\[i\]\s*=\s*min\(clamp,\s*gam0\s*\*\s*np\.exp\(xi\s*\*\s*e_tilde\)\)"
)
FIXED_RE = re.compile(
    r"gamma\[i\]\s*=\s*min\(clamp,\s*gam0\s*\*\s*\(np\.exp\(xi\s*\*\s*e_tilde\)\s*-\s*1\.0\)\)"
)


def is_broken_line(line: str) -> bool:
    return bool(BROKEN_RE.search(line)) and "- 1.0" not in line


def is_fixed_line(line: str) -> bool:
    return bool(FIXED_RE.search(line))


def apply_fix(source: str, broken: str, fixed: str) -> str | None:
    if not is_broken_line(broken) or not is_fixed_line(fixed):
        return None
    return replace_matching_line(source, broken, fixed)


def outputs_match(restored: str, canonical: str) -> bool:
    return bool(FIXED_RE.search(restored)) and not is_broken_line(restored)


def pair_covers_edit(broken: str, fixed: str, edit: dict) -> bool:
    return is_broken_line(broken) and is_fixed_line(fixed) and "gamma[i]" in edit["old"]


def verify(
    submission_pairs: list[tuple[str, str]], truth_edits: list[dict]
) -> VerifyResult:
    return verify_category(
        submission_pairs,
        truth_edits,
        is_broken_line=is_broken_line,
        is_fixed_line=is_fixed_line,
        apply_fix=apply_fix,
        outputs_match=outputs_match,
        pair_covers_edit=pair_covers_edit,
    )