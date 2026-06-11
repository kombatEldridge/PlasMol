from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

PLACEHOLDER = "placeholder"

BrokenCheck = Callable[[str], bool]
FixedCheck = Callable[[str], bool]
ApplyFix = Callable[[str, str, str], str | None]
OutputCheck = Callable[[str, str], bool]
PairCoversEdit = Callable[[str, str, dict], bool]


@dataclass
class VerifyResult:
    lines_correct: bool
    matched_edits: list[dict] = field(default_factory=list)
    unmatched_truth_edits: list[dict] = field(default_factory=list)
    unmatched_submission_pairs: list[tuple[str, str]] = field(default_factory=list)
    reason: str | None = None


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def compact_line(line: str) -> str:
    return re.sub(r"\s+", "", line.strip())


def substantive_edits(truth_edits: list[dict]) -> list[dict]:
    return [
        edit
        for edit in truth_edits
        if PLACEHOLDER not in edit["old"] and PLACEHOLDER not in edit["new"]
    ]


def build_canonical_source(truth_edits: list[dict]) -> str:
    return "\n".join(edit["old"] for edit in substantive_edits(truth_edits))


def apply_truth_bug(canonical: str, truth_edits: list[dict]) -> str:
    buggy = canonical
    for edit in substantive_edits(truth_edits):
        if edit["old"] not in buggy:
            raise ValueError(f"truth edit not found in canonical source: {edit['old']}")
        buggy = buggy.replace(edit["old"], edit["new"], 1)
    return buggy


def replace_matching_line(source: str, broken: str, fixed: str) -> str | None:
    if broken in source:
        return source.replace(broken, fixed, 1)

    broken_norm = normalize_line(broken)
    fixed_norm = normalize_line(fixed)
    broken_compact = compact_line(broken)

    lines = source.splitlines()
    for idx, line in enumerate(lines):
        if (
            normalize_line(line) == broken_norm
            or compact_line(line) == broken_compact
        ):
            lines[idx] = fixed_norm
            return "\n".join(lines)
    return None


def verify_category(
    submission_pairs: list[tuple[str, str]],
    truth_edits: list[dict],
    *,
    is_broken_line: BrokenCheck,
    is_fixed_line: FixedCheck,
    apply_fix: ApplyFix,
    outputs_match: OutputCheck,
    pair_covers_edit: PairCoversEdit,
) -> VerifyResult:
    """Simulate truth injection, apply submitted fixes, compare to canonical output."""
    required = substantive_edits(truth_edits)
    if not required:
        ok = not submission_pairs
        return VerifyResult(
            lines_correct=ok,
            reason=None if ok else "no_fault submissions must have empty line pairs",
            unmatched_submission_pairs=list(submission_pairs),
        )

    canonical = build_canonical_source(truth_edits)
    try:
        restored = apply_truth_bug(canonical, truth_edits)
    except ValueError as exc:
        return VerifyResult(lines_correct=False, reason=str(exc))

    matched_edits: list[dict] = []
    unmatched_pairs: list[tuple[str, str]] = []

    for broken, fixed in submission_pairs:
        if not is_broken_line(broken) or not is_fixed_line(fixed):
            unmatched_pairs.append((broken, fixed))
            continue

        next_restored = apply_fix(restored, broken, fixed)
        if next_restored is None:
            unmatched_pairs.append((broken, fixed))
            continue

        restored = next_restored
        for edit in required:
            if edit in matched_edits:
                continue
            if pair_covers_edit(broken, fixed, edit):
                matched_edits.append(edit)

    lines_correct = (
        outputs_match(restored, canonical)
        and not unmatched_pairs
        and len(matched_edits) == len(required)
    )

    return VerifyResult(
        lines_correct=lines_correct,
        matched_edits=matched_edits if lines_correct else [],
        unmatched_truth_edits=[
            edit for edit in required if edit not in matched_edits
        ],
        unmatched_submission_pairs=unmatched_pairs,
        reason=None
        if lines_correct
        else "submitted fix does not restore canonical source output",
    )