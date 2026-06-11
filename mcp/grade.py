#!/usr/bin/env python3
"""Grade a PlasMol bug-diagnosis submission.

Inputs (env-overridable for testing):
  SUBMISSION = submission.json
  TRUTH      = injected.json

Scoring:
  category_correct : submitted bug_category == injected category
  lines_correct    : category-specific verifier confirms submitted broken/fixed
                     line pairs restore the same canonical source output as the
                     injected truth edits
  score = 1.0 if both ; 0.5 if category only ; 0.0 otherwise
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_MCP_DIR = Path(__file__).resolve().parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

from graders import verify_lines
from graders.common import substantive_edits
from inject import BUG_CATEGORIES

DEFAULT_SUBMISSION = "submission.json"
DEFAULT_TRUTH = "injected.json"


def normalize_submission(sub: dict) -> tuple[list[str], list[str], str | None]:
    broken = sub.get("broken_lines", [])
    fixed = sub.get("fixed_lines", [])

    if isinstance(broken, str):
        broken = [broken] if broken else []
    if isinstance(fixed, str):
        fixed = [fixed] if fixed else []

    if not isinstance(broken, list) or not isinstance(fixed, list):
        return [], [], "broken_lines and fixed_lines must be strings or lists of strings"
    if not all(isinstance(line, str) for line in broken + fixed):
        return [], [], "broken_lines and fixed_lines must contain only strings"
    if len(broken) != len(fixed):
        return [], [], "broken_lines and fixed_lines must have the same length"

    return broken, fixed, None


def truth_pairs(truth: dict) -> list[dict]:
    return [
        {"broken": edit["new"], "fixed": edit["old"]}
        for edit in substantive_edits(truth.get("edits", []))
    ]


def grade_submission(truth: dict, sub: dict) -> dict:
    cat = sub.get("bug_category")
    verdict: dict = {
        "injected_category": truth.get("category"),
        "submitted_category": cat,
        "truth_pairs": truth_pairs(truth),
    }

    if cat not in BUG_CATEGORIES:
        verdict.update(
            score=0.0,
            category_correct=False,
            lines_correct=False,
            reason="invalid bug_category",
        )
        return verdict

    broken, fixed, parse_error = normalize_submission(sub)
    if parse_error:
        verdict.update(
            score=0.0,
            category_correct=False,
            lines_correct=False,
            reason=parse_error,
        )
        return verdict

    category_correct = cat == truth["category"]
    submission_pairs = list(zip(broken, fixed))

    if category_correct:
        line_result = verify_lines(cat, submission_pairs, truth.get("edits", []))
    else:
        from graders.common import VerifyResult

        line_result = VerifyResult(
            lines_correct=False,
            reason="category mismatch; line verifier not run",
        )

    lines_correct = category_correct and line_result.lines_correct

    if category_correct and lines_correct:
        score = 1.0
    elif category_correct:
        score = 0.5
    else:
        score = 0.0

    verdict.update(
        score=score,
        category_correct=category_correct,
        lines_correct=lines_correct,
        submission_pairs=[
            {"broken": broken_line, "fixed": fixed_line}
            for broken_line, fixed_line in submission_pairs
        ],
    )
    if line_result.reason:
        verdict["reason"] = line_result.reason
    return verdict


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "submission",
        nargs="?",
        default=os.environ.get("SUBMISSION", DEFAULT_SUBMISSION),
    )
    parser.add_argument(
        "truth",
        nargs="?",
        default=os.environ.get("TRUTH", DEFAULT_TRUTH),
    )
    args = parser.parse_args()

    submission_path = Path(args.submission)
    truth_path = Path(args.truth)

    if not submission_path.exists():
        print(
            json.dumps(
                {"score": 0.0, "reason": f"submission not found: {submission_path}"},
                indent=2,
            )
        )
        sys.exit(1)
    if not truth_path.exists():
        print(
            json.dumps(
                {"score": 0.0, "reason": f"truth not found: {truth_path}"},
                indent=2,
            )
        )
        sys.exit(1)

    verdict = grade_submission(load_json(truth_path), load_json(submission_path))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()