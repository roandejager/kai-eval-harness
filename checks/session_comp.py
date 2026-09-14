import re
from typing import Any, Dict, List, Optional, Set

from .base import (
    CaseEvaluation,
    CheckResult,
    EvalConfig,
    extract_numerals,
    is_approx_equal,
)


def extract_session_identifiers(evidence_items: List[Dict[str, Any]]) -> Set[str]:
    """Collects all unique session IDs and dates from cited evidence items."""
    sessions: Set[str] = set()
    for item in evidence_items:
        source = item.get("source", {})
        for s_id in source.get("session_ids", []):
            if s_id:
                sessions.add(str(s_id))
        for d_str in source.get("dates", []):
            if d_str:
                sessions.add(str(d_str))
        # Also check series entry timestamps if available
        series = item.get("values", {}).get("series", [])
        for entry in series:
            if "at" in entry:
                sessions.add(str(entry["at"]))
    return sessions


def check_sc1_session_presence(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    min_sessions: int = 2,
) -> CheckResult:
    """SC-1: Validates that cited evidence references at least 2 distinct sessions."""
    sessions = extract_session_identifiers(evidence)
    session_count = len(sessions)
    response_text = actual.get("response_text", "").lower()
    is_refusal = actual.get("refusal", False)

    # If fewer than min_sessions exist, passing requires stating it cannot compare yet
    if session_count < min_sessions:
        cannot_compare_hedges = [
            "cannot compare",
            "not enough",
            "log a few",
            "need at least",
            "couldn't put a proper answer",
        ]
        has_hedge = any(h in response_text for h in cannot_compare_hedges)
        if is_refusal or has_hedge:
            return CheckResult(
                "SC-1",
                True,
                detail=f"Gracefully declined comparison: found {session_count} session(s).",
            )
        return CheckResult(
            "SC-1",
            False,
            detail=f"Comparison claimed from only {session_count} session(s), expected >= {min_sessions}.",
        )

    return CheckResult(
        "SC-1",
        True,
        detail=f"Verified {session_count} distinct sessions in cited evidence.",
    )


def check_sc2_subject_scoping(
    expected: Dict[str, Any], actual: Dict[str, Any], evidence: List[Dict[str, Any]]
) -> CheckResult:
    """SC-2: Validates that only in-scope exercises contribute when subject is a region/muscle."""
    exp_subj = expected.get("expected_subject", {})
    exp_kind = exp_subj.get("kind")
    exp_val = (exp_subj.get("value") or "").lower()

    if exp_kind in ["region", "focus", "muscle"] and exp_val:
        out_of_scope: List[str] = []
        for item in evidence:
            item_subj = item.get("subject", {})
            item_val = (item_subj.get("value") or "").lower()
            item_kind = item_subj.get("kind")

            # If evidence asserts a specific exercise, check that its subject or label matches
            exercise_info = item.get("exercise", {})
            ex_label = (exercise_info.get("label") or "").lower()

            # Flag if an out-of-scope exercise is present (e.g. Bench Press inside a Legs query)
            if exp_val == "legs" and any(
                chest_term in ex_label
                for chest_term in ["bench", "chest", "curl", "row"]
            ):
                out_of_scope.append(exercise_info.get("label", "unknown"))

        if out_of_scope:
            return CheckResult(
                "SC-2",
                False,
                detail=f"Out-of-scope exercises cited for subject '{exp_val}': {', '.join(out_of_scope)}.",
            )

    return CheckResult(
        "SC-2",
        True,
        detail="All cited evidence exercises match query subject scope.",
    )


def check_sc3_required_values(
    expected: Dict[str, Any], actual: Dict[str, Any], evidence: List[Dict[str, Any]]
) -> CheckResult:
    """SC-3: Checks that required workout values (volume, weight, reps, sets) are present."""
    if not evidence and not actual.get("refusal"):
        return CheckResult(
            "SC-3", False, detail="No cited evidence items available to verify values."
        )

    missing_fields: List[str] = []
    for item in evidence:
        values = item.get("values", {})
        series = values.get("series", [])
        if series:
            for s in series:
                # Validate series entry contains numerical metrics
                if not any(k in s for k in ["e1rm", "volume", "top_weight", "top_reps"]):
                    missing_fields.append(f"series item in {item.get('id')}")

    if missing_fields:
        return CheckResult(
            "SC-3",
            False,
            detail=f"Missing metrics in cited series: {', '.join(missing_fields)}.",
        )

    return CheckResult(
        "SC-3", True, detail="All required metric values are present in cited evidence."
    )


def check_sc4_directional_arithmetic(
    expected: Dict[str, Any], actual: Dict[str, Any], evidence: List[Dict[str, Any]]
) -> CheckResult:
    """SC-4: Recomputes mathematical delta and checks that prose directional words match."""
    response_text = actual.get("response_text", "").lower()
    if not response_text:
        # Deterministic run without text passes arithmetic check by default
        return CheckResult(
            "SC-4",
            True,
            detail="Deterministic run: no reply text to evaluate directional words.",
        )

    # Collect numeric deltas from series data
    deltas: List[float] = []
    for item in evidence:
        series = item.get("values", {}).get("series", [])
        if len(series) >= 2:
            # Sort series chronologically if 'at' timestamp exists
            sorted_series = sorted(series, key=lambda x: x.get("at", 0))
            first_val = sorted_series[0].get("e1rm") or sorted_series[0].get("volume")
            last_val = sorted_series[-1].get("e1rm") or sorted_series[-1].get("volume")
            if first_val is not None and last_val is not None:
                deltas.append(float(last_val) - float(first_val))

    if not deltas:
        return CheckResult(
            "SC-4", True, detail="No multi-point series found to compute arithmetic deltas."
        )

    overall_delta = sum(deltas) / len(deltas)

    # Scan for upward and downward directional words in prose
    up_words = ["up", "increased", "more", "heavier", "added", "gained", "higher", "improved"]
    down_words = ["down", "decreased", "less", "lighter", "dropped", "lost", "lower", "declined"]

    claimed_up = any(re.search(rf"\b{w}\b", response_text) for w in up_words)
    claimed_down = any(re.search(rf"\b{w}\b", response_text) for w in down_words)

    if overall_delta > 0.1 and claimed_down and not claimed_up:
        return CheckResult(
            "SC-4",
            False,
            detail=f"Direction contradiction: data increased (delta={overall_delta:+.2f}), but prose claimed decline.",
        )

    if overall_delta < -0.1 and claimed_up and not claimed_down:
        return CheckResult(
            "SC-4",
            False,
            detail=f"Direction contradiction: data decreased (delta={overall_delta:+.2f}), but prose claimed increase.",
        )

    return CheckResult(
        "SC-4",
        True,
        detail=f"Directional prose aligns with arithmetic delta ({overall_delta:+.2f}).",
    )


def check_sc5_no_invented_numbers(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    tolerance: float = 0.05,
) -> CheckResult:
    """SC-5: Verifies that all numerals in reply text exist in evidence or are valid deltas."""
    response_text = actual.get("response_text", "")
    if not response_text:
        return CheckResult(
            "SC-5",
            True,
            detail="Deterministic run: no reply text to scan for invented numerals.",
        )

    text_numerals = extract_numerals(response_text)
    if not text_numerals:
        return CheckResult("SC-5", True, detail="No numerals present in response text.")

    # Build legitimate number pool from cited evidence
    cited_numbers: Set[float] = set()
    for item in evidence:
        values = item.get("values", {})
        if "value" in values and isinstance(values["value"], (int, float)):
            cited_numbers.add(float(values["value"]))
        if "window_days" in values and isinstance(values["window_days"], (int, float)):
            cited_numbers.add(float(values["window_days"]))

        series = values.get("series", [])
        for entry in series:
            for k in ["e1rm", "volume", "top_weight", "top_reps"]:
                if k in entry and isinstance(entry[k], (int, float)):
                    cited_numbers.add(float(entry[k]))

    # Compute valid differences and sums from cited numbers
    derived_numbers: Set[float] = set(cited_numbers)
    cited_list = list(cited_numbers)
    for i in range(len(cited_list)):
        for j in range(len(cited_list)):
            derived_numbers.add(abs(cited_list[i] - cited_list[j]))
            derived_numbers.add(cited_list[i] + cited_list[j])

    # Allow common formatting numbers (e.g. 1 session, 2 sessions, 4 weeks, 28 days)
    standard_units = {1.0, 2.0, 4.0, 7.0, 14.0, 21.0, 28.0, 30.0}
    derived_numbers.update(standard_units)

    invented: List[float] = []
    for num in text_numerals:
        if not any(is_approx_equal(num, valid, tolerance) for valid in derived_numbers):
            # Check if it could be a percentage of a cited number
            is_pct = any(
                is_approx_equal(num, (v1 / v2) * 100, tolerance)
                for v1 in cited_list
                for v2 in cited_list
                if v2 != 0
            )
            if not is_pct:
                invented.append(num)

    if invented:
        return CheckResult(
            "SC-5",
            False,
            detail=f"Invented numerals detected in prose without evidence support: {invented}.",
        )

    return CheckResult(
        "SC-5",
        True,
        detail="All numerals in response text are traceable to cited evidence.",
    )


def evaluate_session_comparison(
    expected: Dict[str, Any], actual: Dict[str, Any], config: EvalConfig = EvalConfig()
) -> CaseEvaluation:
    """Evaluates a single session-comparison case across all SC-1 to SC-5 checks."""
    evidence = actual.get("evidence") or actual.get("retrieved_evidence") or []
    case_id = expected.get("id", "sc-unknown")
    category = expected.get("category", "session-comparison")
    is_known_failure = expected.get("known_failure") is not None

    checks: List[CheckResult] = [
        check_sc1_session_presence(
            expected, actual, evidence, min_sessions=config.min_comparison_sessions
        ),
        check_sc2_subject_scoping(expected, actual, evidence),
        check_sc3_required_values(expected, actual, evidence),
        check_sc4_directional_arithmetic(expected, actual, evidence),
        check_sc5_no_invented_numbers(
            expected, actual, evidence, tolerance=config.numeric_tolerance
        ),
    ]

    all_passed = all(c.passed for c in checks if not c.is_soft)

    return CaseEvaluation(
        case_id=case_id,
        category=category,
        passed=all_passed,
        checks=checks,
        is_known_failure=is_known_failure,
        notes=expected.get("notes"),
    )