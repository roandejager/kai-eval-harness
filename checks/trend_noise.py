import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    CaseEvaluation,
    CheckResult,
    EvalConfig,
    calculate_day_span,
    parse_iso_date,
)


def extract_e1rm_series(
    evidence_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extracts all dated series points containing e1RM values."""
    series_points: List[Dict[str, Any]] = []
    for item in evidence_items:
        series = item.get("values", {}).get("series", [])
        for entry in series:
            if "e1rm" in entry:
                series_points.append(entry)
    # Sort chronologically if timestamps are present
    return sorted(series_points, key=lambda x: x.get("at", 0))


def compute_trend_slope(series: List[Dict[str, Any]]) -> float:
    """Computes a simple linear trend slope across e1RM values."""
    if len(series) < 2:
        return 0.0

    # Extract numerical (index, e1rm) pairs
    y_vals = [float(p["e1rm"]) for p in series]
    n = len(y_vals)
    x_vals = list(range(n))

    mean_x = sum(x_vals) / n
    mean_y = sum(y_vals) / n

    numerator = sum((x_vals[i] - mean_x) * (y_vals[i] - mean_y) for i in range(n))
    denominator = sum((x_vals[i] - mean_x) ** 2 for i in range(n))

    if denominator == 0:
        return 0.0
    return numerator / denominator


def check_tn1_allowed_conclusions(
    expected: Dict[str, Any], actual: Dict[str, Any]
) -> CheckResult:
    """TN-1: Validates that conclusion is one of the four allowed states."""
    allowed = expected.get("expected_evidence", {}).get("allowed_conclusions") or [
        "improving",
        "declining",
        "stable",
        "inconclusive",
    ]

    response_text = actual.get("response_text", "").lower()
    if not response_text:
        return CheckResult(
            "TN-1", True, detail="Deterministic run: no reply text to evaluate."
        )

    matched_conclusions: List[str] = []
    for state in ["improving", "declining", "stable", "inconclusive"]:
        if re.search(rf"\b{state}\b", response_text):
            matched_conclusions.append(state)

    # Check for informal synonyms (e.g. "up", "down", "steady", "not enough")
    if not matched_conclusions:
        if any(w in response_text for w in ["up", "gaining", "higher"]):
            matched_conclusions.append("improving")
        elif any(w in response_text for w in ["down", "dropping", "lagging"]):
            matched_conclusions.append("declining")
        elif any(w in response_text for w in ["held steady", "plateau"]):
            matched_conclusions.append("stable")
        elif any(w in response_text for w in ["not enough", "cannot read"]):
            matched_conclusions.append("inconclusive")

    if matched_conclusions:
        for c in matched_conclusions:
            if c not in allowed:
                return CheckResult(
                    "TN-1",
                    False,
                    detail=f"Conclusion '{c}' is not in permitted set: {allowed}.",
                )

    return CheckResult(
        "TN-1", True, detail="Conclusion aligns with permitted trend states."
    )


def check_tn2_sample_floor(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    min_sessions: int = 4,
    min_days: int = 21,
) -> CheckResult:
    """TN-2: Enforces minimum sessions and day span before claiming improving/declining."""
    series = extract_e1rm_series(evidence)
    response_text = actual.get("response_text", "").lower()

    # Collect all cited dates
    cited_dates: List[str] = []
    for item in evidence:
        for d in item.get("source", {}).get("dates", []):
            if d:
                cited_dates.append(str(d))

    session_count = max(len(series), len(cited_dates))
    day_span = calculate_day_span(cited_dates)

    # Check if a directional trend is claimed
    claims_direction = any(
        w in response_text for w in ["improving", "declining", "is up", "is down"]
    )
    is_hedged = any(
        h in response_text
        for h in ["inconclusive", "not enough", "one good", "one bad", "early signal"]
    )

    if claims_direction and not is_hedged:
        if session_count < min_sessions or (day_span > 0 and day_span < min_days):
            return CheckResult(
                "TN-2",
                False,
                detail=f"Trend claimed with insufficient history ({session_count} sessions, {day_span} days span). Expected >= {min_sessions} sessions / {min_days} days.",
            )

    return CheckResult(
        "TN-2",
        True,
        detail=f"Sample floor satisfied ({session_count} sessions, {day_span} days span).",
    )


def check_tn3_outlier_drop(
    expected: Dict[str, Any], actual: Dict[str, Any], evidence: List[Dict[str, Any]]
) -> CheckResult:
    """TN-3: Mathematically removes the most extreme point; flips in unhedged claims fail."""
    series = extract_e1rm_series(evidence)
    response_text = actual.get("response_text", "").lower()

    if len(series) < 3:
        return CheckResult(
            "TN-3",
            True,
            detail="Series too small to evaluate outlier drop; sample floor rules apply.",
        )

    baseline_slope = compute_trend_slope(series)

    # Find the single most extreme outlier point
    if baseline_slope > 0:
        # Drop the highest e1RM point
        max_idx = max(range(len(series)), key=lambda i: float(series[i]["e1rm"]))
        trimmed_series = [p for i, p in enumerate(series) if i != max_idx]
    else:
        # Drop the lowest e1RM point
        min_idx = min(range(len(series)), key=lambda i: float(series[i]["e1rm"]))
        trimmed_series = [p for i, p in enumerate(series) if i != min_idx]

    new_slope = compute_trend_slope(trimmed_series)

    # Check if trend direction flips or drops to zero
    slope_flipped = (baseline_slope > 0.1 and new_slope <= 0.0) or (
        baseline_slope < -0.1 and new_slope >= 0.0
    )

    has_hedge = any(
        h in response_text
        for h in ["one good session", "one bad session", "outlier", "early", "hedged"]
    )

    if slope_flipped and not has_hedge and response_text:
        return CheckResult(
            "TN-3",
            False,
            detail=f"Direction flips from slope={baseline_slope:.2f} to slope={new_slope:.2f} when extreme point dropped; unhedged claim.",
        )

    return CheckResult(
        "TN-3",
        True,
        detail=f"Trend is robust to outlier drop (baseline slope={baseline_slope:.2f}, trimmed slope={new_slope:.2f}).",
    )


def check_tn4_subject_integrity(
    expected: Dict[str, Any], actual: Dict[str, Any], evidence: List[Dict[str, Any]]
) -> CheckResult:
    """TN-4: Ensures cited series belongs to the resolved subject without borrowing other lifts."""
    exp_subj = expected.get("expected_subject", {})
    exp_val = (exp_subj.get("value") or "").lower()
    exp_kind = exp_subj.get("kind")

    if exp_kind == "exercise" and exp_val:
        for item in evidence:
            ex_info = item.get("exercise", {})
            ex_label = (ex_info.get("label") or "").lower()
            ex_id = (ex_info.get("id") or "").lower()

            if ex_label and exp_val not in ex_label and ex_label not in exp_val:
                # Citing a foreign lift (e.g. Bench Press inside a Snatch query)
                return CheckResult(
                    "TN-4",
                    False,
                    detail=f"Borrowed foreign lift history: cited '{ex_label}' for subject '{exp_val}'.",
                )

    return CheckResult(
        "TN-4",
        True,
        detail="All cited series match requested exercise subject integrity.",
    )


def check_tn5_named_window(
    expected: Dict[str, Any], actual: Dict[str, Any], evidence: List[Dict[str, Any]]
) -> CheckResult:
    """TN-5: Verifies named period in prose matches actual cited window (soft check)."""
    response_text = actual.get("response_text", "").lower()
    if not response_text:
        return CheckResult(
            "TN-5",
            True,
            is_soft=True,
            detail="Deterministic run: no reply text to evaluate.",
        )

    # Detect window phrasing in text
    week_match = re.search(r"(\d+)\s*weeks?", response_text)
    if week_match:
        claimed_weeks = int(week_match.group(1))
        claimed_days = claimed_weeks * 7

        # Check evidence window_days
        actual_window_days = None
        for item in evidence:
            w_days = item.get("values", {}).get("window_days")
            if w_days:
                actual_window_days = int(w_days)
                break

        if actual_window_days and abs(claimed_days - actual_window_days) > 7:
            return CheckResult(
                "TN-5",
                False,
                is_soft=True,
                detail=f"Window mismatch: prose named {claimed_weeks} weeks (~{claimed_days}d), but evidence window is {actual_window_days}d.",
            )

    return CheckResult(
        "TN-5",
        True,
        is_soft=True,
        detail="Named time window matches cited evidence window.",
    )


def evaluate_trend_vs_noise(
    expected: Dict[str, Any], actual: Dict[str, Any], config: EvalConfig = EvalConfig()
) -> CaseEvaluation:
    """Evaluates a single trend-vs-noise case across TN-1 to TN-5 checks."""
    evidence = actual.get("evidence") or actual.get("retrieved_evidence") or []
    case_id = expected.get("id", "tn-unknown")
    category = expected.get("category", "trend-vs-noise")
    is_known_failure = expected.get("known_failure") is not None

    checks: List[CheckResult] = [
        check_tn1_allowed_conclusions(expected, actual),
        check_tn2_sample_floor(
            expected,
            actual,
            evidence,
            min_sessions=config.min_trend_sessions,
            min_days=config.min_trend_days,
        ),
        check_tn3_outlier_drop(expected, actual, evidence),
        check_tn4_subject_integrity(expected, actual, evidence),
        check_tn5_named_window(expected, actual, evidence),
    ]

    # Overall pass requires all non-soft checks to pass
    all_passed = all(c.passed for c in checks if not c.is_soft)

    return CaseEvaluation(
        case_id=case_id,
        category=category,
        passed=all_passed,
        checks=checks,
        is_known_failure=is_known_failure,
        notes=expected.get("notes"),
    )