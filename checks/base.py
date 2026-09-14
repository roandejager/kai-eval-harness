import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


@dataclass
class CheckResult:
    """Represents the outcome of a single granular sub-check (e.g., SC-1, TN-3)."""

    check_id: str
    passed: bool
    is_soft: bool = False
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "check": self.check_id,
            "passed": self.passed,
        }
        if self.is_soft:
            data["soft"] = True
        if self.detail:
            data["detail"] = self.detail
        return data


@dataclass
class CaseEvaluation:
    """Aggregated evaluation results for a single test case across all sub-checks."""

    case_id: str
    category: str
    passed: bool
    checks: List[CheckResult] = field(default_factory=list)
    is_known_failure: bool = False
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.case_id,
            "category": self.category,
            "passed": self.passed,
            "known_failure": self.is_known_failure,
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass
class EvalConfig:
    """Configurable parameters for evaluation checks."""

    min_trend_sessions: int = 4
    min_trend_days: int = 21
    numeric_tolerance: float = 0.05
    min_comparison_sessions: int = 2


def extract_numerals(text: Optional[str]) -> List[float]:
    """Extracts all numbers (integers and floats) from a text string."""
    if not text:
        return []
    # Matches integers and decimals (e.g. 102.5, 4100, 2.5)
    raw_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    results: List[float] = []
    for num in raw_numbers:
        try:
            results.append(float(num))
        except ValueError:
            continue
    return results


def parse_iso_date(date_str: str) -> Optional[datetime]:
    """Safely parses YYYY-MM-DD date strings."""
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except (ValueError, AttributeError):
        return None


def calculate_day_span(dates: List[str]) -> int:
    """Calculates the span in days between the earliest and latest dates in a list."""
    parsed_dates = [parse_iso_date(d) for d in dates if parse_iso_date(d)]
    if len(parsed_dates) < 2:
        return 0
    earliest = min(parsed_dates)
    latest = max(parsed_dates)
    return (latest - earliest).days


def is_approx_equal(a: float, b: float, tolerance: float = 0.05) -> bool:
    """Compares two floating point numbers within a given tolerance."""
    return abs(a - b) <= tolerance