"""
Kai Eval Harness - Evaluation Checks Package.
Provides modular validators for session comparison, trend vs noise, and recommendation grounding.
"""

from .base import CaseEvaluation, CheckResult, EvalConfig
from .session_comp import evaluate_session_comparison

__all__ = [
    "CheckResult",
    "CaseEvaluation",
    "EvalConfig",
    "evaluate_session_comparison",
]