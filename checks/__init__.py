"""
Kai Eval Harness - Evaluation Checks Package.
Provides modular validators for session comparison, trend vs noise, and recommendation grounding.
"""

from .base import CaseEvaluation, CheckResult, EvalConfig

__all__ = ["CheckResult", "CaseEvaluation", "EvalConfig"]