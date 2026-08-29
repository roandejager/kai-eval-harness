import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CaseResult:
    case_id: str
    category: str
    question: str
    intent_pass: bool
    subject_pass: bool
    exercise_violation: bool
    fact_bounds_pass: bool
    refusal_pass: bool
    details: Dict[str, Any]


def normalize(text: Optional[str]) -> Optional[str]:
    """Lowercases and strips whitespace for safe comparison."""
    if text is None:
        return None
    return text.strip().lower()


def evaluate_single_case(
    expected: Dict[str, Any], actual: Dict[str, Any]
) -> CaseResult:
    """Evaluates one actual response against the expected ground truth."""

    # 1. Intent check
    intent_pass = expected.get("expected_intent") == actual.get("intent")

    # 2. Subject resolution check (kind and value must match)
    exp_subj = expected.get("expected_subject") or {}
    act_subj = actual.get("subject") or {}

    exp_kind = exp_subj.get("kind")
    act_kind = act_subj.get("kind")
    exp_val = normalize(exp_subj.get("value"))
    act_val = normalize(act_subj.get("value"))

    subject_pass = (exp_kind == act_kind) and (exp_val == act_val)

    # 3. Exercise constraint violation check
    exercise_violation = False
    if expected.get("must_not_resolve_to_exercise") is True:
        if act_kind == "exercise":
            exercise_violation = True

    # 4. Fact count bounds check
    fact_count = actual.get("fact_count", 0)
    min_facts = expected.get("expected_fact_count_min")
    max_facts = expected.get("expected_fact_count_max")

    fact_bounds_pass = True
    if min_facts is not None and fact_count < min_facts:
        fact_bounds_pass = False
    if max_facts is not None and fact_count > max_facts:
        fact_bounds_pass = False

    # 5. Guarded refusal check
    refusal_pass = True
    if expected.get("category") == "guarded":
        actual_refused = (
            actual.get("refusal") is True or actual.get("intent") == "guarded"
        )
        if not actual_refused:
            refusal_pass = False

    return CaseResult(
        case_id=expected["id"],
        category=expected.get("category", "unclassified"),
        question=expected.get("question", ""),
        intent_pass=intent_pass,
        subject_pass=subject_pass,
        exercise_violation=exercise_violation,
        fact_bounds_pass=fact_bounds_pass,
        refusal_pass=refusal_pass,
        details={
            "expected_intent": expected.get("expected_intent"),
            "actual_intent": actual.get("intent"),
            "expected_subject": exp_subj,
            "actual_subject": act_subj,
            "fact_count": fact_count,
            "min_facts": min_facts,
            "max_facts": max_facts,
        },
    )


def print_report(results: List[CaseResult], title: str = "Kai Evaluation"):
    """Prints a structured per-category breakdown table in the terminal."""
    by_category = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)

    print("\n" + "=" * 85)
    print(f"  {title.upper()} : BENCHMARK REPORT")
    print("=" * 85)
    header = f"{'Category':<22} | {'Tested':<6} | {'Intent %':<10} | {'Subj %':<10} | {'Ex Violations':<14} | {'Fact Bounds %'}"
    print(header)
    print("-" * 85)

    tot_tested = len(results)
    tot_intent = sum(1 for r in results if r.intent_pass)
    tot_subj = sum(1 for r in results if r.subject_pass)
    tot_violations = sum(1 for r in results if r.exercise_violation)
    tot_facts = sum(1 for r in results if r.fact_bounds_pass)

    for cat, items in sorted(by_category.items()):
        count = len(items)
        intent_pct = (sum(1 for i in items if i.intent_pass) / count) * 100
        subj_pct = (sum(1 for i in items if i.subject_pass) / count) * 100
        violations = sum(1 for i in items if i.exercise_violation)
        fact_pct = (sum(1 for i in items if i.fact_bounds_pass) / count) * 100

        row = f"{cat:<22} | {count:<6} | {intent_pct:>9.1f}% | {subj_pct:>9.1f}% | {violations:>14} | {fact_pct:>12.1f}%"
        print(row)

    print("-" * 85)
    overall_intent = (tot_intent / tot_tested * 100) if tot_tested else 0
    overall_subj = (tot_subj / tot_tested * 100) if tot_tested else 0
    overall_facts = (tot_facts / tot_tested * 100) if tot_tested else 0

    overall_row = f"{'OVERALL':<22} | {tot_tested:<6} | {overall_intent:>9.1f}% | {overall_subj:>9.1f}% | {tot_violations:>14} | {overall_facts:>12.1f}%"
    print(overall_row)
    print("=" * 85 + "\n")


def run_benchmark(
    eval_path: str, responses_path: str, export_path: Optional[str] = None
):
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    with open(responses_path, "r", encoding="utf-8") as f:
        responses_data = json.load(f)

    actual_by_id = {item["id"]: item for item in responses_data}
    case_results = []

    for case in eval_data["cases"]:
        case_id = case["id"]
        if case_id not in actual_by_id:
            continue
        res = evaluate_single_case(case, actual_by_id[case_id])
        case_results.append(res)

    if not case_results:
        print("No matching case IDs found between the evaluation and response files.")
        return

    print_report(case_results, title=eval_data.get("name", "Kai Eval"))

    if export_path:
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in case_results], f, indent=2)
        print(f"Detailed case results saved to: {export_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scoring harness for Kai retrieval and subject-resolution eval set."
    )
    parser.add_argument(
        "--eval-set",
        default="eval_set.json",
        help="Path to the eval set JSON file",
    )
    parser.add_argument(
        "--responses",
        default="mock_responses.json",
        help="Path to the actual model responses JSON file",
    )
    parser.add_argument(
        "--export",
        default=None,
        help="Optional path to save full results as JSON",
    )

    args = parser.parse_args()
    run_benchmark(args.eval_set, args.responses, args.export)