```markdown
# Kai Evaluation & Regression Harness

A lightweight, deterministic scoring harness for evaluating intent classification, subject resolution, retrieval volume balance, and safety guardrails in AI workout coaches.

## Overview

Conversational fitness assistants frequently suffer from two common failure modes:
1. **Sub-string collision during entity resolution:** Preprocessing steps (such as stripping apostrophes from contractions like `"how's"`) can leave stray tokens that trigger wildcard matches on unintended exercises (e.g., resolving a general leg question to `"Bicycle Crunch Raised Legs"`).
2. **Retrieval volume imbalance:** Over-retrieving context on broad questions (causing context bloat or hallucinated certainty) and under-retrieving on specific, time-bounded queries.

This harness provides a reproducible evaluation pipeline that scores model responses against a synthetic benchmark suite per category rather than relying on a single aggregate number.

## Metrics Tracked

- **Intent Accuracy (%):** Exact match against expected query intent classification.
- **Subject Resolution Accuracy (%):** Exact match across both subject kind (`region`, `exercise`, `muscle`, `global`, `none`) and normalized target name.
- **Exercise Constraint Violations:** Explicit flag tracking cases that must never resolve to a single exercise.
- **Fact Bounds Pass Rate (%):** Verifies retrieved workout log facts fall within expected `min` and `max` bounds.
- **Safety / Guarded Pass Rate:** Validates that medical, pain, nutrition, and unsupported queries correctly trigger a refusal step-back.

## Quick Start

### 1. Installation

This harness requires Python 3.8+ and uses only standard library modules.

```bash
git clone https://github.com/roandejager/kai-eval-harness.git
cd kai-eval-harness
```

### 2. Run the Benchmark

```bash
# Basic run
py scorer.py --eval-set eval_set.json --responses mock_responses.json

# Run and export full case-by-case details to JSON
py scorer.py --eval-set eval_set.json --responses mock_responses.json --export results.json
```

### 3. Sample Terminal Output

```text
=====================================================================================
  KAI RETRIEVAL AND SUBJECT-RESOLUTION EVAL SET : BENCHMARK REPORT
=====================================================================================
Category               | Tested | Intent %   | Subj %     | Ex Violations  | Fact Bounds %
-------------------------------------------------------------------------------------
guarded                | 1      |     100.0% |     100.0% |              0 |        100.0%
specific               | 1      |     100.0% |     100.0% |              0 |          0.0%
subject-resolution     | 2      |     100.0% |      50.0% |              1 |        100.0%
vague                  | 1      |     100.0% |     100.0% |              0 |          0.0%
-------------------------------------------------------------------------------------
OVERALL                | 5      |     100.0% |      80.0% |              1 |         60.0%
=====================================================================================
```