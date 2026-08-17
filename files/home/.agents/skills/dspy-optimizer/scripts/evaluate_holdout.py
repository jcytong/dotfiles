#!/usr/bin/env python3
"""
Holdout evaluation and error analysis for trained DSPy models.

Loads a pretrained model, runs it on the holdout set exactly once,
prints aggregate metrics, and writes per-example diagnostics.

Usage:
    python evaluate_holdout.py --model pretrained/gepa_model.json
    python evaluate_holdout.py --model pretrained/gepa_model.json --error-analysis
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import dspy
from dotenv import load_dotenv

load_dotenv()

from data_utils import load_from_csv, print_split_summary, stratified_split
from metrics import evaluate_classification

# ── CONFIGURE THESE ──────────────────────────────────────────────────────────

# Output field — the name of the field your Signature produces as its main output.
# Classification: "status"  |  QA: "answer"  |  Extraction: "entities"
OUTPUT_FIELD = "status"

# Evaluation function — configure for your task.
# Classification example (default):
EVALUATE_FN: Callable[[str, List[dspy.Example], List[dspy.Prediction]], Dict[str, Any]] = (
    lambda name, exs, preds: evaluate_classification(
        name,
        [getattr(p, OUTPUT_FIELD) for p in preds],
        [getattr(e, OUTPUT_FIELD) for e in exs],
        "APPROVED",
    )
)
# Generic example (uncomment):
# from metrics import evaluate_from_fn
# def my_score(gold, pred): ...
# EVALUATE_FN = lambda name, exs, preds: evaluate_from_fn(name, exs, preds, my_score)

# Error categories — define how to categorize prediction outcomes for error analysis.
# Classification default: TP/FP/TN/FN based on positive/negative label.
# Override CLASSIFY_OUTCOME_FN for custom error categories.
def _default_classify_outcome(gold: dspy.Example, pred: dspy.Prediction) -> str:
    """Default classification outcome categorizer (TP/FP/TN/FN)."""
    positive = "APPROVED"
    g = getattr(gold, OUTPUT_FIELD, "").upper() == positive.upper()
    p = getattr(pred, OUTPUT_FIELD, "").upper() == positive.upper()
    if g and p:
        return "TP"
    if g and not p:
        return "FN"
    if not g and not p:
        return "TN"
    return "FP"

CLASSIFY_OUTCOME_FN: Callable[[dspy.Example, dspy.Prediction], str] = _default_classify_outcome

# Which outcome categories to show in error analysis (and their display labels).
ERROR_CATEGORIES: Dict[str, str] = {
    "FN": "FALSE NEGATIVES (missed positives)",
    "FP": "FALSE POSITIVES (incorrect approvals)",
}

INFERENCE_MODEL = "openai/gpt-4o"
INFERENCE_TEMPERATURE = 1.0
INFERENCE_MAX_TOKENS = 16000

TRAIN_FRAC = 0.64
DEV_FRAC = 0.16
SPLIT_SEED = 42

DIAGNOSTICS_DIR = Path("analysis")

# ── END CONFIGURATION ────────────────────────────────────────────────────────


def load_data() -> List[dspy.Example]:
    """Replace with your data loading."""
    raise NotImplementedError("Replace with your data loading logic.")


def run_evaluation(
    model: dspy.Module,
    data: List[dspy.Example],
) -> tuple[Dict[str, Any], List[Dict]]:
    """Run model on data, return (aggregate_metrics, per_example_diagnostics)."""
    predictions: List[dspy.Prediction] = []
    diagnostics: List[Dict] = []

    for idx, ex in enumerate(data):
        pred = model(**ex.inputs())
        predictions.append(pred)

        diagnostics.append({
            "index": idx,
            "gold": getattr(ex, OUTPUT_FIELD, ""),
            "prediction": getattr(pred, OUTPUT_FIELD, ""),
            "outcome": CLASSIFY_OUTCOME_FN(ex, pred),
            "confidence": getattr(pred, "confidence", None),
            "reasoning": getattr(pred, "reasoning", ""),
            "gold_reasoning": getattr(ex, "reasoning", ""),
        })

    metrics = EVALUATE_FN("Holdout", data, predictions)
    return metrics, diagnostics


def print_error_analysis(
    diagnostics: List[Dict],
    outcome_type: str,
    label: str,
    limit: int = 5,
) -> None:
    """Print sample errors of a given outcome type."""
    samples = [d for d in diagnostics if d["outcome"] == outcome_type][:limit]
    if not samples:
        print(f"\n  No {outcome_type} examples.")
        return

    print(f"\n  {label} ({len(samples)} shown):")
    for s in samples:
        print(f"    [{s['index']}] Gold={s['gold']} Pred={s['prediction']} Conf={s['confidence']}")
        if s["reasoning"]:
            print(f"         Reasoning: {s['reasoning'][:200]}")
        if s["gold_reasoning"]:
            print(f"         Gold:      {s['gold_reasoning'][:200]}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Holdout evaluation")
    parser.add_argument("--model", required=True, help="Path to pretrained model JSON")
    parser.add_argument("--error-analysis", action="store_true", help="Print error samples")
    parser.add_argument("--error-limit", type=int, default=5, help="Max samples per error category")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Configure LM
    dspy.configure(
        lm=dspy.LM(model=INFERENCE_MODEL, temperature=INFERENCE_TEMPERATURE, max_tokens=INFERENCE_MAX_TOKENS)
    )

    # Load model
    # Replace dspy.Signature with your actual Signature class
    prog = dspy.Predict(dspy.Signature).reset_copy()
    prog.load(str(model_path))
    print(f"Loaded model from {model_path}")

    # Load and split data (only need holdout)
    examples = load_data()
    _, _, data_holdout = stratified_split(
        examples, TRAIN_FRAC, DEV_FRAC, SPLIT_SEED, label_attr=OUTPUT_FIELD,
    )
    print(f"Holdout set: {len(data_holdout)} examples")

    # Evaluate
    metrics, diagnostics = run_evaluation(prog, data_holdout)

    # Error analysis
    if args.error_analysis:
        for category, label in ERROR_CATEGORIES.items():
            print_error_analysis(diagnostics, category, label, limit=args.error_limit)

    # Save diagnostics
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    diag_path = DIAGNOSTICS_DIR / "holdout_diagnostics.jsonl"
    with diag_path.open("w", encoding="utf-8") as f:
        for row in diagnostics:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = DIAGNOSTICS_DIR / "holdout_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\nDiagnostics: {diag_path}")
    print(f"Summary:     {summary_path}")


if __name__ == "__main__":
    main()
