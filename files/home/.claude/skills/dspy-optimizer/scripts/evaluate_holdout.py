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
from typing import Any, Dict, List

import dspy
from dotenv import load_dotenv

load_dotenv()

from data_utils import load_from_csv, print_split_summary, stratified_split
from metrics import evaluate_predictions

# ── CONFIGURE THESE ──────────────────────────────────────────────────────────

POSITIVE_LABEL = "APPROVED"
NEGATIVE_LABEL = "REJECTED"

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


def classify_outcome(gold: str, pred: str, positive: str) -> str:
    g = gold.upper() == positive.upper()
    p = pred.upper() == positive.upper()
    if g and p:
        return "TP"
    if g and not p:
        return "FN"
    if not g and not p:
        return "TN"
    return "FP"


def run_evaluation(
    model: dspy.Module,
    data: List[dspy.Example],
    positive_label: str,
) -> tuple[Dict[str, Any], List[Dict]]:
    """Run model on data, return (aggregate_metrics, per_example_diagnostics)."""
    predictions: List[str] = []
    gold_labels: List[str] = []
    diagnostics: List[Dict] = []

    for idx, ex in enumerate(data):
        pred = model(**ex.inputs())
        pred_label = pred.status
        gold_label = ex.status

        predictions.append(pred_label)
        gold_labels.append(gold_label)

        diagnostics.append({
            "index": idx,
            "gold": gold_label,
            "prediction": pred_label,
            "outcome": classify_outcome(gold_label, pred_label, positive_label),
            "confidence": getattr(pred, "confidence", None),
            "reasoning": getattr(pred, "reasoning", ""),
            "gold_reasoning": getattr(ex, "reasoning", ""),
        })

    metrics = evaluate_predictions("Holdout", predictions, gold_labels, positive_label)
    return metrics, diagnostics


def print_error_analysis(
    diagnostics: List[Dict],
    outcome_type: str,
    limit: int = 5,
) -> None:
    """Print sample errors of a given type (FN or FP)."""
    samples = [d for d in diagnostics if d["outcome"] == outcome_type][:limit]
    if not samples:
        print(f"\n  No {outcome_type} examples.")
        return

    label = "FALSE NEGATIVES (missed positives)" if outcome_type == "FN" else "FALSE POSITIVES (incorrect approvals)"
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
    parser.add_argument("--error-analysis", action="store_true", help="Print FN/FP samples")
    parser.add_argument("--fn-limit", type=int, default=5, help="Max FN samples to show")
    parser.add_argument("--fp-limit", type=int, default=5, help="Max FP samples to show")
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
    _, _, data_holdout = stratified_split(examples, TRAIN_FRAC, DEV_FRAC, SPLIT_SEED)
    print(f"Holdout set: {len(data_holdout)} examples")

    # Evaluate
    metrics, diagnostics = run_evaluation(prog, data_holdout, POSITIVE_LABEL)

    # Error analysis
    if args.error_analysis:
        print_error_analysis(diagnostics, "FN", limit=args.fn_limit)
        print_error_analysis(diagnostics, "FP", limit=args.fp_limit)

    # Save diagnostics
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    diag_path = DIAGNOSTICS_DIR / "holdout_diagnostics.jsonl"
    with diag_path.open("w", encoding="utf-8") as f:
        for row in diagnostics:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = DIAGNOSTICS_DIR / "holdout_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDiagnostics: {diag_path}")
    print(f"Summary:     {summary_path}")


if __name__ == "__main__":
    main()
