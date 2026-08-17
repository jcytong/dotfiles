#!/usr/bin/env python3
"""
Ensemble / Multi-model combination strategy for DSPy models.

Loads two pretrained models and applies a configurable combination strategy.

Built-in strategies:
  - classification_cascade: Cascading approval with confidence thresholds
  - best_by_score: Select the prediction with the highest score from a scoring function

Usage:
    python ensemble.py \
        --model-a pretrained/bsfs.json --threshold-a 0.70 \
        --model-b pretrained/copro.json --threshold-b 0.55

    python ensemble.py \
        --model-a pretrained/gepa_model.json --threshold-a 0.70 \
        --model-b pretrained/copro.json --threshold-b 0.55
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

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
# Classification default:
EVALUATE_FN: Callable[[str, List[dspy.Example], List[dspy.Prediction]], Dict[str, Any]] = (
    lambda name, exs, preds: evaluate_classification(
        name,
        [getattr(p, OUTPUT_FIELD) for p in preds],
        [getattr(e, OUTPUT_FIELD) for e in exs],
        "APPROVED",
    )
)

# Combination strategy — configure for your task.
# Override COMBINE_FN to change how models are combined.
# Signature: (ex, pred_a, pred_b, args) -> dspy.Prediction
#
# Classification cascade (default — uncomment to use):
# Set via command-line args --threshold-a and --threshold-b
#
# Generic best-by-score (uncomment to use):
# def my_score(gold, pred): ...
# COMBINE_FN = lambda ex, pa, pb, args: best_by_score(pa, pb, my_score, ex)

INFERENCE_MODEL = "openai/gpt-4o"
INFERENCE_MAX_TOKENS = 16000

TRAIN_FRAC = 0.64
DEV_FRAC = 0.16
SPLIT_SEED = 42

DIAGNOSTICS_DIR = Path("analysis")

# ── END CONFIGURATION ────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Built-in combination strategies
# ---------------------------------------------------------------------------

def classification_cascade(
    pred_a: dspy.Prediction,
    pred_b: dspy.Prediction,
    threshold_a: float,
    threshold_b: float,
    positive_label: str,
    negative_label: str,
    output_field: str,
) -> dspy.Prediction:
    """
    Cascading approval strategy for classification tasks.

    Stage 1: If model A predicts positive with confidence >= threshold_a, approve.
    Stage 2: If model B predicts positive with confidence >= threshold_b, approve.
    Default: Return negative label.
    """
    a_label = getattr(pred_a, output_field, "").upper()
    b_label = getattr(pred_b, output_field, "").upper()
    a_conf = float(getattr(pred_a, "confidence", 0) or 0)
    b_conf = float(getattr(pred_b, "confidence", 0) or 0)

    if a_label == positive_label.upper() and a_conf >= threshold_a:
        return pred_a
    elif b_label == positive_label.upper() and b_conf >= threshold_b:
        return pred_b
    else:
        return dspy.Prediction(**{output_field: negative_label})


def best_by_score(
    pred_a: dspy.Prediction,
    pred_b: dspy.Prediction,
    score_fn: Callable[[dspy.Example, dspy.Prediction], float],
    gold: dspy.Example,
) -> dspy.Prediction:
    """
    Select the prediction with the higher score.

    Useful for non-classification tasks (QA, summarization, extraction)
    where there's no binary approve/reject threshold.
    """
    score_a = score_fn(gold, pred_a)
    score_b = score_fn(gold, pred_b)
    return pred_a if score_a >= score_b else pred_b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_data() -> List[dspy.Example]:
    """Replace with your data loading."""
    raise NotImplementedError("Replace with your data loading logic.")


def load_model(model_path: str) -> dspy.Module:
    """Load a pretrained model from JSON. Replace Signature with your own."""
    prog = dspy.Predict(dspy.Signature).reset_copy()
    prog.load(model_path)
    return prog


# Default combination function (classification cascade)
COMBINE_FN: Callable = None  # Set below or override in CONFIGURE section


def main():
    parser = argparse.ArgumentParser(description="Ensemble / Multi-model evaluation")
    parser.add_argument("--model-a", required=True, help="Path to model A")
    parser.add_argument("--threshold-a", type=float, default=0.70, help="Confidence threshold for model A")
    parser.add_argument("--model-b", required=True, help="Path to model B")
    parser.add_argument("--threshold-b", type=float, default=0.55, help="Confidence threshold for model B")
    parser.add_argument("--eval-set", choices=["dev", "holdout"], default="dev", help="Which split to evaluate on")
    args = parser.parse_args()

    dspy.configure(
        lm=dspy.LM(model=INFERENCE_MODEL, temperature=1.0, max_tokens=INFERENCE_MAX_TOKENS)
    )

    # Load models
    model_a = load_model(args.model_a)
    model_b = load_model(args.model_b)
    print(f"Model A: {args.model_a} (threshold={args.threshold_a})")
    print(f"Model B: {args.model_b} (threshold={args.threshold_b})")

    # Load data
    examples = load_data()
    data_train, data_dev, data_holdout = stratified_split(
        examples, TRAIN_FRAC, DEV_FRAC, SPLIT_SEED, label_attr=OUTPUT_FIELD,
    )
    eval_data = data_dev if args.eval_set == "dev" else data_holdout
    print(f"Evaluating on {args.eval_set} ({len(eval_data)} examples)")

    # Run individual models
    a_preds = {}
    b_preds = {}
    for ex in eval_data:
        a_preds[id(ex)] = model_a(**ex.inputs())
        b_preds[id(ex)] = model_b(**ex.inputs())

    # Evaluate individually
    a_predictions = [a_preds[id(ex)] for ex in eval_data]
    b_predictions = [b_preds[id(ex)] for ex in eval_data]

    EVALUATE_FN("Model A alone", eval_data, a_predictions)
    EVALUATE_FN("Model B alone", eval_data, b_predictions)

    # Combine predictions
    combine_fn = COMBINE_FN
    if combine_fn is None:
        # Default: classification cascade
        combine_fn = lambda ex, pa, pb, a: classification_cascade(
            pa, pb, a.threshold_a, a.threshold_b,
            "APPROVED", "REJECTED", OUTPUT_FIELD,
        )

    combined_predictions = []
    for ex in eval_data:
        combined = combine_fn(ex, a_preds[id(ex)], b_preds[id(ex)], args)
        combined_predictions.append(combined)

    results = EVALUATE_FN("ENSEMBLE", eval_data, combined_predictions)

    # Save
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DIAGNOSTICS_DIR / "ensemble_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "model_a": args.model_a,
            "threshold_a": args.threshold_a,
            "model_b": args.model_b,
            "threshold_b": args.threshold_b,
            "eval_set": args.eval_set,
            "metrics": results,
        }, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
