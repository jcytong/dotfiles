#!/usr/bin/env python3
"""
Ensemble / Union strategy for combining multiple DSPy models.

Loads two pretrained models and applies a cascading approval strategy:
  Stage 1: High-confidence model approves with strict threshold
  Stage 2: High-recall model approves with looser threshold
  Default: Reject

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
from typing import List

import dspy
from dotenv import load_dotenv

load_dotenv()

from data_utils import load_from_csv, print_split_summary, stratified_split
from metrics import evaluate_predictions

# ── CONFIGURE THESE ──────────────────────────────────────────────────────────

POSITIVE_LABEL = "APPROVED"
NEGATIVE_LABEL = "REJECTED"

INFERENCE_MODEL = "openai/gpt-4o"
INFERENCE_MAX_TOKENS = 16000

TRAIN_FRAC = 0.64
DEV_FRAC = 0.16
SPLIT_SEED = 42

DIAGNOSTICS_DIR = Path("analysis")

# ── END CONFIGURATION ────────────────────────────────────────────────────────


def load_data() -> List[dspy.Example]:
    """Replace with your data loading."""
    raise NotImplementedError("Replace with your data loading logic.")


def load_model(model_path: str) -> dspy.Module:
    """Load a pretrained model from JSON. Replace Signature with your own."""
    prog = dspy.Predict(dspy.Signature).reset_copy()
    prog.load(model_path)
    return prog


def main():
    parser = argparse.ArgumentParser(description="Ensemble / Union evaluation")
    parser.add_argument("--model-a", required=True, help="Path to model A (high-confidence)")
    parser.add_argument("--threshold-a", type=float, default=0.70, help="Confidence threshold for model A")
    parser.add_argument("--model-b", required=True, help="Path to model B (high-recall fallback)")
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
    data_train, data_dev, data_holdout = stratified_split(examples, TRAIN_FRAC, DEV_FRAC, SPLIT_SEED)
    eval_data = data_dev if args.eval_set == "dev" else data_holdout
    print(f"Evaluating on {args.eval_set} ({len(eval_data)} examples)")

    # Run individual models
    a_preds = {}
    b_preds = {}
    for ex in eval_data:
        a_preds[id(ex)] = model_a(**ex.inputs())
        b_preds[id(ex)] = model_b(**ex.inputs())

    # Evaluate individually
    a_labels = [a_preds[id(ex)].status for ex in eval_data]
    b_labels = [b_preds[id(ex)].status for ex in eval_data]
    golds = [ex.status for ex in eval_data]

    evaluate_predictions("Model A alone", a_labels, golds, POSITIVE_LABEL)
    evaluate_predictions("Model B alone", b_labels, golds, POSITIVE_LABEL)

    # Union strategy
    union_preds = []
    for ex in eval_data:
        status = NEGATIVE_LABEL
        a_pred = a_preds[id(ex)]
        b_pred = b_preds[id(ex)]

        a_conf = float(getattr(a_pred, "confidence", 0) or 0)
        b_conf = float(getattr(b_pred, "confidence", 0) or 0)

        # Stage 1: High-confidence model
        if a_pred.status.upper() == POSITIVE_LABEL.upper() and a_conf >= args.threshold_a:
            status = POSITIVE_LABEL
        # Stage 2: High-recall fallback
        elif b_pred.status.upper() == POSITIVE_LABEL.upper() and b_conf >= args.threshold_b:
            status = POSITIVE_LABEL

        union_preds.append(status)

    results = evaluate_predictions("UNION", union_preds, golds, POSITIVE_LABEL)

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
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
