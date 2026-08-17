#!/usr/bin/env python3
"""
GEPA Training Pipeline - Default optimizer for DSPy pipelines.

This is the primary training script. Configure the settings below,
then run:
    python train_gepa.py

Environment variables:
    QUICK_SANITY=1   - Use small subsets for fast smoke tests.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List

import dspy
from dspy.teleprompt import GEPA
from dotenv import load_dotenv

load_dotenv()

# ── CONFIGURE THESE ──────────────────────────────────────────────────────────

# Your dspy.Signature class - import or define it here
# from my_project.signature import MyPipeline
# SIGNATURE_CLASS = MyPipeline

# Data source (pick one and uncomment)
# DATA_PATH = Path("data/examples.csv")
# DATA_FORMAT = "csv"  # or "jsonl"
# INPUT_FIELDS = ["field_1", "field_2"]
# LABEL_FIELD = "label"
# LABEL_MAP = {"yes": "APPROVED", "no": "REJECTED"}  # or None for classification
# REASONING_FIELD = "reasoning"  # or None

# Output field — the name of the field your Signature produces as its main output.
# Classification: "status"  |  QA: "answer"  |  Extraction: "entities"
OUTPUT_FIELD = "status"

# Metric functions — configure for your task.
# Classification example (uncomment):
# from metrics import make_classification_gepa_metric, evaluate_classification, RECALL_WEIGHTS
# GEPA_METRIC = make_classification_gepa_metric("APPROVED", "REJECTED", weights=RECALL_WEIGHTS)
# EVALUATE_FN = lambda name, exs, preds: evaluate_classification(
#     name, [getattr(p, OUTPUT_FIELD) for p in preds],
#     [getattr(e, OUTPUT_FIELD) for e in exs], "APPROVED",
# )
#
# Generic example (uncomment):
# from metrics import make_gepa_metric_from_fn, evaluate_from_fn
# def my_score(gold, pred): ...
# def my_feedback(gold, pred, score): ...
# GEPA_METRIC = make_gepa_metric_from_fn(my_score, my_feedback)
# EVALUATE_FN = lambda name, exs, preds: evaluate_from_fn(name, exs, preds, my_score)

# LM configuration
INFERENCE_MODEL = "openai/gpt-4o"
INFERENCE_TEMPERATURE = 1.0
INFERENCE_MAX_TOKENS = 16000
REFLECTION_MODEL = "openai/gpt-4o"
REFLECTION_TEMPERATURE = 1.0
REFLECTION_MAX_TOKENS = 32000

# GEPA configuration
GEPA_AUTO = "medium"             # "light", "medium", "heavy"
GEPA_NUM_THREADS = 32
GEPA_REFLECTION_MINIBATCH = 8
GEPA_CANDIDATE_STRATEGY = "pareto"  # "pareto" or "current_best"
GEPA_USE_MERGE = True
GEPA_MAX_MERGE = 5
GEPA_SEED = 0

# Data split
TRAIN_FRAC = 0.64
DEV_FRAC = 0.16
SPLIT_SEED = 42

# Output
OUTPUT_DIR = Path("pretrained")
MODEL_FILENAME = "gepa_model.json"
DIAGNOSTICS_DIR = Path("analysis")

# ── END CONFIGURATION ────────────────────────────────────────────────────────

# Import portable utilities (copy these into your project or adjust sys.path)
from data_utils import (
    load_from_csv,
    load_from_jsonl,
    print_split_summary,
    stratified_split,
)
from metrics import (
    RECALL_WEIGHTS,
    evaluate_classification,
    make_classification_gepa_metric,
)

# Default metric functions (classification) — override in CONFIGURE section above
GEPA_METRIC = make_classification_gepa_metric("APPROVED", "REJECTED", weights=RECALL_WEIGHTS)
EVALUATE_FN: Callable[[str, List[dspy.Example], List[dspy.Prediction]], Dict[str, Any]] = (
    lambda name, exs, preds: evaluate_classification(
        name,
        [getattr(p, OUTPUT_FIELD) for p in preds],
        [getattr(e, OUTPUT_FIELD) for e in exs],
        "APPROVED",
    )
)


def load_data() -> List[dspy.Example]:
    """Load data from the configured source. Customize this function."""
    raise NotImplementedError(
        "Configure DATA_PATH, DATA_FORMAT, INPUT_FIELDS, LABEL_FIELD above, "
        "then replace this function body with:\n"
        "  return load_from_csv(DATA_PATH, INPUT_FIELDS, LABEL_FIELD, LABEL_MAP, REASONING_FIELD, OUTPUT_FIELD)\n"
        "or:\n"
        "  return load_from_jsonl(DATA_PATH, INPUT_FIELDS, LABEL_FIELD, LABEL_MAP, REASONING_FIELD, OUTPUT_FIELD)"
    )


def main():
    print("=" * 70)
    print("GEPA TRAINING PIPELINE")
    print("=" * 70)

    sanity_mode = os.getenv("QUICK_SANITY", "0") == "1"

    # 1. Load data
    examples = load_data()

    # 2. Split
    data_train, data_dev, data_holdout = stratified_split(
        examples,
        train_frac=TRAIN_FRAC,
        dev_frac=DEV_FRAC,
        seed=SPLIT_SEED,
        label_attr=OUTPUT_FIELD,
    )
    print_split_summary({
        "Train": data_train,
        "Dev": data_dev,
        "Holdout": data_holdout,
    }, label_attr=OUTPUT_FIELD)

    if sanity_mode:
        data_train = data_train[:50]
        data_dev = data_dev[:10]
        print("[QUICK SANITY] Using reduced datasets\n")

    # 3. Configure LMs
    inference_lm = dspy.LM(
        model=INFERENCE_MODEL,
        temperature=INFERENCE_TEMPERATURE,
        max_tokens=INFERENCE_MAX_TOKENS,
    )
    reflection_lm = dspy.LM(
        model=REFLECTION_MODEL,
        temperature=REFLECTION_TEMPERATURE,
        max_tokens=REFLECTION_MAX_TOKENS,
    )
    dspy.configure(lm=inference_lm)

    # 4. Configure GEPA
    optimizer = GEPA(
        metric=GEPA_METRIC,
        auto=GEPA_AUTO,
        num_threads=GEPA_NUM_THREADS,
        track_stats=True,
        track_best_outputs=True,
        reflection_minibatch_size=GEPA_REFLECTION_MINIBATCH,
        reflection_lm=reflection_lm,
        candidate_selection_strategy=GEPA_CANDIDATE_STRATEGY,
        use_merge=GEPA_USE_MERGE,
        max_merge_invocations=GEPA_MAX_MERGE,
        seed=GEPA_SEED,
    )

    # 5. Compile
    # Replace SIGNATURE_CLASS with your actual signature
    # predict = dspy.Predict(SIGNATURE_CLASS)
    predict = dspy.Predict(dspy.Signature)  # placeholder - replace with your Signature

    print("\nStarting GEPA optimization...")
    compiled_model = optimizer.compile(
        predict,
        trainset=data_train,
        valset=data_dev,
    )
    print("GEPA optimization complete.")

    # 6. Save model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / MODEL_FILENAME
    compiled_model.save(str(save_path))
    print(f"\nModel saved to {save_path}")

    # 7. Evaluate on dev set
    print("\nEvaluating on dev set...")
    predictions = [compiled_model(**ex.inputs()) for ex in data_dev]
    results = EVALUATE_FN("GEPA (dev)", data_dev, predictions)

    # 8. Save diagnostics
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = DIAGNOSTICS_DIR / "gepa_dev_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Diagnostics saved to {summary_path}")


if __name__ == "__main__":
    main()
