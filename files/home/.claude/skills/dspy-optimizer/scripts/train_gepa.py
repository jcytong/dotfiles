#!/usr/bin/env python3
"""
GEPA Training Pipeline - Default optimizer for DSPy classification.

This is the primary training script. Configure the settings below,
then run:
    python train_gepa.py

Environment variables:
    QUICK_SANITY=1   - Use small subsets for fast smoke tests.
"""

import json
import os
from pathlib import Path
from typing import List

import dspy
from dspy.teleprompt import GEPA
from dotenv import load_dotenv

load_dotenv()

# ── CONFIGURE THESE ──────────────────────────────────────────────────────────

# Your dspy.Signature class - import or define it here
# from my_project.signature import MyClassifier
# SIGNATURE_CLASS = MyClassifier

# Data source (pick one and uncomment)
# DATA_PATH = Path("data/examples.csv")
# DATA_FORMAT = "csv"  # or "jsonl"
# INPUT_FIELDS = ["field_1", "field_2"]
# LABEL_FIELD = "label"
# LABEL_MAP = {"yes": "APPROVED", "no": "REJECTED"}  # or None
# REASONING_FIELD = "reasoning"  # or None

# Labels
POSITIVE_LABEL = "APPROVED"
NEGATIVE_LABEL = "REJECTED"

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
    evaluate_predictions,
    make_gepa_metric,
)


def load_data() -> List[dspy.Example]:
    """Load data from the configured source. Customize this function."""
    raise NotImplementedError(
        "Configure DATA_PATH, DATA_FORMAT, INPUT_FIELDS, LABEL_FIELD above, "
        "then replace this function body with:\n"
        "  return load_from_csv(DATA_PATH, INPUT_FIELDS, LABEL_FIELD, LABEL_MAP, REASONING_FIELD)\n"
        "or:\n"
        "  return load_from_jsonl(DATA_PATH, INPUT_FIELDS, LABEL_FIELD, LABEL_MAP, REASONING_FIELD)"
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
    )
    print_split_summary({
        "Train": data_train,
        "Dev": data_dev,
        "Holdout": data_holdout,
    })

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
    dspy.settings.configure(lm=inference_lm)

    # 4. Build metric
    gepa_metric = make_gepa_metric(
        positive_label=POSITIVE_LABEL,
        negative_label=NEGATIVE_LABEL,
        weights=RECALL_WEIGHTS,
    )

    # 5. Configure GEPA
    optimizer = GEPA(
        metric=gepa_metric,
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

    # 6. Compile
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

    # 7. Save model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / MODEL_FILENAME
    compiled_model.save(str(save_path))
    print(f"\nModel saved to {save_path}")

    # 8. Evaluate on dev set
    print("\nEvaluating on dev set...")
    predictions: List[str] = []
    gold_labels: List[str] = []

    for ex in data_dev:
        pred = compiled_model(**ex.inputs())
        predictions.append(pred.status)
        gold_labels.append(ex.status)

    results = evaluate_predictions(
        "GEPA (dev)",
        predictions,
        gold_labels,
        positive_label=POSITIVE_LABEL,
    )

    # 9. Save diagnostics
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = DIAGNOSTICS_DIR / "gepa_dev_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Diagnostics saved to {summary_path}")


if __name__ == "__main__":
    main()
