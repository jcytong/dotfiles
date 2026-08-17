#!/usr/bin/env python3
"""
Compile and save production-ready pretrained models.

Trains the default GEPA model (and optionally others) with full budget,
then saves JSON artifacts for inference-only loading.

Usage:
    python build_pretrained.py
    python build_pretrained.py --include-alternatives
"""

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List

import dspy
from dspy.teleprompt import GEPA, COPRO, BootstrapFewShot
from dotenv import load_dotenv

load_dotenv()

from data_utils import load_from_csv, print_split_summary, stratified_split
from metrics import (
    RECALL_WEIGHTS,
    make_accuracy_metric,
    make_classification_gepa_metric,
    make_classification_metric,
)

# ── CONFIGURE THESE ──────────────────────────────────────────────────────────

# Output field — the name of the field your Signature produces as its main output.
# Classification: "status"  |  QA: "answer"  |  Extraction: "entities"
OUTPUT_FIELD = "status"

# Metric functions — configure for your task.
# Classification defaults:
GEPA_METRIC = make_classification_gepa_metric("APPROVED", "REJECTED", weights=RECALL_WEIGHTS)
SIMPLE_METRIC = make_classification_metric("APPROVED", "REJECTED", weights=RECALL_WEIGHTS)
ACCURACY_METRIC = make_accuracy_metric(output_field=OUTPUT_FIELD)
#
# Generic example (uncomment):
# from metrics import make_gepa_metric_from_fn, make_metric_from_fn
# def my_score(gold, pred): ...
# def my_feedback(gold, pred, score): ...
# GEPA_METRIC = make_gepa_metric_from_fn(my_score, my_feedback)
# SIMPLE_METRIC = make_metric_from_fn(my_score)
# ACCURACY_METRIC = make_metric_from_fn(my_score)

INFERENCE_MODEL = "openai/gpt-4o"
REFLECTION_MODEL = "openai/gpt-4o"

TRAIN_FRAC = 0.64
DEV_FRAC = 0.16
SPLIT_SEED = 42

OUTPUT_DIR = Path("pretrained")

# ── END CONFIGURATION ────────────────────────────────────────────────────────


def load_data() -> List[dspy.Example]:
    """Replace with your data loading."""
    raise NotImplementedError("Replace with your data loading logic.")


def build_gepa(
    predict: dspy.Module,
    data_train: List[dspy.Example],
    data_dev: List[dspy.Example],
) -> dspy.Module:
    """Build production GEPA model with heavy budget."""
    reflection_lm = dspy.LM(model=REFLECTION_MODEL, temperature=1.0, max_tokens=32000)

    optimizer = GEPA(
        metric=GEPA_METRIC,
        auto="heavy",
        num_threads=32,
        track_stats=True,
        track_best_outputs=True,
        reflection_minibatch_size=8,
        reflection_lm=reflection_lm,
        candidate_selection_strategy="pareto",
    )
    return optimizer.compile(predict, trainset=data_train, valset=data_dev)


def build_bsfs(
    predict: dspy.Module,
    data_train: List[dspy.Example],
) -> dspy.Module:
    """Build BootstrapFewShot model."""
    optimizer = BootstrapFewShot(
        metric=ACCURACY_METRIC,
        max_labeled_demos=16,
        max_bootstrapped_demos=4,
        metric_threshold=1,
    )
    return optimizer.compile(predict, trainset=data_train)


def build_copro(
    predict: dspy.Module,
    data_train: List[dspy.Example],
) -> dspy.Module:
    """Build COPRO model."""
    optimizer = COPRO(
        metric=SIMPLE_METRIC,
        depth=4,
        breadth=6,
        init_temperature=0.0,
    )
    return optimizer.compile(predict, trainset=data_train, eval_kwargs={})


def main():
    parser = argparse.ArgumentParser(description="Build pretrained models for production")
    parser.add_argument(
        "--include-alternatives",
        action="store_true",
        help="Also build BSFS and COPRO models alongside GEPA",
    )
    args = parser.parse_args()

    examples = load_data()
    data_train, data_dev, _ = stratified_split(
        examples, TRAIN_FRAC, DEV_FRAC, SPLIT_SEED, label_attr=OUTPUT_FIELD,
    )
    print_split_summary({"Train": data_train, "Dev": data_dev}, label_attr=OUTPUT_FIELD)

    dspy.configure(
        lm=dspy.LM(model=INFERENCE_MODEL, temperature=1.0, max_tokens=16000)
    )

    # Replace with your Signature
    predict = dspy.Predict(dspy.Signature)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Always build GEPA (default)
    print("\n=== Building GEPA (default production model) ===")
    gepa_model = build_gepa(predict, data_train, data_dev)
    gepa_path = OUTPUT_DIR / "gepa_model.json"
    gepa_model.save(str(gepa_path))
    print(f"Saved: {gepa_path}")

    if args.include_alternatives:
        # Build alternatives (only if benchmarks show they outperform GEPA)
        print("\n=== Building BootstrapFewShot (alternative) ===")
        bsfs_model = build_bsfs(predict, data_train)
        bsfs_path = OUTPUT_DIR / "bsfs_model.json"
        bsfs_model.save(str(bsfs_path))
        print(f"Saved: {bsfs_path}")

        print("\n=== Building COPRO (alternative) ===")
        extended_train = data_train + data_dev
        copro_model = build_copro(predict, extended_train)
        copro_path = OUTPUT_DIR / "copro_model.json"
        copro_model.save(str(copro_path))
        print(f"Saved: {copro_path}")

    print("\nDone. Models saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
