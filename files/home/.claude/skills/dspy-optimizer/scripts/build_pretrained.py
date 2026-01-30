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
from typing import List

import dspy
from dspy.teleprompt import GEPA, COPRO, BootstrapFewShot
from dotenv import load_dotenv

load_dotenv()

from data_utils import load_from_csv, print_split_summary, stratified_split
from metrics import (
    RECALL_WEIGHTS,
    make_accuracy_metric,
    make_gepa_metric,
    make_simple_metric,
)

# ── CONFIGURE THESE ──────────────────────────────────────────────────────────

POSITIVE_LABEL = "APPROVED"
NEGATIVE_LABEL = "REJECTED"

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
    gepa_metric = make_gepa_metric(POSITIVE_LABEL, NEGATIVE_LABEL, weights=RECALL_WEIGHTS)

    optimizer = GEPA(
        metric=gepa_metric,
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
    accuracy_metric = make_accuracy_metric()
    optimizer = BootstrapFewShot(
        metric=accuracy_metric,
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
    simple_metric = make_simple_metric(POSITIVE_LABEL, NEGATIVE_LABEL, weights=RECALL_WEIGHTS)
    optimizer = COPRO(
        metric=simple_metric,
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
    data_train, data_dev, _ = stratified_split(examples, TRAIN_FRAC, DEV_FRAC, SPLIT_SEED)
    print_split_summary({"Train": data_train, "Dev": data_dev})

    dspy.settings.configure(
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
