#!/usr/bin/env python3
"""
Benchmark all DSPy optimizers against each other.

Runs GEPA (default) plus traditional and advanced optimizers,
then prints a comparison table sorted by the configured metric.

Usage:
    python benchmark_all.py
    PHASE=1 python benchmark_all.py   # Traditional only
    PHASE=2 python benchmark_all.py   # Advanced only (GEPA, COPRO, MIPROv2)
    QUICK_SANITY=1 python benchmark_all.py  # Fast smoke test

Environment variables:
    PHASE         - "1" (traditional), "2" (advanced), "all" (default)
    QUICK_SANITY  - "1" to use small subsets
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import dspy
from dspy.teleprompt import (
    GEPA,
    COPRO,
    BootstrapFewShot,
    BootstrapFewShotWithRandomSearch,
    LabeledFewShot,
    MIPROv2,
)
from dotenv import load_dotenv

load_dotenv()

from data_utils import load_from_csv, print_split_summary, stratified_split
from metrics import (
    RECALL_WEIGHTS,
    evaluate_classification,
    make_accuracy_metric,
    make_classification_gepa_metric,
    make_classification_metric,
)

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
# Generic example (uncomment):
# from metrics import evaluate_from_fn
# def my_score(gold, pred): ...
# EVALUATE_FN = lambda name, exs, preds: evaluate_from_fn(name, exs, preds, my_score)

# Sort metric — which key in the results dict to sort by (descending).
# Classification: "recall", "precision", "f1", "accuracy"
# Generic: "mean_score"
SORT_METRIC = "recall"
SECONDARY_SORT = "precision"

# Display columns — which metrics to show in the comparison table.
# Classification default:
DISPLAY_COLUMNS = ["recall", "precision", "f1", "fn", "fp"]
# Generic alternative:
# DISPLAY_COLUMNS = ["mean_score"]

# LM configuration
INFERENCE_MODEL = "openai/gpt-4o"
INFERENCE_TEMPERATURE = 1.0
INFERENCE_MAX_TOKENS = 16000
REFLECTION_MODEL = "openai/gpt-4o"

# Data split
TRAIN_FRAC = 0.64
DEV_FRAC = 0.16
SPLIT_SEED = 42

# Output
ANALYSIS_DIR = Path("analysis")

# ── END CONFIGURATION ────────────────────────────────────────────────────────


def load_data() -> List[dspy.Example]:
    """Load data. Replace with your actual data loading."""
    raise NotImplementedError(
        "Replace this function with your data loading logic.\n"
        "Example: return load_from_csv('data.csv', ['field_1'], 'label', {'yes': 'APPROVED'}, output_field=OUTPUT_FIELD)"
    )


@dataclass
class OptimizerResult:
    name: str
    metrics: Optional[Dict[str, Any]]
    error: Optional[str] = None


def run_optimizer(
    name: str,
    compile_fn: Callable,
    eval_data: List[dspy.Example],
) -> OptimizerResult:
    """Run a single optimizer: compile, evaluate, return results."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        compiled = compile_fn()
        predictions = [compiled(**ex.inputs()) for ex in eval_data]
        metrics = EVALUATE_FN(name, eval_data, predictions)
        return OptimizerResult(name=name, metrics=metrics)
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return OptimizerResult(name=name, metrics=None, error=str(exc))


def build_optimizer_jobs(
    predict: dspy.Predict,
    data_train: List[dspy.Example],
    data_dev: List[dspy.Example],
    simple_metric: Callable,
    gepa_metric: Callable,
    accuracy_metric: Callable,
) -> Tuple[List[Tuple[str, Callable]], List[Tuple[str, Callable]]]:
    """
    Build compile functions for all optimizers.

    Returns:
        (traditional_jobs, advanced_jobs) - each is a list of (name, compile_fn).
    """
    reflection_lm = dspy.LM(model=REFLECTION_MODEL, temperature=1.0, max_tokens=32000)

    # ── Traditional (Phase 1) ───────────────────────────────
    traditional = [
        ("Baseline", lambda: predict),
        ("LabeledFewShot k=16", lambda: LabeledFewShot(k=16).compile(predict, trainset=data_train)),
        (
            "BootstrapFewShot",
            lambda: BootstrapFewShot(
                metric=accuracy_metric,
                max_labeled_demos=16,
                max_bootstrapped_demos=4,
                metric_threshold=1,
            ).compile(predict, trainset=data_train),
        ),
        (
            "BSFS+RandomSearch",
            lambda: BootstrapFewShotWithRandomSearch(
                metric=accuracy_metric,
                num_candidate_programs=16,
                max_bootstrapped_demos=4,
                max_labeled_demos=16,
            ).compile(predict, trainset=data_train),
        ),
    ]

    # ── Advanced (Phase 2) ──────────────────────────────────

    def _mipro():
        opt = MIPROv2(metric=simple_metric)
        inner_train, inner_val, _ = stratified_split(data_train, 0.8, 0.2, seed=26, label_attr=OUTPUT_FIELD)
        if not inner_val:
            inner_val = data_dev
        return opt.compile(predict, trainset=inner_train, valset=inner_val)

    def _copro():
        opt = COPRO(
            metric=simple_metric,
            breadth=2,
            depth=2,
            init_temperature=1.4,
        )
        return opt.compile(predict, trainset=data_train, eval_kwargs={})

    def _gepa():
        opt = GEPA(
            metric=gepa_metric,
            auto="medium",
            num_threads=32,
            track_stats=True,
            track_best_outputs=True,
            reflection_minibatch_size=8,
            reflection_lm=reflection_lm,
            candidate_selection_strategy="pareto",
        )
        inner_train, inner_val, _ = stratified_split(data_train, 0.8, 0.2, seed=27, label_attr=OUTPUT_FIELD)
        if not inner_val:
            inner_val = data_dev
        return opt.compile(predict, trainset=inner_train, valset=inner_val)

    advanced = [
        ("MIPROv2", _mipro),
        ("COPRO", _copro),
        ("GEPA (default)", _gepa),
    ]

    return traditional, advanced


def print_comparison_table(results: List[OptimizerResult]) -> None:
    """Print a sorted comparison table of all optimizer results."""
    valid = [r for r in results if r.metrics is not None]
    if not valid:
        print("\nNo results to compare.")
        return

    valid.sort(key=lambda r: (
        -r.metrics.get(SORT_METRIC, 0),
        -r.metrics.get(SECONDARY_SORT, 0),
    ))

    print(f"\n{'='*90}")
    print(f"OPTIMIZER COMPARISON (sorted by {SORT_METRIC})")
    print(f"{'='*90}")

    # Build header
    header = f"{'Optimizer':<30}"
    for col in DISPLAY_COLUMNS:
        header += f" {col.capitalize():<12}"
    print(header)
    print("-" * 90)

    for r in valid:
        m = r.metrics
        row = f"{r.name:<30}"
        for col in DISPLAY_COLUMNS:
            val = m.get(col, "N/A")
            if isinstance(val, float) and val <= 1.0 and col not in ("fn", "fp", "tp", "tn"):
                row += f" {val:<12.1%}"
            elif isinstance(val, float):
                row += f" {val:<12.3f}"
            else:
                row += f" {str(val):<12}"
        print(row)

    best = valid[0]
    print(f"\nBest {SORT_METRIC}: {best.name} ({best.metrics.get(SORT_METRIC, 'N/A')})")


def main():
    phase = os.getenv("PHASE", "all")
    sanity = os.getenv("QUICK_SANITY", "0") == "1"

    examples = load_data()
    data_train, data_dev, data_holdout = stratified_split(
        examples, TRAIN_FRAC, DEV_FRAC, SPLIT_SEED, label_attr=OUTPUT_FIELD,
    )
    print_split_summary(
        {"Train": data_train, "Dev": data_dev, "Holdout": data_holdout},
        label_attr=OUTPUT_FIELD,
    )

    if sanity:
        data_train = data_train[:50]
        data_dev = data_dev[:10]
        print("[QUICK SANITY] Reduced datasets\n")

    dspy.configure(
        lm=dspy.LM(model=INFERENCE_MODEL, temperature=INFERENCE_TEMPERATURE, max_tokens=INFERENCE_MAX_TOKENS)
    )

    # Replace with your signature
    # predict = dspy.Predict(MyPipeline)
    predict = dspy.Predict(dspy.Signature)

    gepa_metric = make_classification_gepa_metric("APPROVED", "REJECTED", weights=RECALL_WEIGHTS)
    simple_metric = make_classification_metric("APPROVED", "REJECTED", weights=RECALL_WEIGHTS)
    accuracy_metric = make_accuracy_metric(output_field=OUTPUT_FIELD)

    traditional, advanced = build_optimizer_jobs(
        predict, data_train, data_dev, simple_metric, gepa_metric, accuracy_metric
    )

    if phase == "1":
        jobs = traditional
    elif phase == "2":
        jobs = advanced
    else:
        jobs = traditional + advanced

    all_results: List[OptimizerResult] = []
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Traditional optimizers can run in parallel (no internal threading)
    trad_names = {name for name, _ in traditional}
    trad_jobs = [(n, f) for n, f in jobs if n in trad_names]
    seq_jobs = [(n, f) for n, f in jobs if n not in trad_names]

    if trad_jobs:
        print(f"\nRunning {len(trad_jobs)} traditional optimizers...")
        for name, compile_fn in trad_jobs:
            result = run_optimizer(name, compile_fn, data_dev)
            all_results.append(result)

    # Advanced optimizers run sequentially (they use internal parallelism)
    if seq_jobs:
        print(f"\nRunning {len(seq_jobs)} advanced optimizers sequentially...")
        for name, compile_fn in seq_jobs:
            result = run_optimizer(name, compile_fn, data_dev)
            all_results.append(result)

    print_comparison_table(all_results)

    # Save results
    summary = []
    for r in all_results:
        entry = {"optimizer": r.name}
        if r.metrics:
            entry["metrics"] = r.metrics
        if r.error:
            entry["error"] = r.error
        summary.append(entry)

    out_path = ANALYSIS_DIR / "benchmark_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
