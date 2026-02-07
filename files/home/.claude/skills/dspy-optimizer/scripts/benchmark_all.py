#!/usr/bin/env python3
"""
Benchmark all DSPy optimizers against each other.

Runs GEPA (default) plus traditional and advanced optimizers,
then prints a comparison table sorted by recall.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    evaluate_predictions,
    make_accuracy_metric,
    make_gepa_metric,
    make_simple_metric,
)

# ── CONFIGURE THESE ──────────────────────────────────────────────────────────

POSITIVE_LABEL = "APPROVED"
NEGATIVE_LABEL = "REJECTED"

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
        "Example: return load_from_csv('data.csv', ['field_1'], 'label', {'yes': 'APPROVED'})"
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
    positive_label: str = "APPROVED",
) -> OptimizerResult:
    """Run a single optimizer: compile, evaluate, return results."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        compiled = compile_fn()
        preds, golds = [], []
        for ex in eval_data:
            pred = compiled(**ex.inputs())
            preds.append(pred.status)
            golds.append(ex.status)
        metrics = evaluate_predictions(name, preds, golds, positive_label)
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
        inner_train, inner_val, _ = stratified_split(data_train, 0.8, 0.2, seed=26)
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
        inner_train, inner_val, _ = stratified_split(data_train, 0.8, 0.2, seed=27)
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

    valid.sort(key=lambda r: (-r.metrics["recall"], -r.metrics["precision"]))

    print(f"\n{'='*90}")
    print("OPTIMIZER COMPARISON (sorted by recall)")
    print(f"{'='*90}")
    print(f"{'Optimizer':<30} {'Recall':<10} {'Precision':<12} {'F1':<8} {'FN':<6} {'FP':<6}")
    print("-" * 90)

    for r in valid:
        m = r.metrics
        print(
            f"{r.name:<30} "
            f"{m['recall']:<10.1%} "
            f"{m['precision']:<12.1%} "
            f"{m['f1']:<8.3f} "
            f"{m['fn']:<6} "
            f"{m['fp']:<6}"
        )

    best = valid[0]
    print(f"\nBest recall: {best.name} ({best.metrics['recall']:.1%})")


def main():
    phase = os.getenv("PHASE", "all")
    sanity = os.getenv("QUICK_SANITY", "0") == "1"

    examples = load_data()
    data_train, data_dev, data_holdout = stratified_split(
        examples, TRAIN_FRAC, DEV_FRAC, SPLIT_SEED
    )
    print_split_summary({"Train": data_train, "Dev": data_dev, "Holdout": data_holdout})

    if sanity:
        data_train = data_train[:50]
        data_dev = data_dev[:10]
        print("[QUICK SANITY] Reduced datasets\n")

    dspy.configure(
        lm=dspy.LM(model=INFERENCE_MODEL, temperature=INFERENCE_TEMPERATURE, max_tokens=INFERENCE_MAX_TOKENS)
    )

    # Replace with your signature
    # predict = dspy.Predict(MyClassifier)
    predict = dspy.Predict(dspy.Signature)

    gepa_metric = make_gepa_metric(POSITIVE_LABEL, NEGATIVE_LABEL, weights=RECALL_WEIGHTS)
    simple_metric = make_simple_metric(POSITIVE_LABEL, NEGATIVE_LABEL, weights=RECALL_WEIGHTS)
    accuracy_metric = make_accuracy_metric()

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
            result = run_optimizer(name, compile_fn, data_dev, POSITIVE_LABEL)
            all_results.append(result)

    # Advanced optimizers run sequentially (they use internal parallelism)
    if seq_jobs:
        print(f"\nRunning {len(seq_jobs)} advanced optimizers sequentially...")
        for name, compile_fn in seq_jobs:
            result = run_optimizer(name, compile_fn, data_dev, POSITIVE_LABEL)
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
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
