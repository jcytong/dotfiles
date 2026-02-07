#!/usr/bin/env python3
"""
Hyperparameter sweep for DSPy optimizers.

Sweeps GEPA configurations first (the default), then optionally
sweeps COPRO and BootstrapFewShot for comparison.

Usage:
    python sweep.py
    python sweep.py --optimizer gepa
    python sweep.py --optimizer copro
    python sweep.py --optimizer bsfs
    python sweep.py --optimizer all
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import dspy
from dspy.teleprompt import GEPA, COPRO, BootstrapFewShot
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

INFERENCE_MODEL = "openai/gpt-4o"
REFLECTION_MODEL = "openai/gpt-4o"

TRAIN_FRAC = 0.64
DEV_FRAC = 0.16
SPLIT_SEED = 42

OUTPUT_DIR = Path("analysis")

# ── END CONFIGURATION ────────────────────────────────────────────────────────


def load_data() -> List[dspy.Example]:
    """Replace with your data loading."""
    raise NotImplementedError("Replace with your data loading logic.")


def test_config(
    name: str,
    compile_fn: Callable,
    eval_data: List[dspy.Example],
    positive_label: str,
) -> Optional[Dict[str, Any]]:
    """Compile and evaluate a single config, return metrics or None on error."""
    try:
        compiled = compile_fn()
        preds = [compiled(**ex.inputs()).status for ex in eval_data]
        golds = [ex.status for ex in eval_data]
        return evaluate_predictions(name, preds, golds, positive_label)
    except Exception as exc:
        print(f"  {name} FAILED: {exc}")
        return None


def sweep_gepa(
    predict: dspy.Predict,
    data_train: List[dspy.Example],
    data_dev: List[dspy.Example],
    gepa_metric: Callable,
) -> List[Tuple[str, Dict, Dict]]:
    """Sweep GEPA configurations."""
    reflection_lm = dspy.LM(model=REFLECTION_MODEL, temperature=1.0, max_tokens=32000)
    results = []

    configs = [
        {"auto": "light", "minibatch": 3, "strategy": "pareto"},
        {"auto": "medium", "minibatch": 3, "strategy": "pareto"},
        {"auto": "medium", "minibatch": 8, "strategy": "pareto"},
        {"auto": "medium", "minibatch": 8, "strategy": "current_best"},
        {"auto": "heavy", "minibatch": 8, "strategy": "pareto"},
    ]

    for cfg in configs:
        name = f"GEPA auto={cfg['auto']} mb={cfg['minibatch']} {cfg['strategy']}"

        def _compile(c=cfg):
            opt = GEPA(
                metric=gepa_metric,
                auto=c["auto"],
                num_threads=32,
                track_stats=True,
                reflection_minibatch_size=c["minibatch"],
                reflection_lm=reflection_lm,
                candidate_selection_strategy=c["strategy"],
            )
            inner_train, inner_val, _ = stratified_split(data_train, 0.8, 0.2, seed=27)
            if not inner_val:
                inner_val = data_dev
            return opt.compile(predict, trainset=inner_train, valset=inner_val)

        metrics = test_config(name, _compile, data_dev, POSITIVE_LABEL)
        if metrics:
            results.append(("GEPA", cfg, metrics))

    return results


def sweep_copro(
    predict: dspy.Predict,
    data_train: List[dspy.Example],
    data_dev: List[dspy.Example],
    simple_metric: Callable,
) -> List[Tuple[str, Dict, Dict]]:
    """Sweep COPRO configurations."""
    results = []

    for depth, breadth, temp in itertools.product([1, 2, 3], [2, 3, 4], [0.0, 0.3]):
        cfg = {"depth": depth, "breadth": breadth, "temperature": temp}
        name = f"COPRO d={depth} b={breadth} t={temp}"

        def _compile(d=depth, b=breadth, t=temp):
            return COPRO(
                metric=simple_metric, depth=d, breadth=b, init_temperature=t
            ).compile(predict, trainset=data_train, eval_kwargs={})

        metrics = test_config(name, _compile, data_dev, POSITIVE_LABEL)
        if metrics:
            results.append(("COPRO", cfg, metrics))

    return results


def sweep_bsfs(
    predict: dspy.Predict,
    data_train: List[dspy.Example],
    data_dev: List[dspy.Example],
    accuracy_metric: Callable,
) -> List[Tuple[str, Dict, Dict]]:
    """Sweep BootstrapFewShot configurations."""
    results = []

    for labeled, boot in itertools.product([16, 32, 48], [4, 8]):
        cfg = {"labeled": labeled, "bootstrapped": boot}
        name = f"BSFS L={labeled} B={boot}"

        def _compile(l=labeled, b=boot):
            return BootstrapFewShot(
                metric=accuracy_metric,
                max_labeled_demos=l,
                max_bootstrapped_demos=b,
                metric_threshold=1,
            ).compile(predict, trainset=data_train)

        metrics = test_config(name, _compile, data_dev, POSITIVE_LABEL)
        if metrics:
            results.append(("BSFS", cfg, metrics))

    return results


def main():
    parser = argparse.ArgumentParser(description="DSPy optimizer hyperparameter sweep")
    parser.add_argument(
        "--optimizer",
        choices=["gepa", "copro", "bsfs", "all"],
        default="gepa",
        help="Which optimizer to sweep (default: gepa)",
    )
    args = parser.parse_args()

    examples = load_data()
    data_train, data_dev, _ = stratified_split(examples, TRAIN_FRAC, DEV_FRAC, SPLIT_SEED)
    print_split_summary({"Train": data_train, "Dev": data_dev})

    dspy.configure(
        lm=dspy.LM(model=INFERENCE_MODEL, temperature=1.0, max_tokens=16000)
    )

    # Replace with your Signature
    predict = dspy.Predict(dspy.Signature)

    gepa_metric = make_gepa_metric(POSITIVE_LABEL, NEGATIVE_LABEL, weights=RECALL_WEIGHTS)
    simple_metric = make_simple_metric(POSITIVE_LABEL, NEGATIVE_LABEL, weights=RECALL_WEIGHTS)
    accuracy_metric = make_accuracy_metric()

    all_results = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.optimizer in ("gepa", "all"):
        print("\n=== GEPA SWEEP ===")
        all_results.extend(sweep_gepa(predict, data_train, data_dev, gepa_metric))

    if args.optimizer in ("copro", "all"):
        print("\n=== COPRO SWEEP ===")
        all_results.extend(sweep_copro(predict, data_train, data_dev, simple_metric))

    if args.optimizer in ("bsfs", "all"):
        print("\n=== BSFS SWEEP ===")
        all_results.extend(sweep_bsfs(predict, data_train, data_dev, accuracy_metric))

    # Sort by recall then precision
    all_results.sort(key=lambda x: (-x[2]["recall"], -x[2]["precision"]))

    print(f"\n{'='*80}")
    print("SWEEP RESULTS (sorted by recall)")
    print(f"{'='*80}")
    for optimizer, cfg, metrics in all_results:
        cfg_str = " ".join(f"{k}={v}" for k, v in cfg.items())
        print(
            f"  {optimizer:<8} {cfg_str:<40} "
            f"R={metrics['recall']:.1%} P={metrics['precision']:.1%} F1={metrics['f1']:.3f}"
        )

    out_path = OUTPUT_DIR / f"sweep_{args.optimizer}_results.json"
    serializable = [
        {"optimizer": opt, "config": cfg, "metrics": m}
        for opt, cfg, m in all_results
    ]
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
