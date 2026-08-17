"""
Portable metric functions for DSPy optimization pipelines.

Three categories:
  1. GEPA feedback metrics  - return dspy.Prediction(score=, feedback=)
  2. Simple per-example metrics - return float (for non-GEPA optimizers)
  3. Aggregate evaluation     - compute scores from predictions

Generic factories (task-agnostic):
  - make_gepa_metric_from_fn(score_fn, feedback_fn) — wraps custom scoring into GEPA format
  - make_metric_from_fn(score_fn) — wraps custom scoring for non-GEPA optimizers
  - evaluate_from_fn(name, examples, predictions, score_fn) — generic aggregate evaluation

Classification helpers (binary classification):
  - make_classification_gepa_metric() — GEPA metric with TP/FP/TN/FN feedback
  - make_classification_metric() — simple weighted metric for non-GEPA optimizers
  - evaluate_classification() — precision/recall/F1/accuracy evaluation

Usage:
    # Generic (any task):
    from metrics import make_gepa_metric_from_fn, make_metric_from_fn, evaluate_from_fn

    # Classification:
    from metrics import (
        make_classification_gepa_metric,
        make_classification_metric,
        evaluate_classification,
    )

    # Backward-compatible aliases:
    from metrics import make_gepa_metric, make_simple_metric, evaluate_predictions
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import dspy


# ---------------------------------------------------------------------------
# Configuration (classification-specific)
# ---------------------------------------------------------------------------

@dataclass
class MetricWeights:
    """
    Score weights for the four classification outcomes.

    Adjust these to shift the decision boundary:
      - Recall-priority:  tp=1.0, fn=0.0, tn=0.8, fp=0.2
      - Precision-priority: tp=1.0, fn=0.3, tn=0.8, fp=0.0
      - Balanced:          tp=1.0, fn=0.0, tn=1.0, fp=0.0
    """
    tp: float = 1.0
    fn: float = 0.0
    tn: float = 0.8
    fp: float = 0.2


RECALL_WEIGHTS = MetricWeights(tp=1.0, fn=0.0, tn=0.8, fp=0.2)
PRECISION_WEIGHTS = MetricWeights(tp=1.0, fn=0.3, tn=0.8, fp=0.0)
BALANCED_WEIGHTS = MetricWeights(tp=1.0, fn=0.0, tn=1.0, fp=0.0)


# ---------------------------------------------------------------------------
# 1a. Generic GEPA Feedback Metrics (task-agnostic)
# ---------------------------------------------------------------------------

def make_gepa_metric_from_fn(
    score_fn: Callable[[dspy.Example, dspy.Prediction], float],
    feedback_fn: Callable[[dspy.Example, dspy.Prediction, float], str],
) -> Callable:
    """
    Generic GEPA metric factory: wraps custom scoring and feedback functions
    into the 5-arg GEPA signature returning dspy.Prediction(score=, feedback=).

    Args:
        score_fn: (gold, pred) -> float score.
        feedback_fn: (gold, pred, score) -> str feedback explaining the score.

    Returns:
        A GEPA-compatible metric function.

    Example (QA exact-match):
        def qa_score(gold, pred):
            return float(gold.answer.strip().lower() == pred.answer.strip().lower())

        def qa_feedback(gold, pred, score):
            if score == 1.0:
                return f"Correct: '{pred.answer}'"
            return f"Wrong: expected '{gold.answer}', got '{pred.answer}'"

        metric = make_gepa_metric_from_fn(qa_score, qa_feedback)
    """
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        score = score_fn(gold, pred)
        feedback = feedback_fn(gold, pred, score)
        return dspy.Prediction(score=score, feedback=feedback)

    return metric


# ---------------------------------------------------------------------------
# 1b. Classification GEPA Feedback Metrics
# ---------------------------------------------------------------------------

def make_classification_gepa_metric(
    positive_label: str = "APPROVED",
    negative_label: str = "REJECTED",
    output_field: str = "status",
    weights: MetricWeights = RECALL_WEIGHTS,
    input_preview_attr: Optional[str] = None,
    input_preview_max_chars: int = 3000,
) -> Callable:
    """
    Factory that returns a GEPA-compatible metric for binary classification.

    The returned function accepts 5 args (gold, pred, trace, pred_name, pred_trace)
    and returns dspy.Prediction(score=float, feedback=str).

    Args:
        positive_label: Label value for the positive class.
        negative_label: Label value for the negative class.
        output_field: Attribute name on the Example/Prediction containing the label.
        weights: Score weights for TP/FN/TN/FP outcomes.
        input_preview_attr: Optional attribute name to include a preview of the
                            input in failure feedback (e.g., "work_experience").
        input_preview_max_chars: Max characters for the input preview.

    Returns:
        A GEPA-compatible metric function.
    """

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        gold_label = getattr(gold, output_field, "").upper()
        pred_label = getattr(pred, output_field, "").upper()
        pred_reasoning = getattr(pred, "reasoning", "No reasoning provided")

        is_positive = gold_label == positive_label.upper()
        pred_positive = pred_label == positive_label.upper()

        if is_positive and pred_positive:
            score = weights.tp
            feedback = (
                f"TRUE POSITIVE - Correctly classified as {positive_label}.\n"
                f"Reasoning: {pred_reasoning}\n"
            )
        elif is_positive and not pred_positive:
            score = weights.fn
            feedback = (
                f"FALSE NEGATIVE (CRITICAL) - Missed a {positive_label} example!\n"
                f"Model Reasoning: {pred_reasoning}\n"
                f"CRITIQUE: The reasoning failed to identify positive signals.\n"
            )
            if input_preview_attr:
                preview = _get_preview(gold, input_preview_attr, input_preview_max_chars)
                if preview:
                    feedback += f"Input Preview: {preview}\n"
        elif not is_positive and not pred_positive:
            score = weights.tn
            feedback = (
                f"TRUE NEGATIVE - Correctly classified as {negative_label}.\n"
                f"Reasoning: {pred_reasoning}\n"
            )
        else:
            score = weights.fp
            feedback = (
                f"FALSE POSITIVE - Incorrectly classified as {positive_label}.\n"
                f"Model Reasoning: {pred_reasoning}\n"
                f"CRITIQUE: Incorrectly identified positive signals.\n"
            )

        # Append ground-truth reasoning when available
        gold_reasoning = getattr(gold, "reasoning", "")
        if gold_reasoning:
            feedback += (
                f"\nGround truth reasoning (learn from this): {gold_reasoning}"
            )

        return dspy.Prediction(score=score, feedback=feedback)

    return metric


# Backward-compatible alias
make_gepa_metric = make_classification_gepa_metric


# ---------------------------------------------------------------------------
# 2a. Generic Simple Metrics (task-agnostic, for non-GEPA optimizers)
# ---------------------------------------------------------------------------

def make_metric_from_fn(
    score_fn: Callable[[dspy.Example, dspy.Prediction], float],
) -> Callable:
    """
    Generic simple metric factory for non-GEPA optimizers.

    Wraps a (gold, pred) -> float scoring function into the standard
    DSPy metric signature (example, pred, trace) -> float.

    Args:
        score_fn: (gold, pred) -> float score.

    Returns:
        A DSPy-compatible metric function.

    Example (QA exact-match):
        def qa_score(gold, pred):
            return float(gold.answer.strip().lower() == pred.answer.strip().lower())

        metric = make_metric_from_fn(qa_score)
    """
    def metric(example, pred, trace=None):
        return score_fn(example, pred)

    return metric


# ---------------------------------------------------------------------------
# 2b. Classification Simple Metrics (for non-GEPA optimizers)
# ---------------------------------------------------------------------------

def make_classification_metric(
    positive_label: str = "APPROVED",
    negative_label: str = "REJECTED",
    output_field: str = "status",
    weights: MetricWeights = RECALL_WEIGHTS,
) -> Callable:
    """
    Factory that returns a simple float metric for classification with non-GEPA optimizers.

    The returned function accepts (example, pred, trace) and returns a float.
    """

    def metric(example, pred, trace=None):
        gold_label = getattr(example, output_field, "").upper()
        pred_label = getattr(pred, output_field, "").upper()

        is_positive = gold_label == positive_label.upper()
        pred_positive = pred_label == positive_label.upper()

        if is_positive and pred_positive:
            return weights.tp
        elif is_positive and not pred_positive:
            return weights.fn
        elif not is_positive and not pred_positive:
            return weights.tn
        else:
            return weights.fp

    return metric


# Backward-compatible alias
make_simple_metric = make_classification_metric


def make_accuracy_metric(output_field: str = "status") -> Callable:
    """Simple accuracy metric: 1.0 if output fields match, 0.0 otherwise."""

    def metric(example, pred, trace=None):
        return float(
            getattr(example, output_field, "").lower()
            == getattr(pred, output_field, "").lower()
        )

    return metric


# ---------------------------------------------------------------------------
# 3a. Generic Aggregate Evaluation (task-agnostic)
# ---------------------------------------------------------------------------

def evaluate_from_fn(
    name: str,
    examples: List[dspy.Example],
    predictions: List[dspy.Prediction],
    score_fn: Callable[[dspy.Example, dspy.Prediction], float],
) -> Dict[str, Any]:
    """
    Generic evaluation: compute mean score and per-example scores.

    Args:
        name: Display name for the model/optimizer.
        examples: List of gold examples.
        predictions: List of model predictions.
        score_fn: (gold, pred) -> float scoring function.

    Returns:
        Dict with mean_score and per_example scores.

    Example:
        def qa_score(gold, pred):
            return float(gold.answer.strip().lower() == pred.answer.strip().lower())

        results = evaluate_from_fn("GEPA", examples, predictions, qa_score)
    """
    scores = [score_fn(ex, pred) for ex, pred in zip(examples, predictions)]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    print(f"\n{name} Results (n={len(scores)}):")
    print(f"  Mean Score: {mean_score:.3f}")

    return {
        "mean_score": mean_score,
        "n": len(scores),
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# 3b. Classification Aggregate Evaluation
# ---------------------------------------------------------------------------

def evaluate_classification(
    name: str,
    predictions: List[str],
    gold_labels: List[str],
    positive_label: str = "APPROVED",
) -> Dict[str, Any]:
    """
    Compute and print precision, recall, F1, accuracy from prediction lists.

    Args:
        name: Display name for the optimizer/model.
        predictions: List of predicted labels.
        gold_labels: List of ground-truth labels.
        positive_label: Which label counts as "positive".

    Returns:
        Dict with tp, fp, fn, tn, precision, recall, f1, accuracy.
    """
    pos = positive_label.upper()
    tp = sum(1 for p, g in zip(predictions, gold_labels) if p.upper() == pos and g.upper() == pos)
    fp = sum(1 for p, g in zip(predictions, gold_labels) if p.upper() == pos and g.upper() != pos)
    fn = sum(1 for p, g in zip(predictions, gold_labels) if p.upper() != pos and g.upper() == pos)
    tn = sum(1 for p, g in zip(predictions, gold_labels) if p.upper() != pos and g.upper() != pos)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(predictions) if predictions else 0.0

    print(f"\n{name} Results (n={len(predictions)}):")
    print(f"  Precision : {precision:.1%}")
    print(f"  Recall    : {recall:.1%}")
    print(f"  F1        : {f1:.3f}")
    print(f"  Accuracy  : {accuracy:.1%}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# Backward-compatible alias
evaluate_predictions = evaluate_classification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_preview(example, attr: str, max_chars: int) -> str:
    text = getattr(example, attr, "")
    if not text:
        return ""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."
