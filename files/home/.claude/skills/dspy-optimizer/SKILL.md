---
name: dspy-optimizer
description: "Build, train, and optimize DSPy classification pipelines. Use when creating a new DSPy Signature for classification, training/optimizing a DSPy program with teleprompt optimizers (GEPA, BootstrapFewShot, COPRO, MIPROv2, LabeledFewShot, KNNFewShot, BootstrapFewShotWithRandomSearch), evaluating model performance (recall, precision, F1), running hyperparameter sweeps, building ensemble/union strategies, performing error analysis (false negatives/positives), or compiling pretrained models for production deployment. Triggers on: dspy, optimize prompt, train classifier, company screener, prompt optimization, teleprompt, few-shot, bootstrap, copro, mipro, gepa, recall optimization, precision optimization, ensemble strategy, model evaluation, hyperparameter sweep, pretrained model, dspy signature."
---

# DSPy Optimizer

Procedural guide for building, training, evaluating, and deploying DSPy classification pipelines.

**Default optimizer: GEPA** (Generative Prompt Adaptation). GEPA is the recommended first choice for all new pipelines. It uses reflection-based prompt optimization with textual feedback from your metric function. Other optimizers (BootstrapFewShot, COPRO, MIPROv2, etc.) should be benchmarked against GEPA and only preferred when they demonstrably outperform it in experimentation.

## Portable Scripts

All code is in portable, project-agnostic scripts under `scripts/`. Copy these into your project and configure the `CONFIGURE THESE` section at the top of each file.

| Script | Purpose | Workflow Step |
|--------|---------|---------------|
| [data_utils.py](scripts/data_utils.py) | Load CSV/JSONL, stratified split | Step 2 |
| [metrics.py](scripts/metrics.py) | GEPA feedback metrics, simple metrics, evaluation | Step 3 |
| [train_gepa.py](scripts/train_gepa.py) | Default GEPA training pipeline | Step 4 |
| [benchmark_all.py](scripts/benchmark_all.py) | Compare GEPA vs all other optimizers | Step 5 |
| [sweep.py](scripts/sweep.py) | Hyperparameter grid search | Step 5a |
| [evaluate_holdout.py](scripts/evaluate_holdout.py) | Holdout eval + error analysis | Steps 6 & 8 |
| [ensemble.py](scripts/ensemble.py) | Union/cascade multi-model strategy | Step 7 |
| [build_pretrained.py](scripts/build_pretrained.py) | Compile production JSON artifacts | Step 9 |
| [inference.py](scripts/inference.py) | Production inference wrapper | Step 9 |

## Workflow

```
1. Define Signature  -->  2. Prepare Data  -->  3. Define Metrics (with feedback for GEPA)
        |                                              |
        v                                              v
4. Train with GEPA (default)  -->  5. Benchmark Other Optimizers (optional)
        |                                              |
        v                                              v
6. Evaluate on Holdout  -->  7. Ensemble Strategy  -->  8. Error Analysis
        |
        v
9. Compile & Deploy to Production
```

## Step 1: Define a DSPy Signature

Create a `dspy.Signature` subclass with a detailed docstring prompt and typed input/output fields.

```python
import dspy

class MyClassifier(dspy.Signature):
    """
    [Detailed role description and instructions in markdown]

    **Classification Rules:**
    [Numbered rules with examples for each class]

    **Examples:**
    [3-4 examples covering: clear positive, clear negative, borderline cases]
    """
    # Inputs
    field_1: str = dspy.InputField(description="Description of field 1")
    # Outputs
    status: str = dspy.OutputField(description="CLASS_A or CLASS_B")
    confidence: float = dspy.OutputField(description="Confidence 0.0-1.0")
    reasoning: str = dspy.OutputField(description="Explanation for the decision")
```

Key prompt design patterns:
- Embed the full classification rubric in the docstring (not in field descriptions)
- Include confidence score calibration guidance
- Add 3-4 concrete examples covering approve, reject, and borderline cases
- Use markdown formatting with headers, bullet lists, and code blocks

## Step 2: Prepare Data

> Full implementation: [scripts/data_utils.py](scripts/data_utils.py)

Load labeled examples into `dspy.Example` objects with stratified train/dev/holdout splits.

```python
from data_utils import load_from_csv, stratified_split, print_split_summary

examples = load_from_csv(
    path="data/examples.csv",
    input_fields=["field_1"],
    label_field="label",
    label_map={"yes": "APPROVED", "no": "REJECTED"},
    reasoning_field="reasoning",  # optional
)

data_train, data_dev, data_holdout = stratified_split(examples, train_frac=0.64, dev_frac=0.16, seed=42)
print_split_summary({"Train": data_train, "Dev": data_dev, "Holdout": data_holdout})
```

Also supports JSONL via `load_from_jsonl()`. Split guidelines:
- **Train**: ~64% for optimizer compilation
- **Dev**: ~16% for optimizer tuning and comparison
- **Holdout**: ~20% untouched until final evaluation
- Always use a fixed seed for reproducibility

## Step 3: Define Metric Functions

> Full implementation: [scripts/metrics.py](scripts/metrics.py)

Three kinds of metrics: **GEPA feedback metrics** (score + textual feedback), **simple metrics** (float, for non-GEPA optimizers), and **aggregate evaluation** (precision/recall/F1).

### GEPA Feedback Metrics

GEPA requires a metric that accepts 5 arguments `(gold, pred, trace, pred_name, pred_trace)` and returns `dspy.Prediction(score=float, feedback=str)`. Use the factory from `metrics.py`:

```python
from metrics import make_gepa_metric, RECALL_WEIGHTS, PRECISION_WEIGHTS

# Recall-priority (default): TP=1.0, FN=0.0, TN=0.8, FP=0.2
gepa_metric = make_gepa_metric(
    positive_label="APPROVED",
    negative_label="REJECTED",
    weights=RECALL_WEIGHTS,
    input_preview_attr="field_1",  # include input preview in FN feedback
)
```

The `MetricWeights` dataclass controls the decision boundary:
- **RECALL_WEIGHTS** (default): `tp=1.0, fn=0.0, tn=0.8, fp=0.2` - shifts boundary so model approves when P(positive) > ~37.5%
- **PRECISION_WEIGHTS**: `tp=1.0, fn=0.3, tn=0.8, fp=0.0` - penalizes false positives heavily
- **BALANCED_WEIGHTS**: `tp=1.0, fn=0.0, tn=1.0, fp=0.0` - standard accuracy

Key feedback design principles:
- FN feedback must be rich (include input preview, model reasoning, critique)
- Include ground-truth reasoning when available
- Be specific in critiques (what patterns were missed)
- Score asymmetry encodes priority

### Simple Metrics (non-GEPA)

```python
from metrics import make_simple_metric, make_accuracy_metric

simple_recall = make_simple_metric("APPROVED", "REJECTED", weights=RECALL_WEIGHTS)
accuracy = make_accuracy_metric()
```

### Aggregate Evaluation

```python
from metrics import evaluate_predictions
results = evaluate_predictions("Model Name", predictions, gold_labels, positive_label="APPROVED")
# prints: Precision, Recall, F1, Accuracy, confusion matrix
```

## Step 4: Train with GEPA (Default)

> Full implementation: [scripts/train_gepa.py](scripts/train_gepa.py)

GEPA uses a reflection LM to analyze prediction failures via textual feedback, then algorithmically rewrites instructions.

```python
from dspy.teleprompt import GEPA

reflection_lm = dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000)

optimizer = GEPA(
    metric=gepa_metric,                   # from Step 3
    auto="medium",                         # budget: "light", "medium", "heavy"
    num_threads=32,
    track_stats=True,
    track_best_outputs=True,
    reflection_minibatch_size=8,
    reflection_lm=reflection_lm,
    candidate_selection_strategy="pareto", # explore recall-precision tradeoff
)

compiled_model = optimizer.compile(predict, trainset=data_train, valset=data_dev)
compiled_model.save("pretrained/gepa_model.json")
```

### GEPA Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `metric` | *required* | Feedback metric returning `dspy.Prediction(score=, feedback=)` |
| `auto` | `None` | Budget preset: `"light"`, `"medium"`, `"heavy"`. Mutually exclusive with `max_full_evals` / `max_metric_calls` |
| `max_full_evals` | `None` | Number of full evaluation passes. Alternative to `auto` |
| `max_metric_calls` | `None` | Raw budget cap. Alternative to `auto` |
| `reflection_lm` | *required* | LM for reflection/instruction rewriting. Use a strong model |
| `reflection_minibatch_size` | `3` | Feedback examples per reflection batch. Higher = richer context |
| `candidate_selection_strategy` | `"pareto"` | `"pareto"` explores recall-precision tradeoff; `"current_best"` optimizes single metric |
| `num_threads` | `None` | Parallel evaluation threads |
| `track_stats` | `False` | Enable `detailed_results` on compiled model |
| `track_best_outputs` | `False` | Track best outputs per example (requires `track_stats=True`) |
| `use_merge` | `True` | Merge-based optimization (combines best candidates) |
| `max_merge_invocations` | `5` | Cap on merge operations |
| `skip_perfect_score` | `True` | Skip reflection on already-perfect examples |
| `component_selector` | `"round_robin"` | How GEPA picks which predictor to optimize |
| `seed` | `0` | Random seed for reproducibility |

### Budget Guidance

- `auto="light"`: Quick exploration. Good for prototyping.
- `auto="medium"`: Recommended default.
- `auto="heavy"`: Thorough optimization. Use for final production training.
- `max_full_evals=N`: Manual control. Each eval = `len(trainset) + len(valset)` calls.

### Why GEPA is the Default

1. **Feedback-driven**: Reads textual feedback explaining *why* predictions failed, enabling targeted instruction improvement.
2. **Pareto exploration**: With `candidate_selection_strategy="pareto"`, explores the full recall-precision tradeoff.
3. **Reflection LM**: Uses a separate strong model to analyze failures and rewrite instructions.
4. **Proven results**: Produces the best recall while maintaining acceptable precision in production use.

## Step 5: Benchmark Other Optimizers (Optional)

> Full implementation: [scripts/benchmark_all.py](scripts/benchmark_all.py)

After GEPA, optionally benchmark other optimizers. Only prefer an alternative if it demonstrably outperforms GEPA on your holdout set.

```bash
python benchmark_all.py                  # all optimizers
PHASE=1 python benchmark_all.py          # traditional only (fast)
PHASE=2 python benchmark_all.py          # advanced only (GEPA, COPRO, MIPROv2)
QUICK_SANITY=1 python benchmark_all.py   # smoke test
```

See [references/optimizers.md](references/optimizers.md) for the full optimizer catalog.

| Optimizer | Cost | Best For |
|-----------|------|----------|
| **GEPA (default)** | **High** | **Feedback-driven prompt optimization** |
| Baseline | Free | Establishing floor |
| LabeledFewShot | Low | Quick demo injection |
| BootstrapFewShot | Medium | General best single model |
| BSFS + RandomSearch | Medium-High | Robust demo selection |
| KNNFewShot | Medium | Semantic demo retrieval |
| MIPROv2 | High | Instruction optimization |
| COPRO | High | Recall/precision tuning |

## Step 5a: Hyperparameter Sweep

> Full implementation: [scripts/sweep.py](scripts/sweep.py)

```bash
python sweep.py --optimizer gepa    # GEPA sweep (default)
python sweep.py --optimizer copro   # COPRO sweep
python sweep.py --optimizer bsfs    # BootstrapFewShot sweep
python sweep.py --optimizer all     # Everything
```

GEPA sweep grid (budget x minibatch x strategy):
```
light / 3 / pareto
medium / 3 / pareto
medium / 8 / pareto
medium / 8 / current_best
heavy / 8 / pareto
```

## Step 6: Evaluate on Holdout

> Full implementation: [scripts/evaluate_holdout.py](scripts/evaluate_holdout.py)

Run the best config on the holdout set **once** for final numbers. Never tune on holdout.

```bash
python evaluate_holdout.py --model pretrained/gepa_model.json
python evaluate_holdout.py --model pretrained/gepa_model.json --error-analysis
```

## Step 7: Ensemble / Union Strategy

> Full implementation: [scripts/ensemble.py](scripts/ensemble.py)

Combine complementary models with confidence thresholds:

```bash
python ensemble.py \
    --model-a pretrained/gepa_model.json --threshold-a 0.70 \
    --model-b pretrained/copro_model.json --threshold-b 0.55
```

Logic:
- **Stage 1**: High-confidence model (model A) with strict threshold -> approve if confident
- **Stage 2**: High-recall model (model B) with looser threshold -> fallback approve
- **Default**: Reject if neither model approves

## Step 8: Error Analysis

> Included in: [scripts/evaluate_holdout.py](scripts/evaluate_holdout.py) (`--error-analysis` flag)

```bash
python evaluate_holdout.py --model pretrained/gepa_model.json \
    --error-analysis --fn-limit 10 --fp-limit 10
```

Use error analysis to:
- Identify patterns in misclassified examples
- Refine signature docstring rules and examples
- Add new rejection/approval rules
- Then **recompile all models** (prompt text changed)

## Step 9: Compile & Deploy to Production

> Build script: [scripts/build_pretrained.py](scripts/build_pretrained.py)
> Inference wrapper: [scripts/inference.py](scripts/inference.py)

### Build production models

```bash
python build_pretrained.py                       # GEPA only (default)
python build_pretrained.py --include-alternatives # + BSFS, COPRO
```

### Production inference

```python
from inference import Classifier

clf = Classifier("pretrained/gepa_model.json")
result = clf.predict(field_1="input text here")
print(result)  # {"status": "APPROVED", "confidence": 0.92, "reasoning": "..."}
```

Or via CLI:
```bash
python inference.py --model pretrained/gepa_model.json --input "some text"
python inference.py --model pretrained/gepa_model.json --input-json '{"field_1": "text"}'
```

Key deployment rules:
- **Default to GEPA**: Unless benchmarks show another optimizer outperforms it
- **One-way dependency**: `experiments/ -> production/` (OK), never the reverse
- Production code loads JSON only, never imports from experiments
- LM is configured at inference time (can differ from training LM)
- After any prompt change, **all models must be recompiled**

## Environment Flags

| Variable | Effect |
|----------|--------|
| `QUICK_SANITY=1` | Limit training/eval to small subsets for fast smoke-testing |
| `SKIP_OPTIMIZERS=1` | Skip heavy optimizer sweeps when module is imported |
| `PHASE=1\|2\|all` | Control which optimizer phases run in benchmark_all.py |

## Project Layout Convention

```
experiments/
  my_classifier/
    data/
      examples.csv                    # Labeled dataset
    gepa_experiments.py               # GEPA-focused experiment (default)
    all_optimizers_experiments.py      # Benchmark all optimizers
    pretrained/
      gepa_model.json                 # GEPA pretrained model (default)
    analysis/
      holdout_diagnostics.jsonl       # Per-example results
      holdout_summary.json            # Aggregate metrics
      benchmark_results.json          # All-optimizer comparison

production/ml/
  my_classifier/
    core.py                           # Production Signature + inference wrapper
    pretrained/
      gepa_model.json                 # Default production model
      bsfs_model.json                 # Alternative (if outperforms GEPA)
```
