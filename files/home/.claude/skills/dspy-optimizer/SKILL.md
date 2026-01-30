---
name: dspy-optimizer
description: "Build, train, and optimize DSPy classification pipelines. Use when creating a new DSPy Signature for classification, training/optimizing a DSPy program with teleprompt optimizers (GEPA, BootstrapFewShot, COPRO, MIPROv2, LabeledFewShot, KNNFewShot, BootstrapFewShotWithRandomSearch), evaluating model performance (recall, precision, F1), running hyperparameter sweeps, building ensemble/union strategies, performing error analysis (false negatives/positives), or compiling pretrained models for production deployment. Triggers on: dspy, optimize prompt, train classifier, company screener, prompt optimization, teleprompt, few-shot, bootstrap, copro, mipro, gepa, recall optimization, precision optimization, ensemble strategy, model evaluation, hyperparameter sweep, pretrained model, dspy signature."
---

# DSPy Optimizer

Procedural guide for building, training, evaluating, and deploying DSPy classification pipelines. Based on proven patterns from `experiments/eir_screener/` and `experiments/company_screener/`.

**Default optimizer: GEPA** (Generative Prompt Adaptation). GEPA is the recommended first choice for all new pipelines. It uses reflection-based prompt optimization with textual feedback from your metric function. Other optimizers (BootstrapFewShot, COPRO, MIPROv2, etc.) should be benchmarked against GEPA and only preferred when they demonstrably outperform it in experimentation.

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

    **Output Format:**
    ```json
    {
      "status": "CLASS_A | CLASS_B",
      "confidence": 0.0,
      "reasoning": "<explanation>"
    }
    ```

    **Classification Rules:**
    [Numbered rules with examples for each class]

    **Examples:**
    [3-4 examples covering: clear Class A, clear Class B, borderline cases]
    """
    # Inputs
    field_1: str = dspy.InputField(description="Description of field 1")
    field_2: str = dspy.InputField(description="Description of field 2")
    # Outputs
    status: str = dspy.OutputField(description="CLASS_A or CLASS_B")
    confidence: float = dspy.OutputField(description="Confidence 0.0-1.0")
    reasoning: str = dspy.OutputField(description="Explanation for the decision")
```

Key prompt design patterns:
- Embed the full classification rubric in the docstring (not in field descriptions)
- Include confidence score calibration guidance (e.g., 0.9+ = clear-cut, 0.5-0.69 = borderline)
- Add 3-4 concrete examples covering approve, reject, borderline, and dynamic rule cases
- Use markdown formatting with headers, bullet lists, and code blocks for structure
- Include a `[DYNAMIC LIST]` section for rules that change between runs (e.g., portfolio conflicts)

## Step 2: Prepare Data

Load labeled examples into `dspy.Example` objects with stratified train/dev/holdout splits.

```python
import csv
import random
from pathlib import Path

# Load from CSV
examples = []
with Path("data/examples.csv").open("r") as f:
    for row in csv.DictReader(f):
        ex = dspy.Example(
            field_1=row["field_1"],
            field_2=row["field_2"],
            status=row["label"],
        ).with_inputs("field_1", "field_2")  # Mark which fields are inputs
        examples.append(ex)

# Stratified split preserving class ratios
def stratified_split(examples, train_frac=0.20, dev_frac=0.16, seed=25):
    rng = random.Random(seed)
    buckets = {}
    for ex in examples:
        buckets.setdefault(ex.status, []).append(ex)
    for items in buckets.values():
        rng.shuffle(items)

    def allocate(frac):
        out = []
        for status, items in buckets.items():
            take = max(1, int(len(items) * frac)) if items else 0
            out.extend(items[:take])
            buckets[status] = items[take:]
        rng.shuffle(out)
        return out

    train = allocate(train_frac)
    dev = allocate(dev_frac)
    holdout = [ex for remaining in buckets.values() for ex in remaining]
    rng.shuffle(holdout)
    return train, dev, holdout

data_train, data_dev, data_holdout = stratified_split(examples)
```

Split guidelines:
- **Train**: 20-25% for optimizer compilation (demos + bootstrapping)
- **Dev**: ~15-20% for optimizer tuning and comparison
- **Holdout**: ~60% untouched until final evaluation
- Always use a fixed `random.seed()` for reproducibility
- Print split summaries showing class counts and positive-class percentage

## Step 3: Define Metric Functions

Three kinds of metrics are needed: **GEPA feedback metrics** (per-example, return score + textual feedback), **simple optimization metrics** (per-example, return float, for non-GEPA optimizers), and **evaluation metrics** (aggregate, for reporting).

### GEPA Feedback Metrics (per-example, return `dspy.Prediction(score=..., feedback=...)`)

GEPA requires a metric that accepts 5 arguments `(gold, pred, trace, pred_name, pred_trace)` and returns a `dspy.Prediction` with `score` (float) and `feedback` (string). The feedback is used by GEPA's reflection LM to understand *why* predictions failed and propose improved instructions.

```python
import dspy

def recall_metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    GEPA-compatible metric that returns score + textual feedback.

    GEPA uses the feedback to reflect on failures and algorithmically
    improve instructions through its reflection LM.

    Scoring strategy (recall-priority):
    - TP = 1.0 (max reward: correctly identified positive)
    - FN = 0.0 (max penalty: missed a positive - CRITICAL)
    - TN = 0.8 (high reward, but less than TP)
    - FP = 0.2 (small reward: better than FN, but not ideal)

    This shifts the decision boundary so the model approves when
    P(positive) > ~37.5% instead of >50%, boosting recall.
    """
    pred_reasoning = getattr(pred, 'reasoning', 'No reasoning provided')

    if gold.status == "APPROVED" and pred.status == "APPROVED":
        score = 1.0
        feedback = (
            f"TRUE POSITIVE - Correctly identified strong candidate.\n"
            f"Reasoning: {pred_reasoning}\n"
        )
    elif gold.status == "APPROVED" and pred.status == "REJECTED":
        score = 0.0
        # Rich feedback is critical for GEPA reflection - explain what was missed
        gold_reasoning = getattr(gold, 'reasoning', '')
        work_preview = gold.work_experience[:3000] if hasattr(gold, 'work_experience') else ''
        feedback = (
            f"FALSE NEGATIVE (CRITICAL) - Missed a high-quality candidate!\n"
            f"Work Experience Preview: {work_preview}\n"
            f"Model Reasoning: {pred_reasoning}\n"
            f"CRITIQUE: The reasoning failed to acknowledge positive signals.\n"
        )
        if gold_reasoning:
            feedback += f"Ground truth reasoning: {gold_reasoning}\n"
    elif gold.status == "REJECTED" and pred.status == "REJECTED":
        score = 0.8
        feedback = (
            f"TRUE NEGATIVE - Correctly rejected weak candidate.\n"
            f"Reasoning: {pred_reasoning}\n"
        )
    else:  # FP
        score = 0.2
        feedback = (
            f"FALSE POSITIVE - Approved a candidate who should be rejected.\n"
            f"Model Reasoning: {pred_reasoning}\n"
            f"CRITIQUE: Incorrectly identified positive signals where there were none.\n"
        )

    # Append ground truth reasoning for GEPA to learn from
    extra = getattr(gold, "reasoning", "")
    if extra:
        feedback += f"\nGround truth context: {extra}"

    return dspy.Prediction(score=score, feedback=feedback)
```

Key feedback design principles for GEPA:
- **FN feedback must be rich**: Include work experience preview, model reasoning, and a critique of what was missed. GEPA uses this to revise instructions.
- **Include ground truth reasoning** when available so GEPA can learn the "why" behind labels.
- **Be specific in critiques**: Instead of "wrong answer", explain what patterns were missed or incorrectly identified.
- **Score asymmetry encodes priority**: The gap between TP (1.0) and FN (0.0) being larger than TN (0.8) and FP (0.2) forces recall optimization.

### Simple Optimization Metrics (per-example, return float)

For non-GEPA optimizers (BootstrapFewShot, MIPROv2, etc.) that don't use textual feedback:

```python
def validate_answer(example, pred, trace=None):
    """Binary accuracy - used by BootstrapFewShot, MIPROv2."""
    return example.status.lower() == pred.status.lower()

def recall_metric(example, pred, trace=None):
    """Recall-focused - used by COPRO when minimizing false negatives is critical."""
    if example.status == "APPROVED" and pred.status == "APPROVED":
        return 1.0   # True positive
    elif example.status == "APPROVED" and pred.status == "REJECTED":
        return 0.0   # False negative - HEAVILY penalize
    elif example.status == "REJECTED" and pred.status == "REJECTED":
        return 0.6   # True negative
    else:
        return 0.4   # False positive - acceptable cost

def precision_metric(example, pred, trace=None):
    """Precision-focused - penalizes false positives more."""
    if example.status == "APPROVED" and pred.status == "APPROVED":
        return 1.0
    elif example.status == "REJECTED" and pred.status == "REJECTED":
        return 0.8
    elif example.status == "REJECTED" and pred.status == "APPROVED":
        return 0.0   # False positive - HEAVILY penalize
    else:
        return 0.3   # False negative
```

Metric selection:
- `recall_metric_with_feedback` for GEPA (default - always use this first)
- `validate_answer` for balanced accuracy (BootstrapFewShot, MIPROv2)
- `recall_metric` for recall-focused non-GEPA optimizers (COPRO)
- `precision_metric` for high-precision filtering (minimize false positives)

### Evaluation Metrics (aggregate, for reporting)

```python
def evaluate_optimizer(name, predictions, gold_labels):
    tp = sum(1 for p, a in zip(predictions, gold_labels) if p == a == "APPROVED")
    fp = sum(1 for p, a in zip(predictions, gold_labels) if p == "APPROVED" and a == "REJECTED")
    fn = sum(1 for p, a in zip(predictions, gold_labels) if p == "REJECTED" and a == "APPROVED")
    tn = sum(1 for p, a in zip(predictions, gold_labels) if p == "REJECTED" and a == "REJECTED")
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = (tp + tn) / len(predictions)
    print(f"{name}: P={precision:.1%} R={recall:.1%} F1={f1:.3f} TP={tp} FP={fp} FN={fn}")
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
```

## Step 4: Train with GEPA (Default)

GEPA (Generative Prompt Adaptation) is the default optimizer. It uses a reflection LM to analyze prediction failures via textual feedback from your metric, then algorithmically rewrites instructions to fix those failures. This makes it uniquely powerful for classification tasks where you can describe *why* predictions are wrong.

### GEPA Configuration

```python
import dspy
from dspy.teleprompt import GEPA

# Configure inference LM
dspy.settings.configure(lm=dspy.LM("openai/gpt-5-nano", temperature=1.0, max_tokens=16000))
predict = dspy.Predict(MyClassifier)

# Configure GEPA with a separate, stronger reflection LM
reflection_lm = dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000)

optimizer = GEPA(
    metric=recall_metric_with_feedback,  # Must return dspy.Prediction(score=, feedback=)
    auto="medium",                        # Budget: "light", "medium", or "heavy"
    num_threads=32,                       # Parallel evaluation threads
    track_stats=True,                     # Enable detailed result tracking
    track_best_outputs=True,              # Track best outputs per example
    reflection_minibatch_size=8,          # Batch size for feedback reflection
    reflection_lm=reflection_lm,          # Stronger model for prompt rewriting
    candidate_selection_strategy="pareto", # Explore recall-precision tradeoff
)

compiled_model = optimizer.compile(
    predict,
    trainset=data_train,
    valset=data_dev,  # Separate valset for Pareto tracking (recommended)
)
```

### GEPA Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `metric` | *required* | Feedback metric returning `dspy.Prediction(score=, feedback=)` |
| `auto` | `None` | Budget preset: `"light"`, `"medium"`, `"heavy"`. Mutually exclusive with `max_full_evals` / `max_metric_calls` |
| `max_full_evals` | `None` | Number of full evaluation passes. Alternative to `auto` |
| `max_metric_calls` | `None` | Raw budget cap on total metric invocations. Alternative to `auto` |
| `reflection_lm` | *required* | LM for reflection/instruction rewriting. Use a strong model (e.g., `gpt-5`) |
| `reflection_minibatch_size` | `3` | Number of feedback examples per reflection batch. Higher = richer context for rewriting |
| `candidate_selection_strategy` | `"pareto"` | `"pareto"` explores recall-precision tradeoff; `"current_best"` optimizes single metric |
| `num_threads` | `None` | Parallel evaluation threads |
| `track_stats` | `False` | Enable `detailed_results` on compiled model |
| `track_best_outputs` | `False` | Track best outputs per example (requires `track_stats=True`) |
| `use_merge` | `True` | Enable merge-based optimization (combines best candidates) |
| `max_merge_invocations` | `5` | Cap on prompt merging operations |
| `skip_perfect_score` | `True` | Skip reflection on examples that already score perfectly |
| `component_selector` | `"round_robin"` | How GEPA picks which predictor to optimize next |
| `seed` | `0` | Random seed for reproducibility |
| `log_dir` | `None` | Directory for optimization logs |

### GEPA Budget Guidance

- `auto="light"`: Quick exploration. Good for prototyping and sanity checks.
- `auto="medium"`: Recommended default. Balances exploration depth with cost.
- `auto="heavy"`: Thorough optimization. Use for final production training.
- `max_full_evals=N`: Manual control. Each eval passes over `len(trainset) + len(valset)` examples.

### Accessing GEPA Results

```python
# After compilation with track_stats=True:
compiled_model = optimizer.compile(predict, trainset=data_train, valset=data_dev)

# Pareto frontier (when candidate_selection_strategy="pareto")
pareto_scores = compiled_model.detailed_results.val_aggregate_scores

# Save optimized model
compiled_model.save("pretrained/gepa_model.json")
```

### Why GEPA is the Default

1. **Feedback-driven**: Unlike other optimizers that only see pass/fail scores, GEPA reads textual feedback explaining *why* predictions failed, enabling targeted instruction improvement.
2. **Pareto exploration**: With `candidate_selection_strategy="pareto"`, GEPA explores the full recall-precision tradeoff rather than collapsing to a single objective.
3. **Reflection LM**: Uses a separate strong model to analyze failures and rewrite instructions, decoupling "thinking about the task" from "doing the task".
4. **Proven results**: In EiR screener experiments, GEPA with recall-priority feedback metrics produced the best recall while maintaining acceptable precision.

## Step 5: Benchmark Other Optimizers (Optional)

After training with GEPA, optionally benchmark other optimizers to verify GEPA is the best choice. Only prefer an alternative if it demonstrably outperforms GEPA on your holdout set.

```python
from dspy.teleprompt import (
    LabeledFewShot, BootstrapFewShot, BootstrapFewShotWithRandomSearch,
    KNNFewShot, MIPROv2, COPRO,
)

def test_optimizer(name, compile_func, eval_examples):
    compiled = compile_func()
    preds, golds = [], []
    for ex in eval_examples:
        pred = compiled(**ex.inputs())
        preds.append(pred.status)
        golds.append(ex.status)
    return evaluate_optimizer(name, preds, golds)
```

### Optimizer configurations

See [references/optimizers.md](references/optimizers.md) for the full optimizer catalog with recommended hyperparameters, selection criteria, and grid search configurations.

Summary of available optimizers (default first, then cheapest to most expensive):

| Optimizer | Cost | Best For | Key Params |
|-----------|------|----------|------------|
| **GEPA (default)** | **High** | **Feedback-driven prompt optimization, recall-precision tradeoff** | **`auto`, `reflection_lm`, `reflection_minibatch_size`, `candidate_selection_strategy`** |
| Baseline | Free | Establishing floor | None |
| LabeledFewShot | Low | Quick demo injection | `k` |
| BootstrapFewShot | Medium | General best single model | `max_labeled_demos`, `max_bootstrapped_demos` |
| BSFS + RandomSearch | Medium-High | Robust demo selection | `num_candidate_programs` |
| KNNFewShot | Medium | Semantic demo retrieval | `k`, `vectorizer` |
| MIPROv2 | High | Instruction optimization | `metric` |
| COPRO | High | Recall/precision tuning | `depth`, `breadth`, `init_temperature`, `metric` |

## Step 5a: Hyperparameter Sweep

After identifying top 2-3 optimizers from Steps 4-5, run a grid search on the dev set.

### GEPA Sweep

GEPA's main tuning levers are budget (`auto` or `max_full_evals`), `reflection_minibatch_size`, and `candidate_selection_strategy`:

```python
import json

def gepa_sweep():
    results = []
    reflection_lm = dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000)

    for auto, minibatch, strategy in [
        ("light", 3, "pareto"),
        ("medium", 3, "pareto"),
        ("medium", 8, "pareto"),
        ("medium", 8, "current_best"),
        ("heavy", 8, "pareto"),
    ]:
        def _compile(a=auto, m=minibatch, s=strategy):
            opt = GEPA(
                metric=recall_metric_with_feedback,
                auto=a,
                num_threads=32,
                track_stats=True,
                reflection_minibatch_size=m,
                reflection_lm=reflection_lm,
                candidate_selection_strategy=s,
            )
            return opt.compile(predict, trainset=data_train, valset=data_dev)
        metrics = test_optimizer(f"GEPA auto={auto} mb={minibatch} {strategy}", _compile, data_dev)
        if metrics:
            results.append(("GEPA", {"auto": auto, "minibatch": minibatch, "strategy": strategy}, metrics))

    results.sort(key=lambda x: (-x[2]["recall"], -x[2]["precision"]))
    with open("gepa_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
```

### Non-GEPA Sweep

```python
import itertools, json

def run_sweep():
    results = []
    # COPRO grid
    for depth, breadth, temp in itertools.product([1,2,3], [2,3,4], [0.0, 0.3]):
        def _compile(d=depth, b=breadth, t=temp):
            return COPRO(metric=recall_metric, depth=d, breadth=b,
                         init_temperature=t).compile(predict, trainset=data_train)
        metrics = test_optimizer(f"COPRO d{depth} b{breadth} t{temp}", _compile, data_dev)
        if metrics:
            results.append(("COPRO", {"depth": depth, "breadth": breadth, "temp": temp}, metrics))

    # BootstrapFewShot grid
    for labeled, boot in itertools.product([16, 32, 48], [4, 8]):
        def _compile(l=labeled, b=boot):
            return BootstrapFewShot(metric=validate_answer, max_labeled_demos=l,
                                    max_bootstrapped_demos=b, metric_threshold=1
                                    ).compile(predict, trainset=data_train)
        metrics = test_optimizer(f"BSFS L{labeled} B{boot}", _compile, data_dev)
        if metrics:
            results.append(("BSFS", {"labeled": labeled, "boot": boot}, metrics))

    results.sort(key=lambda x: (-x[2]["recall"], -x[2]["precision"]))
    with open("sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
```

Important: use default parameter capture (`d=depth`) in lambda/closures to avoid late-binding bugs.

## Step 6: Evaluate on Holdout

Run the best config(s) on the holdout set **once** for final numbers. Never tune on holdout.

```python
# Compile best model with full train+dev data if desired
extended_train = data_train + data_dev
best_model = BootstrapFewShot(
    metric=validate_answer, max_labeled_demos=16,
    max_bootstrapped_demos=4, metric_threshold=1
).compile(predict, trainset=extended_train)

preds, golds = [], []
for ex in data_holdout:
    pred = best_model(**ex.inputs())
    preds.append(pred.status)
    golds.append(ex.status)
evaluate_optimizer("Holdout Eval", preds, golds)
```

## Step 7: Ensemble / Union Strategy

Combine complementary models (e.g., one accuracy-focused + one recall-focused) with confidence thresholds.

```python
# Precompute predictions from both models
bsfs_preds = {ex: bsfs_model(**ex.inputs()) for ex in data_holdout}
copro_preds = {ex: copro_model(**ex.inputs()) for ex in data_holdout}

BSFS_MIN_CONF = 0.70
COPRO_MIN_CONF = 0.55

union_preds, golds = [], []
for ex in data_holdout:
    status = "REJECTED"
    bpr = bsfs_preds[ex]
    if bpr.status == "APPROVED" and bpr.confidence >= BSFS_MIN_CONF:
        status = "APPROVED"
    else:
        cpr = copro_preds[ex]
        if cpr.status == "APPROVED" and cpr.confidence >= COPRO_MIN_CONF:
            status = "APPROVED"
    union_preds.append(status)
    golds.append(ex.status)

evaluate_optimizer("UNION", union_preds, golds)
```

Union strategy logic:
- **Stage 1**: High-confidence model (BSFS) with strict threshold -> approve if confident
- **Stage 2**: High-recall model (COPRO) with looser threshold -> fallback approve
- **Default**: Reject if neither model approves

Tune thresholds on dev set, evaluate final numbers on holdout.

## Step 8: Error Analysis

Inspect false negatives and false positives to find patterns for prompt improvement.

```python
# False negatives (missed positives)
for ex in data_holdout:
    pred = model(**ex.inputs())
    if ex.status == "APPROVED" and pred.status == "REJECTED":
        print(f"MISSED: {ex.field_1}")
        print(f"  Confidence: {pred.confidence:.2f}")
        print(f"  Reasoning: {pred.reasoning}\n")

# False positives (incorrect approvals)
for ex in data_holdout:
    pred = model(**ex.inputs())
    if ex.status == "REJECTED" and pred.status == "APPROVED":
        print(f"FALSE APPROVE: {ex.field_1}")
        print(f"  Confidence: {pred.confidence:.2f}")
        print(f"  Reasoning: {pred.reasoning}\n")
```

Use error analysis to:
- Identify patterns in misclassified examples
- Refine signature docstring rules and examples
- Add new rejection/approval rules
- Update the dynamic conflict list
- Then **recompile all models** (prompt text changed)

## Step 9: Compile & Deploy to Production

Save optimized models as JSON artifacts for inference-only loading.

```python
# build_pretrained_models.py
from pathlib import Path
from dspy.teleprompt import GEPA, BootstrapFewShot, COPRO

MODEL_DIR = Path("terrain/ml/my_classifier/pretrained_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# GEPA (default production model)
reflection_lm = dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000)
gepa_prog = GEPA(
    metric=recall_metric_with_feedback,
    auto="heavy",  # Use heavy budget for production training
    num_threads=32,
    track_stats=True,
    track_best_outputs=True,
    reflection_minibatch_size=8,
    reflection_lm=reflection_lm,
    candidate_selection_strategy="pareto",
).compile(predict, trainset=data_train, valset=data_dev)
gepa_prog.save(MODEL_DIR / "gepa_screener.json")

# Alternative models (only if they outperformed GEPA in benchmarks)
bsfs_prog = BootstrapFewShot(
    metric=validate_answer, max_labeled_demos=16,
    max_bootstrapped_demos=4, metric_threshold=1,
).compile(predict, trainset=data_train)
bsfs_prog.save(MODEL_DIR / "bsfs16_4.json")

copro_prog = COPRO(
    metric=recall_metric, depth=4, breadth=6, init_temperature=0.0,
).compile(predict, trainset=data_train + data_dev, eval_kwargs={})
copro_prog.save(MODEL_DIR / "copro_d4_b6.json")
```

### Production inference wrapper

```python
class OptimizedClassifier:
    def __init__(self, model_name="gepa_screener"):
        model_dir = Path(__file__).resolve().parent / "pretrained_models"
        prog = dspy.Predict(MyClassifier).reset_copy()
        prog.load(model_dir / f"{model_name}.json")
        self.model = prog

    def predict(self, field_1: str, field_2: str) -> dict:
        pred = self.model(field_1=field_1, field_2=field_2)
        return {"status": pred.status, "confidence": pred.confidence, "reasoning": pred.reasoning}
```

Key deployment rules:
- **Default to GEPA**: Unless benchmarks show another optimizer outperforms it, ship the GEPA model
- **One-way dependency**: `experiments/ -> terrain/` (OK), `terrain/ -> experiments/` (NOT OK)
- Production code loads JSON only, never imports from experiments
- LM is configured at inference time (can differ from training LM)
- After any prompt change, **all models must be recompiled**

## Environment Flags

| Variable | Effect |
|----------|--------|
| `QUICK_SANITY=1` | Limit dev evaluation to 10 rows for fast smoke-testing |
| `SKIP_OPTIMIZERS=1` | Skip heavy optimizer sweeps when module is imported (not run directly) |

## Project Layout Convention

```
experiments/
  my_classifier/
    data/
      Examples.csv                    # Labeled dataset
    gepa_screener_experiments.py      # GEPA-focused experiment (default)
    all_optimizers_experiments.py     # Benchmark all optimizers (comparison)
    optimizer_sweep.py                # Hyperparameter grid search
    holdout_union_eval.py             # Union strategy evaluation
    build_pretrained_models.py        # Compile -> production JSON
    data_loader.py                    # Shared data loading utility
    pretrained/
      gepa_screener.json              # GEPA pretrained model (default)
    analysis/
      gepa_val_diagnostics.jsonl      # Per-candidate GEPA results
      gepa_val_summary.json           # Aggregate GEPA metrics
      false_negative_analysis.py
      false_positive_analysis.py

terrain/ml/
  my_classifier/
    core.py                           # Production Signature + OptimizedClassifier
    pretrained_models/
      gepa_screener.json              # Default production model
      bsfs16_4.json                   # Alternative (if outperforms GEPA)
      copro_d4_b6.json                # Alternative (if outperforms GEPA)
```
