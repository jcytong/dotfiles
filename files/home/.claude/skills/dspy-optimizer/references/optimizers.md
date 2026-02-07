# DSPy Teleprompt Optimizer Reference

## Optimizer Catalog

### 1. GEPA (Default)

GEPA (Generative Prompt Adaptation) is the **default optimizer** for all new pipelines. It uses a reflection LM to analyze prediction failures via textual feedback from your metric function, then algorithmically rewrites instructions to fix those failures.

```python
from dspy.teleprompt import GEPA

reflection_lm = dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000)

optimizer = GEPA(
    metric=gepa_metric,                    # Must return dspy.Prediction(score=, feedback=)
    auto="medium",                          # Budget: "light", "medium", "heavy"
    num_threads=32,
    track_stats=True,
    track_best_outputs=True,
    reflection_minibatch_size=8,
    reflection_lm=reflection_lm,
    candidate_selection_strategy="pareto",
)
compiled = optimizer.compile(predict, trainset=data_train, valset=data_dev)
```

**Key parameters:**
- `metric`: Must accept 5 args `(gold, pred, trace, pred_name, pred_trace)` and return `dspy.Prediction(score=float, feedback=str)`
- `auto`: Budget preset (`"light"` / `"medium"` / `"heavy"`). Mutually exclusive with `max_full_evals` / `max_metric_calls`
- `max_full_evals`: Manual budget (e.g., `5` = 5 full passes over train+val). Alternative to `auto`
- `reflection_lm`: Separate strong LM for reflection. Required (or provide `instruction_proposer`)
- `reflection_minibatch_size`: Number of feedback examples per reflection batch (default: `3`, recommended: `3-8`)
- `candidate_selection_strategy`: `"pareto"` (explores multi-objective tradeoff) or `"current_best"` (optimizes single metric)
- `use_merge`: Enable merge-based optimization combining best candidates (default: `True`)
- `max_merge_invocations`: Cap on merge operations (default: `5`)
- `skip_perfect_score`: Skip reflection on already-perfect examples (default: `True`)
- `component_selector`: How GEPA picks which predictor to optimize (`"round_robin"` default, also `"all"` or a callable)
- `log_dir`: Directory for checkpointing optimization state (enables resume)
- `use_wandb`: Enable Weights & Biases experiment tracking (default: `False`)
- `use_mlflow`: Enable MLflow experiment tracking (default: `False`)

**Metric function signature:**
```python
def my_gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score = ...  # float
    feedback = ...  # str explaining WHY this score
    return dspy.Prediction(score=score, feedback=feedback)
```

**When to use:** Default choice for any pipeline. Especially strong when you can write rich textual feedback explaining prediction failures. Works with classification, QA, extraction, summarization, or any task with a scorable metric.

### 2. Baseline Predictor

Raw `dspy.Predict(Signature)` with no optimization. Run this first to establish a performance floor.

```python
predict = dspy.Predict(MyPipeline)
# No compilation needed
```

### 3. LabeledFewShot (LFS)

Injects `k` labeled examples as few-shot demos. No bootstrapping or instruction optimization.

```python
from dspy.teleprompt import LabeledFewShot
optimizer = LabeledFewShot(k=16)
compiled = optimizer.compile(predict, trainset=data_train)
```

**Hyperparameters:** `k` (grid: `[16, 24, 32, 40]`)

**When to use:** Quick baseline improvement. Low cost.

### 4. BootstrapFewShot (BSFS)

Bootstraps demos by running the model on training data and keeping examples where the metric passes.

```python
from dspy.teleprompt import BootstrapFewShot
optimizer = BootstrapFewShot(
    metric=accuracy_metric,
    max_labeled_demos=16,
    max_bootstrapped_demos=4,
    metric_threshold=1,
)
compiled = optimizer.compile(predict, trainset=data_train)
```

**Hyperparameters:** `max_labeled_demos` (grid: `[16, 32, 48]`), `max_bootstrapped_demos` (grid: `[4, 8]`)

**When to use:** Best general-purpose single non-GEPA model. Medium cost.

### 5. BootstrapFewShotWithRandomSearch (BSFSWRS)

Runs BSFS multiple times with random subsets and picks the best candidate program.

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
optimizer = BootstrapFewShotWithRandomSearch(
    metric=accuracy_metric,
    num_candidate_programs=16,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
)
compiled = optimizer.compile(predict, trainset=data_train)
```

**When to use:** When BSFS varies across runs. Medium-high cost.

### 6. KNNFewShot

Selects demos via k-nearest-neighbor semantic similarity at inference time.

```python
from dspy.teleprompt import KNNFewShot
optimizer = KNNFewShot(k=5, trainset=data_train, vectorizer=openai_embeddings)
compiled = optimizer.compile(predict)
```

**When to use:** When input varies widely and relevant demos depend on similarity. Requires embeddings at inference.

### 7. MIPROv2

Task-aware instruction and demonstration optimization.

```python
from dspy.teleprompt import MIPROv2
optimizer = MIPROv2(metric=simple_metric)
compiled = optimizer.compile(predict, trainset=data_train, valset=data_dev)
```

**When to use:** When prompt wording matters. High cost. Requires trainset + valset.

### 8. COPRO (Cost-Aware Prompt Optimizer)

Iteratively optimizes prompts by breadth-first search over instruction candidates.

```python
from dspy.teleprompt import COPRO
optimizer = COPRO(
    metric=simple_metric,
    depth=4,
    breadth=6,
    init_temperature=0.0,
)
compiled = optimizer.compile(predict, trainset=data_train, eval_kwargs={})
```

**Hyperparameters:** `depth` (grid: `[1,2,3,4]`), `breadth` (grid: `[2,3,4,6]`), `init_temperature` (grid: `[0.0, 0.3, 1.4]`)

**When to use:** Targeted metric optimization. The `eval_kwargs={}` parameter is required.

### 9. SIMBA (Stochastic Mini-Batch Adaptation)

Uses stochastic mini-batch sampling to identify challenging examples and focus optimization effort on them.

```python
from dspy.teleprompt import SIMBA
optimizer = SIMBA(metric=simple_metric, max_steps=12, max_demos=10)
compiled = optimizer.compile(student=predict, trainset=data_train)
```

**Key parameters:** `max_steps` (optimization iterations), `max_demos` (max demonstrations to include)

**When to use:** When training data has uneven difficulty and you want the optimizer to focus on hard examples. Medium-high cost.

## Hyperparameter Sweep Grids

```python
# GEPA: 5 configs (default - sweep first)
gepa_grid = [
    # (auto, reflection_minibatch_size, candidate_selection_strategy)
    ("light", 3, "pareto"),
    ("medium", 3, "pareto"),
    ("medium", 8, "pareto"),
    ("medium", 8, "current_best"),
    ("heavy", 8, "pareto"),
]

# COPRO: 18 configs
copro_grid = itertools.product(
    [1, 2, 3],       # depth
    [2, 3, 4],       # breadth
    [0.0, 0.3],      # init_temperature
)

# BootstrapFewShot: 6 configs
bsfs_grid = itertools.product(
    [16, 32, 48],     # max_labeled_demos
    [4, 8],           # max_bootstrapped_demos
)

# LabeledFewShot: 4 configs
lfs_grid = [16, 24, 32, 40]  # k values
```

## Optimizer Selection Guide

```
Default choice for any new pipeline?
  -> GEPA with feedback metric (auto="medium", candidate_selection_strategy="pareto")

Need quick baseline / sanity check?
  -> LabeledFewShot (k=16)

Need best single non-GEPA model?
  -> BootstrapFewShot (16 labeled, 4 boot)

Want to optimize for your task metric?
  -> GEPA with a custom feedback metric (start here)
  -> Fallback: COPRO with your metric function

Need best overall performance?
  -> GEPA (start here, benchmark others against it)
  -> If others outperform: Ensemble strategy combining complementary models

Input varies widely, need adaptive demos?
  -> KNNFewShot

Want DSPy to rewrite your instructions?
  -> GEPA (does this natively via reflection LM)
  -> Fallback: MIPROv2

Training data has uneven difficulty?
  -> SIMBA (focuses on challenging examples via mini-batch sampling)
```

**Decision rule**: Always start with GEPA. Only use alternatives if benchmarking shows they outperform GEPA on your specific holdout set.

## Closure Gotcha

When defining compile functions in a loop, use default parameter capture to avoid late-binding:

```python
# WRONG - all closures share the same variable
for depth in [1, 2, 3]:
    def _compile():
        return COPRO(depth=depth, ...)  # Always uses depth=3

# CORRECT - capture current value
for depth in [1, 2, 3]:
    def _compile(d=depth):
        return COPRO(depth=d, ...)  # Uses 1, 2, 3 respectively
```
