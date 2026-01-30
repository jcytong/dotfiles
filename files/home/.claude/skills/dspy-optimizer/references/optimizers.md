# DSPy Teleprompt Optimizer Reference

## Optimizer Catalog

### 1. GEPA (Default)

GEPA (Generative Prompt Adaptation) is the **default optimizer** for all new pipelines. It uses a reflection LM to analyze prediction failures via textual feedback from your metric function, then algorithmically rewrites instructions to fix those failures.

```python
from dspy.teleprompt import GEPA

reflection_lm = dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000)

optimizer = GEPA(
    metric=recall_metric_with_feedback,   # Must return dspy.Prediction(score=, feedback=)
    auto="medium",                         # Budget: "light", "medium", "heavy"
    num_threads=32,                        # Parallel evaluation threads
    track_stats=True,                      # Enable detailed result tracking
    track_best_outputs=True,               # Track best outputs per example
    reflection_minibatch_size=8,           # Batch size for feedback reflection
    reflection_lm=reflection_lm,           # Stronger model for prompt rewriting
    candidate_selection_strategy="pareto", # Explore recall-precision tradeoff
)
compiled = optimizer.compile(predict, trainset=data_train, valset=data_dev)
```

**Key parameters:**
- `metric`: Must accept 5 args `(gold, pred, trace, pred_name, pred_trace)` and return `dspy.Prediction(score=float, feedback=str)`
- `auto`: Budget preset (`"light"` / `"medium"` / `"heavy"`). Mutually exclusive with `max_full_evals` / `max_metric_calls`
- `max_full_evals`: Manual budget (e.g., `5` = 5 full passes over train+val). Alternative to `auto`
- `reflection_lm`: Separate strong LM for reflection. Required (or provide `instruction_proposer`)
- `reflection_minibatch_size`: Number of feedback examples per reflection batch (default: `3`, recommended: `3-8`)
- `candidate_selection_strategy`: `"pareto"` (explores recall-precision tradeoff) or `"current_best"` (single metric)
- `use_merge`: Enable merge-based optimization combining best candidates (default: `True`)
- `max_merge_invocations`: Cap on merge operations (default: `5`)
- `skip_perfect_score`: Skip reflection on already-perfect examples (default: `True`)
- `component_selector`: How GEPA picks which predictor to optimize (`"round_robin"` default)

**Metric function requirements:**
```python
def my_gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Must accept 5 args. Return dspy.Prediction(score=, feedback=)."""
    score = ...  # float
    feedback = ...  # str explaining WHY this score
    return dspy.Prediction(score=score, feedback=feedback)
```

**When to use:** Default choice for all classification tasks. Especially strong when you can write rich textual feedback explaining prediction failures. Use `candidate_selection_strategy="pareto"` when optimizing for recall-precision tradeoff.

**GEPA sweep grid:**
```python
gepa_configs = [
    ("light", 3, "pareto"),
    ("medium", 3, "pareto"),
    ("medium", 8, "pareto"),
    ("medium", 8, "current_best"),
    ("heavy", 8, "pareto"),
]
```

### 2. Baseline Predictor

Raw `dspy.Predict(Signature)` with no optimization. Always run this first to establish a performance floor.

```python
predict = dspy.Predict(MyClassifier)
# No compilation needed
```

### 3. LabeledFewShot (LFS)

Injects `k` labeled examples as few-shot demos. No bootstrapping or instruction optimization.

```python
from dspy.teleprompt import LabeledFewShot

optimizer = LabeledFewShot(k=16)
compiled = optimizer.compile(predict, trainset=data_train)
```

**Hyperparameters:**
- `k`: Number of labeled demos to include (default grid: `[16, 24, 32, 40]`)

**When to use:** Quick baseline improvement over raw prediction. Low cost.

### 4. BootstrapFewShot (BSFS)

Bootstraps additional demonstrations by running the model on training data and keeping examples where the metric passes.

```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=validate_answer,
    max_labeled_demos=16,
    max_bootstrapped_demos=4,
    metric_threshold=1,
)
compiled = optimizer.compile(predict, trainset=data_train)
```

**Hyperparameters:**
- `max_labeled_demos`: Number of gold-label demos (grid: `[16, 32, 48]`)
- `max_bootstrapped_demos`: Number of self-generated demos (grid: `[4, 8]`)
- `metric_threshold`: Minimum metric score for bootstrapped demo acceptance (typically `1` for exact match)
- `metric`: Per-example metric function

**When to use:** Best general-purpose single model. Medium cost. Usually the strongest single performer.

### 5. BootstrapFewShotWithRandomSearch (BSFSWRS)

Runs BSFS multiple times with random subsets and picks the best candidate program.

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

optimizer = BootstrapFewShotWithRandomSearch(
    metric=validate_answer,
    num_candidate_programs=16,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
)
compiled = optimizer.compile(predict, trainset=data_train)
```

**Hyperparameters:**
- `num_candidate_programs`: Number of random programs to evaluate (default: `16`)
- Same as BSFS for demo counts

**When to use:** When BSFS performance varies across runs and you want more robustness. Higher cost than BSFS.

### 6. KNNFewShot

Selects demos via k-nearest-neighbor semantic similarity at inference time.

```python
from dspy.teleprompt import KNNFewShot
from openai import OpenAI
import numpy as np

client = OpenAI()

def openai_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    embeddings = np.array([e.embedding for e in response.data], dtype=np.float32)
    return embeddings[0] if len(embeddings) == 1 else embeddings

optimizer = KNNFewShot(k=5, trainset=data_train, vectorizer=openai_embeddings)
compiled = optimizer.compile(predict)
```

**Hyperparameters:**
- `k`: Number of nearest neighbors to retrieve (default: `5`)
- `vectorizer`: Embedding function (must handle single string and list of strings)

**When to use:** When input examples vary widely and relevant demos depend on similarity. Requires embedding API calls at inference time.

### 7. MIPROv2

Task-aware instruction and demonstration optimization. Optimizes both the prompt instructions and demo selection.

```python
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(metric=validate_answer)
compiled = optimizer.compile(predict, trainset=data_train, valset=data_dev)
```

**Hyperparameters:**
- `metric`: Per-example metric function

**When to use:** When prompt wording matters and you want DSPy to rewrite instructions. High cost. Requires both trainset and valset.

### 8. COPRO (Cost-Aware Prompt Optimizer)

Iteratively optimizes prompts by breadth-first search over instruction candidates, scored by a custom metric.

```python
from dspy.teleprompt import COPRO

optimizer = COPRO(
    metric=recall_metric,
    depth=4,
    breadth=6,
    init_temperature=0.0,
)
compiled = optimizer.compile(predict, trainset=data_train, eval_kwargs={})
```

**Hyperparameters:**
- `depth`: Search depth for instruction refinement (grid: `[1, 2, 3, 4]`)
- `breadth`: Number of candidate instructions per depth level (grid: `[2, 3, 4, 6]`)
- `init_temperature`: Starting temperature for generation (grid: `[0.0, 0.3, 1.4]`)
- `metric`: Per-example metric (use `recall_metric` for recall-focused, `precision_metric` for precision-focused)
- `prompt_model`: Optional separate LM for generating prompt candidates (e.g., `dspy.LM(model="openai/gpt-4.1")`)

**When to use:** When you need to optimize for a specific metric (recall vs precision). Best paired with BSFS in a union strategy. High cost. The `eval_kwargs={}` parameter is required.

## Hyperparameter Sweep Grid

Recommended sweep configurations:

```python
# GEPA: 5 configs (default optimizer - sweep first)
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

Need high recall (don't miss positives)?
  -> GEPA with recall-priority feedback metric (default)
  -> Fallback: COPRO with recall_metric

Need high precision (minimize false positives)?
  -> GEPA with precision-priority feedback metric
  -> Fallback: COPRO with precision_metric

Need best overall performance?
  -> GEPA (start here, benchmark others against it)
  -> If others outperform: Union strategy: BSFS (high conf) + COPRO (lower conf)

Input varies widely, need adaptive demos?
  -> KNNFewShot

Want DSPy to rewrite your instructions?
  -> GEPA (does this natively via reflection LM)
  -> Fallback: MIPROv2
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
