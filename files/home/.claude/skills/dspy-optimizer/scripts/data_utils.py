"""
Portable data loading and stratified splitting for DSPy classification pipelines.

Usage:
    from data_utils import load_from_csv, load_from_jsonl, stratified_split, print_split_summary
"""

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dspy


def load_from_csv(
    path: str | Path,
    input_fields: List[str],
    label_field: str,
    label_map: Optional[Dict[str, str]] = None,
    reasoning_field: Optional[str] = None,
) -> List[dspy.Example]:
    """
    Load labeled examples from a CSV file.

    Args:
        path: Path to CSV file.
        input_fields: Column names to use as dspy.InputField values.
        label_field: Column name containing the classification label.
        label_map: Optional dict mapping raw labels to canonical labels
                   (e.g., {"yes": "APPROVED", "no": "REJECTED"}).
        reasoning_field: Optional column with ground-truth reasoning.

    Returns:
        List of dspy.Example objects with inputs marked.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    examples: List[dspy.Example] = []
    skipped = 0

    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_label = row.get(label_field, "").strip()

            if label_map:
                label = label_map.get(raw_label.lower())
                if label is None:
                    skipped += 1
                    continue
            else:
                label = raw_label
                if not label:
                    skipped += 1
                    continue

            fields = {}
            for field in input_fields:
                value = row.get(field, "").strip()
                if not value:
                    skipped += 1
                    break
                fields[field] = value
            else:
                fields["status"] = label
                if reasoning_field and row.get(reasoning_field):
                    fields["reasoning"] = row[reasoning_field].strip()

                ex = dspy.Example(**fields).with_inputs(*input_fields)
                examples.append(ex)

    print(f"Loaded {len(examples)} examples from {path.name} (skipped {skipped})")
    _print_label_counts(examples)
    return examples


def load_from_jsonl(
    path: str | Path,
    input_fields: List[str],
    label_field: str = "status",
    label_map: Optional[Dict[str, str]] = None,
    reasoning_field: Optional[str] = None,
) -> List[dspy.Example]:
    """
    Load labeled examples from a JSONL file (one JSON object per line).

    Args:
        path: Path to JSONL file.
        input_fields: Keys to use as dspy.InputField values.
        label_field: Key containing the classification label.
        label_map: Optional mapping from raw labels to canonical labels.
        reasoning_field: Optional key with ground-truth reasoning.

    Returns:
        List of dspy.Example objects with inputs marked.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")

    examples: List[dspy.Example] = []
    skipped = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)

            raw_label = str(payload.get(label_field, "")).strip()
            if label_map:
                label = label_map.get(raw_label.lower())
                if label is None:
                    skipped += 1
                    continue
            else:
                label = raw_label
                if not label:
                    skipped += 1
                    continue

            fields = {}
            valid = True
            for field in input_fields:
                value = str(payload.get(field, "")).strip()
                if not value:
                    skipped += 1
                    valid = False
                    break
                fields[field] = value

            if not valid:
                continue

            fields["status"] = label
            if reasoning_field and payload.get(reasoning_field):
                fields["reasoning"] = str(payload[reasoning_field]).strip()

            ex = dspy.Example(**fields).with_inputs(*input_fields)
            examples.append(ex)

    print(f"Loaded {len(examples)} examples from {path.name} (skipped {skipped})")
    _print_label_counts(examples)
    return examples


def stratified_split(
    examples: List[dspy.Example],
    train_frac: float = 0.64,
    dev_frac: float = 0.16,
    seed: int = 42,
    label_attr: str = "status",
) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """
    Split examples into train/dev/holdout while preserving class ratios.

    Args:
        examples: Full dataset.
        train_frac: Fraction allocated to training.
        dev_frac: Fraction allocated to dev/validation.
        seed: Random seed for reproducibility.
        label_attr: Attribute name containing the label.

    Returns:
        (train, dev, holdout) tuple.
    """
    rng = random.Random(seed)
    buckets: Dict[str, List[dspy.Example]] = {}

    for ex in examples:
        label = getattr(ex, label_attr)
        buckets.setdefault(label, []).append(ex)

    for items in buckets.values():
        rng.shuffle(items)

    def _allocate(fraction: float) -> List[dspy.Example]:
        allocated: List[dspy.Example] = []
        for label, items in buckets.items():
            count = len(items)
            if count == 0:
                continue
            take = int(round(count * fraction))
            if take == 0 and fraction > 0 and count > 1:
                take = 1
            take = min(take, len(items))
            allocated.extend(items[:take])
            buckets[label] = items[take:]
        rng.shuffle(allocated)
        return allocated

    train = _allocate(train_frac)
    dev = _allocate(dev_frac)
    holdout: List[dspy.Example] = []
    for remaining in buckets.values():
        holdout.extend(remaining)
    rng.shuffle(holdout)

    return train, dev, holdout


def print_split_summary(
    splits: Dict[str, List[dspy.Example]],
    label_attr: str = "status",
) -> None:
    """Print a formatted summary of dataset splits with class counts."""
    print("\nDataset split summary (stratified):")
    for name, data in splits.items():
        by_label: Dict[str, int] = {}
        for ex in data:
            label = getattr(ex, label_attr)
            by_label[label] = by_label.get(label, 0) + 1
        parts = ", ".join(f"{k}: {v}" for k, v in sorted(by_label.items()))
        total = len(data)
        print(f"  {name:<12}: {total:>4} rows | {parts}")
    print()


def _print_label_counts(
    examples: List[dspy.Example],
    label_attr: str = "status",
) -> None:
    counts: Dict[str, int] = {}
    for ex in examples:
        label = getattr(ex, label_attr, "UNKNOWN")
        counts[label] = counts.get(label, 0) + 1
    for label, count in sorted(counts.items()):
        print(f"  - {label}: {count}")
