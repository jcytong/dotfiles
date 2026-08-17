#!/usr/bin/env python3
"""
Production inference wrapper for pretrained DSPy models.

Loads a pretrained JSON model and runs predictions.
Can be used as a library or standalone CLI.

Usage as CLI:
    python inference.py --model pretrained/gepa_model.json --input "some text"

Usage as library:
    from inference import Predictor
    model = Predictor("pretrained/gepa_model.json")
    result = model.predict(field_1="some text")

    # Backward-compatible alias:
    from inference import Classifier
    clf = Classifier("pretrained/gepa_model.json")
    result = clf.predict(field_1="some text")
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import dspy
from dotenv import load_dotenv

load_dotenv()

# ── CONFIGURE THESE ──────────────────────────────────────────────────────────

INFERENCE_MODEL = "openai/gpt-4o"
INFERENCE_TEMPERATURE = 1.0
INFERENCE_MAX_TOKENS = 16000

# Output fields — which fields to extract from predictions.
# Classification: ["status", "confidence", "reasoning"]
# QA: ["answer", "reasoning"]
# Extraction: ["entities", "reasoning"]
OUTPUT_FIELDS = ["status", "confidence", "reasoning"]

# ── END CONFIGURATION ────────────────────────────────────────────────────────


class Predictor:
    """
    Production predictor that loads a pretrained DSPy model.

    Usage:
        model = Predictor("pretrained/gepa_model.json")
        result = model.predict(field_1="input text")
        print(result)  # {"status": "APPROVED", "confidence": 0.92, "reasoning": "..."}
    """

    def __init__(
        self,
        model_path: str | Path,
        model: str = INFERENCE_MODEL,
        temperature: float = INFERENCE_TEMPERATURE,
        max_tokens: int = INFERENCE_MAX_TOKENS,
        output_fields: List[str] = None,
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._output_fields = output_fields or OUTPUT_FIELDS

        dspy.configure(
            lm=dspy.LM(model=model, temperature=temperature, max_tokens=max_tokens)
        )

        # Replace dspy.Signature with your actual Signature class
        self._prog = dspy.Predict(dspy.Signature).reset_copy()
        self._prog.load(str(model_path))

    def predict(self, **inputs) -> Dict[str, Any]:
        """
        Run prediction with the loaded model.

        Args:
            **inputs: Keyword arguments matching your Signature's InputFields.

        Returns:
            Dict with the configured OUTPUT_FIELDS.
        """
        pred = self._prog(**inputs)
        return {field: getattr(pred, field, None) for field in self._output_fields}

    def predict_batch(self, examples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Run predictions on a list of input dicts."""
        return [self.predict(**ex) for ex in examples]


# Backward-compatible alias
Classifier = Predictor


def main():
    parser = argparse.ArgumentParser(description="Run inference with a pretrained DSPy model")
    parser.add_argument("--model", required=True, help="Path to pretrained model JSON")
    parser.add_argument("--input", help="Single input value (for single-field signatures)")
    parser.add_argument("--input-json", help="JSON object with input fields")
    parser.add_argument("--input-field", default="field_1", help="Field name for --input (default: field_1)")
    args = parser.parse_args()

    predictor = Predictor(args.model)

    if args.input_json:
        inputs = json.loads(args.input_json)
    elif args.input:
        inputs = {args.input_field: args.input}
    else:
        parser.error("Provide --input or --input-json")
        return

    result = predictor.predict(**inputs)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
