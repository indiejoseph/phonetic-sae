#!/usr/bin/env python3
"""Inspect the actual Qwen3-ForcedAligner API and methods.

This script investigates what the real model actually provides.

Usage:
    python scripts/inspect_aligner_api.py
"""

import argparse
import inspect
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def inspect_model_api(language: str = "en", device: str = "cuda"):
    """Inspect what methods and attributes the model actually has."""

    print(f"\n{'='*70}")
    print(f"QWEN3-FORCEDALIGNER ACTUAL API INSPECTION")
    print(f"{'='*70}\n")

    try:
        from qwen_asr import Qwen3ForcedAligner
        import torch

        print(f"Loading model...")
        device_map = "cuda:0" if device == "cuda" else "cpu"
        model = Qwen3ForcedAligner.from_pretrained(
            "Qwen/Qwen3-ForcedAligner-0.6B",
            dtype=torch.bfloat16,
            device_map=device_map,
        )

        print(f"✅ Model loaded!\n")

        # Inspect model methods
        print(f"{'='*70}")
        print(f"MODEL METHODS & FUNCTIONS")
        print(f"{'='*70}\n")

        model_methods = {}
        for name in dir(model):
            if not name.startswith("_"):
                attr = getattr(model, name)
                if callable(attr):
                    # Try to get signature
                    try:
                        sig = inspect.signature(attr)
                        model_methods[name] = str(sig)
                    except:
                        model_methods[name] = "callable (signature unknown)"

        # Print key methods
        key_methods = [
            "align", "forward", "generate", "encode", "decode",
            "process", "predict", "infer", "get_phonemes",
            "tokenize", "encode_phonemes", "extract_features",
        ]

        print("Looking for key methods:")
        for method in key_methods:
            if method in model_methods:
                print(f"  ✅ {method}: {model_methods[method]}")

        print("\nAll callable methods:")
        for method, sig in sorted(model_methods.items())[:30]:
            print(f"  • {method}: {sig}")

        # Note: qwen_asr doesn't use a separate processor like transformers

        # Inspect config
        print(f"\n{'='*70}")
        print(f"MODEL CONFIG")
        print(f"{'='*70}\n")

        if hasattr(model, "config"):
            config = model.config
            print("Config attributes:")
            for key in sorted(config.__dict__.keys())[:20]:
                val = getattr(config, key)
                if not callable(val):
                    print(f"  • {key}: {val}")

        # Try actual inference to see what the model expects
        print(f"\n{'='*70}")
        print(f"TESTING ACTUAL MODEL INPUT/OUTPUT")
        print(f"{'='*70}\n")

        print("Testing inference...")
        try:
            # Try different input patterns
            test_inputs = [
                ("text + audio", {"text": "hello", "audio": torch.randn(1, 16000)}),
                ("text only", {"text": "hello"}),
                ("forward with text", "hello"),
            ]

            for description, test_input in test_inputs:
                try:
                    print(f"  Testing: {description}...")
                    if isinstance(test_input, dict):
                        output = model(**test_input)
                    else:
                        output = model(test_input)

                    print(f"    ✅ Success!")
                    print(f"    Output type: {type(output).__name__}")
                    if hasattr(output, "keys"):
                        print(f"    Output keys: {list(output.keys())}")
                    elif isinstance(output, (list, tuple)):
                        print(f"    Output length: {len(output)}")
                        for i, item in enumerate(output[:3]):
                            print(f"      [{i}]: {type(item).__name__}")

                except Exception as e:
                    print(f"    ✗ Failed: {str(e)[:100]}")

        except Exception as e:
            print(f"Could not test inference: {e}")

        # Save inspection results
        inspection_results = {
            "model_methods": list(model_methods.keys())[:50],
            "key_methods_found": [m for m in key_methods if m in model_methods],
            "config_keys": list(config.__dict__.keys()) if hasattr(model, "config") else [],
        }

        output_file = Path("aligner_api_inspection.json")
        with open(output_file, "w") as f:
            json.dump(inspection_results, f, indent=2)

        print(f"\n✅ Inspection saved to: {output_file}")

    except Exception as e:
        logger.error(f"Failed to inspect model: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Inspect Qwen3-ForcedAligner actual API")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use",
    )

    args = parser.parse_args()
    inspect_model_api(device=args.device)


if __name__ == "__main__":
    main()
