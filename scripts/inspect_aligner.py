#!/usr/bin/env python3
"""Inspect Qwen3-ForcedAligner to verify phoneme sets and model properties.

This script loads the forced aligner and prints:
- Available phoneme sets per language
- Model configuration
- Expected output format

Usage:
    python scripts/inspect_aligner.py
    python scripts/inspect_aligner.py --lang en
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from src.alignment import QwenForcedAligner, LanguageSpecificAligner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def inspect_single_language(language: str, device: str = "cuda"):
    """Inspect forced aligner for a single language."""
    print(f"\n{'=' * 70}")
    print(f"LANGUAGE: {language.upper()}")
    print(f"{'=' * 70}")

    try:
        aligner = QwenForcedAligner(device=device, language=language)

        # Get phoneme inventory
        phonemes = aligner.get_phoneme_inventory()
        print(f"\n✅ Loaded aligner for '{language}'")
        print(f"   Phonemes ({len(phonemes)}): {phonemes[:20]}...")  # Print first 20

        # Save to file
        output_file = Path(f"phoneme_inventory_{language}.json")
        with open(output_file, "w") as f:
            json.dump({
                "language": language,
                "num_phonemes": len(phonemes),
                "phonemes": phonemes,
            }, f, indent=2)
        print(f"   Saved to: {output_file}")

        # Try validation
        test_phonemes = phonemes[:3]
        valid = aligner.validate_phonemes(test_phonemes)
        print(f"   Sample validation: {test_phonemes} → Valid: {valid}")

        # Model info
        print(f"\nModel Info:")
        print(f"   Model type: {type(aligner.model).__name__}")
        print(f"   Device: {aligner.device}")

        # Config
        if hasattr(aligner.model, "config"):
            config = aligner.model.config
            print(f"   Config keys: {list(config.__dict__.keys())[:10]}...")

        return True

    except Exception as e:
        logger.error(f"Failed for language '{language}': {e}")
        return False


def inspect_all_languages(device: str = "cuda"):
    """Inspect forced aligner for all supported languages."""
    print(f"\n{'=' * 70}")
    print(f"QWEN3-FORCEDALIGNER INSPECTION TOOL")
    print(f"{'=' * 70}")

    languages = ["en", "zh", "yue"]
    results = {}

    for lang in languages:
        success = inspect_single_language(lang, device=device)
        results[lang] = "✅ Success" if success else "❌ Failed"

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for lang, status in results.items():
        print(f"  {lang}: {status}")

    # Save summary
    with open("aligner_inspection_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to: aligner_inspection_summary.json")

    # Print phoneme inventory files
    print(f"\n{'=' * 70}")
    print("PHONEME INVENTORY FILES")
    print(f"{'=' * 70}")
    for lang in languages:
        inv_file = Path(f"phoneme_inventory_{lang}.json")
        if inv_file.exists():
            print(f"  ✅ {inv_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Qwen3-ForcedAligner properties and phoneme sets"
    )
    parser.add_argument(
        "--lang",
        choices=["en", "zh", "yue", "all"],
        default="all",
        help="Language to inspect (default: all)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use",
    )

    args = parser.parse_args()

    if args.lang == "all":
        inspect_all_languages(device=args.device)
    else:
        inspect_single_language(args.lang, device=args.device)


if __name__ == "__main__":
    main()
