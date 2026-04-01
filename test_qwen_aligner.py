#!/usr/bin/env python
"""Quick test of the updated Qwen3ForcedAligner integration."""

import torch
from src.alignment.forced_aligner import QwenForcedAligner, PhonemeAlignment

def test_aligner_import():
    """Test that we can import and initialize the aligner."""
    print("Testing Qwen3ForcedAligner import and initialization...")

    try:
        # Test English aligner
        aligner = QwenForcedAligner(device="cuda", language="en")
        print("✅ English aligner initialized successfully")

        # Check phoneme set
        phonemes = aligner.get_phoneme_inventory()
        print(f"✅ English phoneme inventory: {len(phonemes)} phonemes")

        # Test Mandarin aligner
        aligner_zh = QwenForcedAligner(device="cuda", language="zh")
        print("✅ Mandarin aligner initialized successfully")

        phonemes_zh = aligner_zh.get_phoneme_inventory()
        print(f"✅ Mandarin phoneme inventory: {len(phonemes_zh)} phonemes")

        # Test Cantonese aligner
        aligner_yue = QwenForcedAligner(device="cuda", language="yue")
        print("✅ Cantonese aligner initialized successfully")

        phonemes_yue = aligner_yue.get_phoneme_inventory()
        print(f"✅ Cantonese phoneme inventory: {len(phonemes_yue)} phonemes")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure qwen-asr is installed: pip install qwen-asr")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_aligner_import()
    exit(0 if success else 1)
