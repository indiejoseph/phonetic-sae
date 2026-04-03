#!/usr/bin/env python3
"""Prepare combined dataset: HF audio + Qwen3-TTS codec extraction.

Self-contained script that:
1. Loads indiejoseph/tts20250516 from HuggingFace
2. Extracts codec tokens from each sample's audio using Qwen3TTSTokenizer
3. Adds codec + lang columns
4. Saves with ds.save_to_disk() (preserves audio in Arrow format)

Usage:
    # GPU (recommended for codec extraction speed):
    PYTHONPATH="." python scripts/prepare_dataset.py \
        --output data/combined \
        --device cuda

    # CPU (slow but works):
    PYTHONPATH="." python scripts/prepare_dataset.py \
        --output data/combined \
        --device cpu \
        --max-samples 100

    # Filter by language:
    PYTHONPATH="." python scripts/prepare_dataset.py \
        --output data/combined_yue \
        --lang yue
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """Detect language from text content.

    Simple heuristic: if text contains CJK characters, check ratio.
    Falls back to 'en' for Latin-script text.
    """
    cjk_pattern = re.compile(
        r'[\u3000-\u303f\u3040-\u30ff\u3100-\u312f\u3400-\u4dbf'
        r'\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]'
    )
    cjk_chars = len(cjk_pattern.findall(text))
    total_chars = len(text.strip())
    if total_chars == 0:
        return "en"
    if cjk_chars / total_chars > 0.3:
        return "zh"  # Could be zh or yue — dataset's own lang field is more reliable
    return "en"


def main():
    parser = argparse.ArgumentParser(
        description="Prepare combined dataset with Qwen3-TTS codec extraction"
    )
    parser.add_argument("--hf-dataset", default="indiejoseph/tts20250516")
    parser.add_argument("--output", type=Path, default=Path("data/combined"))
    parser.add_argument("--lang", default=None, help="Filter: en, zh, yue")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for codec extraction",
    )
    parser.add_argument(
        "--tokenizer-id",
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        help="Qwen3-TTS tokenizer model ID",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for codec extraction (1 = sequential)",
    )
    args = parser.parse_args()

    # ── Step 1: Load HF dataset ──────────────────────────────────────
    from datasets import load_dataset

    logger.info(f"Loading dataset: {args.hf_dataset}")
    ds = load_dataset(args.hf_dataset, split="train")
    logger.info(f"Dataset loaded: {len(ds)} samples")

    # Filter by language if the dataset has a 'lang' column
    if args.lang and "lang" in ds.column_names:
        ds = ds.filter(lambda x: x.get("lang") == args.lang)
        logger.info(f"Filtered to lang={args.lang}: {len(ds)} samples")

    if args.max_samples and args.max_samples < len(ds):
        ds = ds.select(range(args.max_samples))
        logger.info(f"Truncated to {args.max_samples} samples")

    # ── Step 2: Load Qwen3-TTS tokenizer for codec extraction ────────
    from qwen_tts import Qwen3TTSTokenizer

    device_map = "cuda:0" if args.device == "cuda" else args.device
    logger.info(f"Loading Qwen3-TTS tokenizer: {args.tokenizer_id} on {device_map}")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_id,
        device_map=device_map,
    )
    logger.info("Tokenizer loaded")

    # ── Step 3: Extract codec tokens ─────────────────────────────────
    errors = 0

    def extract_codec(example):
        """Extract codec tokens from audio using Qwen3TTSTokenizer."""
        nonlocal errors
        try:
            audio = example["audio"]
            audio_array = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]

            enc = tokenizer.encode(audio_array, sr=sr)
            # enc.audio_codes is List[torch.LongTensor], each shape (T, 16)
            codes = enc.audio_codes[0].cpu().tolist()
            example["codec"] = codes

            # Add lang if not present
            if "lang" not in example or not example.get("lang"):
                example["lang"] = detect_language(example.get("text", ""))

            return example
        except Exception as e:
            errors += 1
            if errors <= 10:
                logger.warning(f"Codec extraction failed: {e}")
            elif errors == 11:
                logger.warning("Suppressing further codec extraction warnings...")
            example["codec"] = None
            return example

    logger.info("Extracting codec tokens from audio...")
    ds = ds.map(extract_codec)

    # Remove samples where codec extraction failed
    before = len(ds)
    ds = ds.filter(lambda x: x["codec"] is not None)
    after = len(ds)
    if before != after:
        logger.info(f"Removed {before - after} samples with failed codec extraction")
    logger.info(f"Final dataset: {after} samples, errors: {errors}")

    # ── Step 4: Save ─────────────────────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(args.output))
    logger.info(f"Saved to {args.output}")

    # Print summary
    logger.info("Column names: " + ", ".join(ds.column_names))
    if "lang" in ds.column_names:
        from collections import Counter

        lang_counts = Counter(ds["lang"])
        for lang, count in sorted(lang_counts.items()):
            logger.info(f"  {lang}: {count}")


if __name__ == "__main__":
    main()
