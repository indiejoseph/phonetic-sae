#!/usr/bin/env python3
"""Generate synthetic test datasets for Phase 1 validation.

Creates small JSONL datasets with synthetic audio for quick pipeline testing
without needing real speech samples.

Usage:
    python scripts/generate_test_dataset.py --output data/test_dataset --num-samples 10
    python scripts/generate_test_dataset.py --lang en zh yue --output data/multilingual_test
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Sample texts for each language
SAMPLE_TEXTS = {
    "en": [
        "hello world",
        "the quick brown fox",
        "phoneme alignment test",
        "machine learning is fun",
        "speech synthesis works",
        "neural networks are powerful",
        "activation capture",
        "sparse autoencoders",
        "mechanistic interpretability",
        "language models",
    ],
    "zh": [
        "你好世界",
        "机器学习很有趣",
        "语音合成",
        "神经网络",
        "文本转语音",
        "音素对齐",
        "稀疏自编码器",
        "机械可解释性",
        "深度学习",
        "自然语言处理",
    ],
    "yue": [
        "你好世界",
        "機器學習",
        "語音合成",
        "神經網絡",
        "文本轉語音",
        "音素對齐",
        "稀疏自編碼器",
        "機械可解釋性",
        "深度學習",
        "自然語言處理",
    ],
}

LANGUAGE_CODES = {
    "en": "en",
    "english": "en",
    "zh": "zh",
    "mandarin": "zh",
    "yue": "yue",
    "cantonese": "yue",
}


def generate_synthetic_audio(duration: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate synthetic audio (white noise for testing).

    Args:
        duration: Audio duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Audio array of shape (num_samples,)
    """
    num_samples = int(duration * sample_rate)
    # White noise
    audio = np.random.randn(num_samples).astype(np.float32) * 0.1
    return audio


def create_dataset(
    output_dir: Path,
    languages: List[str],
    num_samples_per_lang: int = 10,
) -> Dict[str, int]:
    """Create a test dataset in JSONL format.

    Args:
        output_dir: Output directory for dataset
        languages: List of language codes (en, zh, yue)
        num_samples_per_lang: Number of samples per language

    Returns:
        Dictionary with sample counts per language
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create audio directory
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Create JSONL dataset file
    jsonl_path = output_dir / "dataset.jsonl"
    sample_count = 0
    counts_per_lang = {}

    with open(jsonl_path, "w") as f:
        for lang_name in languages:
            lang_code = LANGUAGE_CODES.get(lang_name.lower(), lang_name)

            if lang_code not in SAMPLE_TEXTS:
                logger.warning(f"Unsupported language: {lang_code}")
                continue

            texts = SAMPLE_TEXTS[lang_code]
            count = 0

            for i in range(num_samples_per_lang):
                # Use available texts cyclically
                text = texts[i % len(texts)]

                # Generate synthetic audio
                audio = generate_synthetic_audio(duration=1.0 + np.random.uniform(-0.3, 0.3))
                audio_tensor = torch.from_numpy(audio).unsqueeze(0)

                # Save audio file
                audio_filename = f"{lang_code}_{sample_count:04d}.wav"
                audio_path = audio_dir / audio_filename
                torchaudio.save(str(audio_path), audio_tensor, 16000)

                # Write JSONL entry
                entry = {
                    "text": text,
                    "lang": lang_code,
                    "audio_path": str(audio_path),
                }
                f.write(json.dumps(entry) + "\n")

                count += 1
                sample_count += 1

                if (count) % 5 == 0:
                    logger.info(f"  {lang_code}: Generated {count}/{num_samples_per_lang} samples")

            counts_per_lang[lang_code] = count
            logger.info(f"✅ Language {lang_code}: {count} samples")

    return jsonl_path, counts_per_lang


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic test dataset for Phase 1 validation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/test_dataset",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples per language",
    )
    parser.add_argument(
        "--lang",
        nargs="+",
        default=["en"],
        help="Languages to include (en, zh, yue)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Duration of synthetic audio in seconds",
    )

    args = parser.parse_args()

    logger.info("\n" + "=" * 70)
    logger.info("GENERATING SYNTHETIC TEST DATASET")
    logger.info("=" * 70 + "\n")

    logger.info(f"Output directory: {args.output}")
    logger.info(f"Languages: {args.lang}")
    logger.info(f"Samples per language: {args.num_samples}")
    logger.info(f"Audio duration: {args.duration}s\n")

    try:
        jsonl_path, counts = create_dataset(
            output_dir=args.output,
            languages=args.lang,
            num_samples_per_lang=args.num_samples,
        )

        total_samples = sum(counts.values())

        logger.info("\n" + "=" * 70)
        logger.info(f"✅ DATASET CREATED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"\nDataset file: {jsonl_path}")
        logger.info(f"Total samples: {total_samples}\n")

        for lang, count in counts.items():
            logger.info(f"  {lang}: {count} samples")

        logger.info("\n" + "=" * 70)
        logger.info("NEXT STEPS")
        logger.info("=" * 70)
        logger.info(f"\nUse this dataset in Phase 1:\n")
        logger.info(f"python scripts/capture_with_alignment.py \\")
        logger.info(f"  --model qwen3tts \\")
        logger.info(f"  --dataset custom \\")
        logger.info(f"  --dataset-file {jsonl_path} \\")
        logger.info(f"  --lang en \\")
        logger.info(f"  --output data/pilot_activations \\")
        logger.info(f"  --num-samples {total_samples} \\")
        logger.info(f"  --device cuda\n")

    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
