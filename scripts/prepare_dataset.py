#!/usr/bin/env python3
"""Prepare combined dataset: HF audio + out.jsonl codec tokens.

Merges indiejoseph/tts20250516 (audio, text, phone, spk_emb, etc.)
with out.jsonl (pre-extracted Qwen3-TTS codec tokens) by matching text.

Output: A parquet dataset with audio paths + codec + metadata, ready for
activation capture with generate_voice_clone().

Usage:
    # Build full dataset (all languages)
    PYTHONPATH="." python scripts/prepare_dataset.py \
        --hf-dataset indiejoseph/tts20250516 \
        --codec-file data/out.jsonl \
        --output data/combined_dataset

    # Build small debug subset
    PYTHONPATH="." python scripts/prepare_dataset.py \
        --hf-dataset indiejoseph/tts20250516 \
        --codec-file data/out.jsonl \
        --output data/combined_dataset \
        --lang en \
        --max-samples 100
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_codec_index(codec_file: Path, target_lang: str = None):
    """Index out.jsonl by text for fast lookup.

    Args:
        codec_file: Path to out.jsonl
        target_lang: If set, only index this language (e.g., "en", "zh", "yue")

    Returns:
        dict mapping normalized text -> {codec, lang, speech_token}
    """
    logger.info(f"Indexing {codec_file}...")
    index = {}
    lang_counts = defaultdict(int)
    skipped = 0

    with open(codec_file) as f:
        for line_num, line in enumerate(f):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            lang = row.get("lang", "")
            if target_lang and lang != target_lang:
                continue

            text = row.get("text", "").strip()
            if not text:
                continue

            # Normalize text for matching (strip whitespace, lowercase)
            key = text.strip()
            index[key] = {
                "codec": row.get("codec"),
                "speech_token": row.get("speech_token"),
                "lang": lang,
            }
            lang_counts[lang] += 1

    logger.info(f"Indexed {len(index)} records (skipped {skipped} parse errors)")
    for lang, count in sorted(lang_counts.items()):
        logger.info(f"  {lang}: {count}")

    return index


def prepare_dataset(args):
    """Main dataset preparation."""
    from datasets import load_dataset

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build codec index from out.jsonl
    codec_index = build_codec_index(
        Path(args.codec_file),
        target_lang=args.lang,
    )

    # Step 2: Stream HF dataset and match with codec index
    logger.info(f"Loading HF dataset: {args.hf_dataset}...")
    ds = load_dataset(
        args.hf_dataset,
        split="train",
        streaming=True,
    )

    matched = []
    unmatched = 0
    processed = 0
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    import soundfile as sf
    import numpy as np

    for row in ds:
        text = row.get("text", "").strip()
        if not text:
            continue

        # Match with codec index
        codec_entry = codec_index.get(text)
        if codec_entry is None:
            unmatched += 1
            continue

        # Language filter
        if args.lang and codec_entry["lang"] != args.lang:
            continue

        # Save audio to disk
        audio = row["audio"]
        audio_filename = f"sample_{processed:06d}.wav"
        audio_path = audio_dir / audio_filename
        sf.write(str(audio_path), audio["array"], audio["sampling_rate"])

        # Build combined record
        record = {
            "id": processed,
            "text": text,
            "lang": codec_entry["lang"],
            "audio_path": str(audio_path),
            "sample_rate": audio["sampling_rate"],
            "duration": row.get("duration", 0.0),
            "phone": row.get("phone", ""),
            "codec": codec_entry["codec"],
            "speech_token": codec_entry["speech_token"],
            "spk_emb": row.get("spk_emb"),
        }
        matched.append(record)
        processed += 1

        if processed % 1000 == 0:
            logger.info(f"  Processed {processed} samples (unmatched: {unmatched})...")

        if args.max_samples and processed >= args.max_samples:
            logger.info(f"Reached max_samples={args.max_samples}, stopping.")
            break

    logger.info(f"\nMatching complete:")
    logger.info(f"  Matched: {len(matched)}")
    logger.info(f"  Unmatched: {unmatched}")

    if not matched:
        logger.error("No matched samples! Check that out.jsonl texts match HF dataset texts.")
        return

    # Step 3: Save combined dataset
    # Save as JSONL (lightweight, easy to load)
    jsonl_path = output_dir / "dataset.jsonl"
    logger.info(f"Saving dataset to {jsonl_path}...")

    # For JSONL, exclude large fields that are saved separately
    with open(jsonl_path, "w") as f:
        for record in matched:
            # Save codec inline (it's the key addition)
            # Save spk_emb as list of floats
            row = {
                "id": record["id"],
                "text": record["text"],
                "lang": record["lang"],
                "audio_path": record["audio_path"],
                "sample_rate": record["sample_rate"],
                "duration": record["duration"],
                "phone": record["phone"],
                "codec": record["codec"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Save speaker embeddings separately (they're large float arrays)
    spk_emb_path = output_dir / "spk_embeddings.jsonl"
    with open(spk_emb_path, "w") as f:
        for record in matched:
            if record.get("spk_emb") is not None:
                f.write(json.dumps({
                    "id": record["id"],
                    "spk_emb": record["spk_emb"],
                }) + "\n")

    # Save metadata
    meta = {
        "hf_dataset": args.hf_dataset,
        "codec_file": args.codec_file,
        "lang_filter": args.lang,
        "total_samples": len(matched),
        "unmatched": unmatched,
        "lang_distribution": {},
    }
    lang_dist = defaultdict(int)
    for r in matched:
        lang_dist[r["lang"]] += 1
    meta["lang_distribution"] = dict(lang_dist)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Dataset saved to {output_dir}")
    logger.info(f"  dataset.jsonl:       {len(matched)} records (text + audio_path + codec)")
    logger.info(f"  spk_embeddings.jsonl: speaker embeddings")
    logger.info(f"  audio/:              {len(matched)} wav files")
    logger.info(f"  metadata.json:       dataset info")
    logger.info(f"  Language distribution: {dict(lang_dist)}")
    logger.info(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare combined dataset: HF audio + out.jsonl codec"
    )
    parser.add_argument(
        "--hf-dataset",
        default="indiejoseph/tts20250516",
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--codec-file",
        type=Path,
        default=Path("data/out.jsonl"),
        help="Path to out.jsonl with pre-extracted codec tokens",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/combined_dataset"),
        help="Output directory",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "zh", "yue", None],
        default=None,
        help="Filter to specific language (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to process (for debugging)",
    )

    args = parser.parse_args()
    prepare_dataset(args)


if __name__ == "__main__":
    main()
