#!/usr/bin/env python3
"""Merge indiejoseph/tts20250516 (audio) + out.jsonl (codec) → combined dataset.

Usage:
    HF_TOKEN=hf_xxx python merge_dataset.py \
        --codec-file data/out.jsonl \
        --output data/combined \
        --lang yue \
        --max-samples 100  # optional, for testing
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import soundfile as sf
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def build_codec_index(codec_file, target_lang=None):
    """Index out.jsonl by text → codec."""
    index = {}
    with open(codec_file) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if target_lang and row.get("lang") != target_lang:
                continue
            text = row.get("text", "").strip()
            if text:
                index[text] = row["codec"]
    logger.info(f"Indexed {len(index)} codec entries (lang={target_lang})")
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dataset", default="indiejoseph/tts20250516")
    parser.add_argument("--codec-file", type=Path, default=Path("data/out.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/combined"))
    parser.add_argument("--lang", default=None, help="Filter: en, zh, yue")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    out = args.output
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    codec_index = build_codec_index(args.codec_file, target_lang=args.lang)

    ds = load_dataset(args.hf_dataset, split="train", streaming=True)

    matched = 0
    unmatched = 0

    with open(out / "dataset.jsonl", "w") as f_out:
        for row in ds:
            text = row.get("text", "").strip()
            codec = codec_index.get(text)
            if codec is None:
                unmatched += 1
                continue

            # Save audio
            audio = row["audio"]
            wav_path = audio_dir / f"{matched:06d}.wav"
            sf.write(str(wav_path), audio["array"], audio["sampling_rate"])

            # Write combined record
            f_out.write(
                json.dumps(
                    {
                        "id": matched,
                        "text": text,
                        "audio_path": str(wav_path),
                        "sample_rate": audio["sampling_rate"],
                        "duration": row.get("duration", 0.0),
                        "phone": row.get("phone", ""),
                        "codec": codec,
                        "spk_emb": row.get("spk_emb"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            matched += 1
            if matched % 1000 == 0:
                logger.info(f"  {matched} matched, {unmatched} unmatched")
            if args.max_samples and matched >= args.max_samples:
                break

    logger.info(f"Done: {matched} matched, {unmatched} unmatched → {out}/dataset.jsonl")


if __name__ == "__main__":
    main()
