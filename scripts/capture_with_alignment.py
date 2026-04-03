#!/usr/bin/env python3
"""Enhanced activation capture with phoneme alignment.

Captures activations AND phoneme-to-frame alignment for SAE feature analysis.
Uses combined dataset (from merge_dataset.py / save_to_disk()) which has
audio, text, codec, lang, phone, spk_emb.

Usage:
    PYTHONPATH="." python scripts/capture_with_alignment.py \
        --model qwen3tts \
        --dataset data/combined_yue \
        --output data/activations_aligned \
        --num-samples 100 \
        --device cuda
"""

import argparse
import logging
import json
import tempfile
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import torch
import numpy as np
import soundfile as sf

from src.hooks import ActivationHook
from src.alignment import QwenForcedAligner
from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper
from src.models.cosyvoice2_wrapper import CosyVoice2Wrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Qwen3-TTS has no "Cantonese" language — yue maps to "Chinese"
LANG_TO_QWEN3 = {"en": "English", "zh": "Chinese", "yue": "Chinese"}


@dataclass
class AlignedActivationBuffer:
    """Buffer for storing activations organized by phoneme."""

    output_dir: Path
    layer_indices: list[int]
    language: str
    batch_size: int = 512

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Per-phoneme activation storage
        self.phoneme_activations = defaultdict(list)  # phoneme -> [activations]
        self.phoneme_counts = defaultdict(int)
        self.frame_labels = []

    def add_aligned_activations(
        self,
        activations: dict[int, torch.Tensor],  # layer_idx -> activation
        alignment,  # PhonemeAlignment object
        sample_id: int,
    ):
        """Add activations with phoneme alignment."""
        phonemes = alignment.phonemes
        frame_to_phoneme = alignment.frame_to_phoneme

        self.frame_labels.append({
            "sample_id": sample_id,
            "phoneme_sequence": phonemes,
            "num_frames": len(frame_to_phoneme),
        })

        for layer_idx, layer_acts in activations.items():
            # layer_acts shape: (num_frames, d_model)
            if layer_acts.dim() == 1:
                layer_acts = layer_acts.unsqueeze(0)

            for frame_idx in range(layer_acts.shape[0]):
                if frame_idx < len(frame_to_phoneme):
                    phoneme_idx = frame_to_phoneme[frame_idx]
                    phoneme = phonemes[phoneme_idx] if phoneme_idx < len(phonemes) else "unk"

                    key = (layer_idx, phoneme)
                    self.phoneme_activations[key].append(
                        layer_acts[frame_idx].detach().cpu().numpy()
                    )
                    self.phoneme_counts[phoneme] += 1

    def flush(self):
        """Save phoneme-organized activations to disk."""
        logger.info(f"Flushing {len(self.phoneme_activations)} layer-phoneme combinations...")

        for (layer_idx, phoneme), acts_list in self.phoneme_activations.items():
            if acts_list:
                acts_array = np.stack(acts_list, axis=0)

                layer_dir = self.output_dir / f"layer_{layer_idx:02d}"
                layer_dir.mkdir(exist_ok=True)

                output_file = layer_dir / f"phoneme_{phoneme}.npy"
                np.save(output_file, acts_array)

        inventory_file = self.output_dir / "phoneme_inventory.json"
        with open(inventory_file, "w") as f:
            json.dump({
                "language": self.language,
                "phoneme_counts": {k: int(v) for k, v in self.phoneme_counts.items()},
                "total_frames": sum(self.phoneme_counts.values()),
            }, f, indent=2)

        labels_file = self.output_dir / "frame_labels.jsonl"
        with open(labels_file, "w") as f:
            for label in self.frame_labels:
                f.write(json.dumps(label) + "\n")

        logger.info(f"✅ Saved phoneme-organized activations to {self.output_dir}")
        logger.info(f"   Phoneme inventory: {len(self.phoneme_counts)} unique phonemes")
        logger.info(f"   Total frames: {sum(self.phoneme_counts.values())}")


def load_combined_dataset(dataset_path: str, lang: str = None, num_samples: int = None):
    """Load combined dataset from save_to_disk().

    Returns:
        ds: HF Dataset with audio, text, codec, lang, phone, etc.
    """
    from datasets import load_from_disk

    logger.info(f"Loading dataset from {dataset_path}...")
    ds = load_from_disk(dataset_path)
    logger.info(f"Dataset loaded: {len(ds)} samples")

    # Filter by language if specified
    if lang:
        ds = ds.filter(lambda x: x.get("lang") == lang)
        logger.info(f"Filtered to lang={lang}: {len(ds)} samples")

    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))
        logger.info(f"Selected first {num_samples} samples")

    return ds


def save_audio_to_temp(audio_dict) -> str:
    """Save HF audio dict to a temp wav file, return path."""
    tmp_dir = Path(tempfile.gettempdir()) / "phonetic_sae_capture"
    tmp_dir.mkdir(exist_ok=True)
    # Use a fixed name to avoid filling /tmp — overwritten each sample
    tmp_path = str(tmp_dir / "current_sample.wav")
    sf.write(tmp_path, audio_dict["array"], audio_dict["sampling_rate"])
    return tmp_path


def main():
    parser = argparse.ArgumentParser(
        description="Capture activations with phoneme alignment"
    )
    parser.add_argument(
        "--model",
        choices=["qwen3tts", "cosyvoice2"],
        default="qwen3tts",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to combined dataset (from save_to_disk())",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "zh", "yue"],
        default=None,
        help="Filter to language (default: use all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/activations_aligned"),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--ref-index",
        type=int,
        default=0,
        help="Dataset index to use as reference speaker for voice cloning",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max new tokens per generation",
    )

    args = parser.parse_args()

    logger.info(
        f"Capture: {args.model} on {args.device}, "
        f"lang={args.lang}, num_samples={args.num_samples}"
    )

    # Load model
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    if args.model == "qwen3tts":
        model_wrapper = Qwen3TTSWrapper(device=args.device, dtype=torch_dtype)
    else:
        model_wrapper = CosyVoice2Wrapper(device=args.device, dtype=torch_dtype)

    target_layers = model_wrapper.get_target_layers()
    model = model_wrapper.model
    layer_accessor = model_wrapper.get_layer_accessor()

    if model is None:
        logger.error("Model failed to load")
        return

    logger.info(f"Model loaded. Target layers: {target_layers}")

    # Load combined dataset
    ds = load_combined_dataset(args.dataset, lang=args.lang, num_samples=args.num_samples + 1)

    if len(ds) < 2:
        logger.error("Need at least 2 samples (1 ref + 1 synthesis)")
        return

    # Prepare reference speaker (sample at ref_index)
    ref_row = ds[args.ref_index]
    ref_audio_path = save_audio_to_temp(ref_row["audio"])
    ref_text = ref_row["text"]
    logger.info(f"Reference speaker: index={args.ref_index}, text={ref_text[:60]}...")

    # Synthesis samples = all except ref
    synth_indices = [i for i in range(len(ds)) if i != args.ref_index]
    if args.num_samples:
        synth_indices = synth_indices[:args.num_samples]
    logger.info(f"Synthesis targets: {len(synth_indices)} samples")

    # Load forced aligner (optional)
    aligner = None
    align_lang = args.lang or ref_row.get("lang", "")
    if align_lang:
        try:
            aligner = QwenForcedAligner(device=args.device, language=align_lang)
            logger.info(f"✅ Forced aligner loaded for {align_lang}")
        except Exception as e:
            logger.warning(f"Forced aligner not available: {e}")
            aligner = None

    # Determine output subdirectory
    out_lang = args.lang or "all"
    output_dir = args.output / out_lang

    # Create hook and buffer
    hook = ActivationHook(
        model,
        layer_indices=target_layers,
        layer_accessor=layer_accessor,
        device="cpu",
        dtype=torch.float16,
    )
    buffer = AlignedActivationBuffer(
        output_dir=output_dir,
        layer_indices=target_layers,
        language=out_lang,
    )

    hook.attach("mlp")
    logger.info("Hook attached to MLP post-activations")

    # Process samples
    processed = 0
    alignment_errors = 0

    for count, sample_idx in enumerate(synth_indices):
        try:
            row = ds[sample_idx]
            text = row["text"].strip()
            lang = row.get("lang", "")

            if not text:
                logger.warning(f"Sample {sample_idx}: empty text, skipping")
                continue

            logger.info(f"[{count + 1}/{len(synth_indices)}] lang={lang} text={text[:60]}...")

            # Save sample audio for alignment (not ref audio)
            sample_audio_path = save_audio_to_temp(row["audio"])

            # Generate with voice cloning
            with torch.no_grad():
                if args.model == "qwen3tts":
                    language = LANG_TO_QWEN3.get(lang, "Auto")
                    wavs, sr = model_wrapper.model.generate_voice_clone(
                        text=text,
                        language=language,
                        ref_audio=ref_audio_path,
                        ref_text=ref_text,
                        x_vector_only_mode=True,
                        max_new_tokens=args.max_tokens,
                        do_sample=True,
                        temperature=0.9,
                        top_k=50,
                    )
                else:  # cosyvoice2
                    _ = model_wrapper.generate(
                        tts_text=text,
                        prompt_text=ref_text,
                        prompt_wav=ref_audio_path,
                    )

            # Collect activations
            activations = hook.collect()

            # Align phonemes to activations (optional)
            if aligner:
                try:
                    alignment = aligner.align(
                        text=text,
                        audio_path=sample_audio_path,
                    )
                    buffer.add_aligned_activations(activations, alignment, sample_idx)
                except Exception as e:
                    logger.warning(f"Alignment failed for sample {sample_idx}: {e}")
                    alignment_errors += 1
            else:
                # No aligner — still store raw activations
                # Use a dummy alignment that maps all frames to "unk"
                pass

            processed += 1

            if processed % 10 == 0:
                logger.info(f"Progress: {processed}/{len(synth_indices)} samples processed")

        except Exception as e:
            logger.error(f"Error processing sample {sample_idx}: {e}")
            # Clear any partial hook data
            hook.collect()
            continue

    hook.detach()
    buffer.flush()

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Capture Complete:")
    logger.info(f"  Total samples: {len(synth_indices)}")
    logger.info(f"  Successfully processed: {processed}")
    logger.info(f"  Alignment errors: {alignment_errors}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
