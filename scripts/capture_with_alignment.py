#!/usr/bin/env python3
"""Enhanced activation capture with phoneme alignment.

Captures activations AND phoneme-to-frame alignment for SAE feature analysis.

Usage:
    python scripts/capture_with_alignment.py \
        --model qwen3tts \
        --dataset-file data/out.jsonl \
        --lang en \
        --output data/activations_aligned/en
"""

import argparse
import logging
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import torch
import numpy as np

from src.hooks import ActivationHook
from src.data.activation_buffer import ActivationBuffer
from src.data.dataset_prep import CustomDataset
from src.alignment import QwenForcedAligner, LanguageSpecificAligner
from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper
from src.models.cosyvoice2_wrapper import CosyVoice2Wrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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
        """Add activations with phoneme alignment.

        Args:
            activations: dict mapping layer index to activation tensor
            alignment: PhonemeAlignment object from forced aligner
            sample_id: Sample identifier
        """
        phonemes = alignment.phonemes
        frame_to_phoneme = alignment.frame_to_phoneme

        # Store frame-level labels
        self.frame_labels.append({
            "sample_id": sample_id,
            "phoneme_sequence": phonemes,
            "num_frames": len(frame_to_phoneme),
        })

        # Reorganize activations by phoneme
        for layer_idx, layer_acts in activations.items():
            # layer_acts shape: (num_frames, d_model)
            if layer_acts.dim() == 1:
                layer_acts = layer_acts.unsqueeze(0)

            for frame_idx in range(layer_acts.shape[0]):
                if frame_idx < len(frame_to_phoneme):
                    phoneme_idx = frame_to_phoneme[frame_idx]
                    phoneme = phonemes[phoneme_idx] if phoneme_idx < len(phonemes) else "unk"

                    # Store activation with layer and phoneme index
                    key = (layer_idx, phoneme)
                    self.phoneme_activations[key].append(
                        layer_acts[frame_idx].detach().cpu().numpy()
                    )
                    self.phoneme_counts[phoneme] += 1

    def flush(self):
        """Save phoneme-organized activations to disk."""
        logger.info(f"Flushing {len(self.phoneme_activations)} layer-phoneme combinations...")

        # Save per-layer, per-phoneme data
        for (layer_idx, phoneme), acts_list in self.phoneme_activations.items():
            if acts_list:
                acts_array = np.stack(acts_list, axis=0)  # (n_frames, d_model)

                layer_dir = self.output_dir / f"layer_{layer_idx:02d}"
                layer_dir.mkdir(exist_ok=True)

                # Save as numpy file
                output_file = layer_dir / f"phoneme_{phoneme}.npy"
                np.save(output_file, acts_array)

        # Save phoneme inventory and counts
        inventory_file = self.output_dir / "phoneme_inventory.json"
        with open(inventory_file, "w") as f:
            json.dump({
                "language": self.language,
                "phoneme_counts": {k: int(v) for k, v in self.phoneme_counts.items()},
                "total_frames": sum(self.phoneme_counts.values()),
            }, f, indent=2)

        # Save frame labels
        labels_file = self.output_dir / "frame_labels.jsonl"
        with open(labels_file, "w") as f:
            for label in self.frame_labels:
                f.write(json.dumps(label) + "\n")

        logger.info(f"✅ Saved phoneme-organized activations to {self.output_dir}")
        logger.info(f"   Phoneme inventory: {len(self.phoneme_counts)} unique phonemes")
        logger.info(f"   Total frames: {sum(self.phoneme_counts.values())}")


def main():
    parser = argparse.ArgumentParser(
        description="Capture activations with phoneme alignment"
    )
    parser.add_argument(
        "--model",
        choices=["qwen3tts", "cosyvoice2"],
        default="qwen3tts",
        help="Which model to use",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        required=True,
        help="Path to dataset file (CSV or JSONL format)",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "zh", "yue"],
        default="en",
        help="Language code for alignment",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/activations_aligned"),
        help="Output directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to process",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default="float16",
        help="Data type",
    )

    args = parser.parse_args()

    logger.info(
        f"Enhanced capture: {args.model} on {args.device}, "
        f"language={args.lang}, num_samples={args.num_samples}"
    )

    # Load model
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    if args.model == "qwen3tts":
        model_wrapper = Qwen3TTSWrapper(device=args.device, dtype=torch_dtype)
        target_layers = model_wrapper.get_target_layers()
        model = model_wrapper.model
    else:
        model_wrapper = CosyVoice2Wrapper(device=args.device, dtype=torch_dtype)
        target_layers = model_wrapper.get_target_layers()
        model = model_wrapper.model

    if model is None:
        logger.error("Model failed to load")
        return

    logger.info(f"Model loaded. Target layers: {target_layers}")

    # Load dataset
    try:
        dataset = CustomDataset(args.dataset_file)
        pairs = dataset.filter_by_language(
            {"en": "English", "zh": "Mandarin", "yue": "Cantonese"}[args.lang]
        )
        pairs = pairs[:args.num_samples]
        logger.info(f"Loaded {len(pairs)} samples for language {args.lang}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Load forced aligner
    try:
        aligner = QwenForcedAligner(device=args.device, language=args.lang)
        logger.info(f"✅ Forced aligner loaded for {args.lang}")
    except Exception as e:
        logger.error(f"Failed to load forced aligner: {e}")
        logger.warning("Falling back to standard activation capture (no alignment)")
        aligner = None

    # Create hook and buffer
    hook = ActivationHook(
        model,
        layer_indices=target_layers,
        device="cpu",
        dtype=torch.float16,
    )
    buffer = AlignedActivationBuffer(
        output_dir=args.output / args.lang,
        layer_indices=target_layers,
        language=args.lang,
    )

    hook.attach("mlp")
    logger.info("Hook attached to MLP post-activations")

    # Process samples
    processed = 0
    alignment_errors = 0

    for sample_idx, pair in enumerate(pairs):
        try:
            if not pair.text:
                logger.warning(f"Sample {sample_idx}: empty text, skipping")
                continue

            logger.info(
                f"[{sample_idx + 1}/{len(pairs)}] Processing: {pair.text[:60]}..."
            )

            # Forward pass
            input_ids = model_wrapper.tokenizer.encode(
                pair.text, return_tensors="pt"
            ).to(args.device)

            with torch.no_grad():
                _ = model(input_ids)

            # Collect activations
            activations = hook.collect()
            hook.reset()

            # Align (optional)
            if aligner and pair.ref_audio_path:
                try:
                    alignment = aligner.align(
                        text=pair.text,
                        audio_path=pair.ref_audio_path,
                    )
                    buffer.add_aligned_activations(activations, alignment, sample_idx)
                except Exception as e:
                    logger.warning(f"Alignment failed for sample {sample_idx}: {e}")
                    alignment_errors += 1
            else:
                logger.warning(
                    f"Sample {sample_idx}: no aligner or no audio path, skipping alignment"
                )

            processed += 1

            if processed % 10 == 0:
                logger.info(f"Progress: {processed}/{len(pairs)} samples processed")

        except Exception as e:
            logger.error(f"Error processing sample {sample_idx}: {e}")
            continue

    hook.detach()
    buffer.flush()

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Capture Complete:")
    logger.info(f"  Total samples: {len(pairs)}")
    logger.info(f"  Successfully processed: {processed}")
    logger.info(f"  Alignment errors: {alignment_errors}")
    logger.info(f"  Output directory: {args.output / args.lang}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
