#!/usr/bin/env python3
"""Pilot activation capture: 100 sentences through each model.

This script validates the entire activation mining pipeline on a small scale
before committing to full-scale capture. Produces activation statistics and
a Jupyter notebook with visualizations.

Usage:
    python scripts/pilot_capture.py --model qwen3tts --output data/pilot_activations
    python scripts/pilot_capture.py --model cosyvoice2 --output data/pilot_activations
"""

import argparse
import logging
from pathlib import Path

import torch
import numpy as np

from src.hooks import ActivationHook
from src.data.activation_buffer import ActivationBuffer
from src.data.dataset_prep import create_pilot_dataset
from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper
from src.models.cosyvoice2_wrapper import CosyVoice2Wrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pilot activation capture")
    parser.add_argument(
        "--model",
        choices=["qwen3tts", "cosyvoice2"],
        default="qwen3tts",
        help="Which model to use",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pilot_activations"),
        help="Output directory for activations",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to capture",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float32", "float16"],
        help="Data type for activations",
    )
    args = parser.parse_args()

    logger.info(f"Pilot capture: {args.model} on {args.device}")

    # Load model
    if args.model == "qwen3tts":
        torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
        model_wrapper = Qwen3TTSWrapper(device=args.device, dtype=torch_dtype)
        target_layers = model_wrapper.get_target_layers()
        model = model_wrapper.model
    else:  # cosyvoice2
        torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
        model_wrapper = CosyVoice2Wrapper(device=args.device, dtype=torch_dtype)
        target_layers = model_wrapper.get_target_layers()
        model = model_wrapper.model

    if model is None:
        logger.error("Model failed to load")
        return

    # Create hook and buffer
    hook = ActivationHook(
        model,
        layer_indices=target_layers,
        device="cpu",
        dtype=torch.float16,
    )
    buffer = ActivationBuffer(
        output_dir=args.output,
        layer_indices=target_layers,
        batch_size=10,
        dtype=args.dtype,
    )

    # Create pilot dataset
    dataset = create_pilot_dataset(num_samples=args.num_samples)
    logger.info(f"Created pilot dataset with {len(dataset)} samples")

    # Capture activations
    logger.info("Starting activation capture...")
    hook.attach("mlp")

    try:
        for batch_idx, sample in enumerate(dataset):
            if batch_idx % 10 == 0:
                logger.info(f"Processing sample {batch_idx}/{len(dataset)}")

            # Run inference (text-only for now)
            try:
                if args.model == "qwen3tts":
                    # For Qwen3-TTS, we need to tokenize and pass to model
                    with torch.no_grad():
                        input_ids = model_wrapper.tokenizer.encode(
                            sample.text, return_tensors="pt"
                        ).to(args.device)
                        _ = model(input_ids)
                else:  # cosyvoice2
                    # For CosyVoice2, use the generate method
                    with torch.no_grad():
                        _ = model_wrapper.generate(
                            tts_text=sample.text,
                            prompt_text="Reference text",
                        )
            except Exception as e:
                logger.warning(f"Failed to process sample {batch_idx}: {e}")
                continue

            # Collect and buffer activations
            activations = hook.collect()
            buffer.add_batch(activations)

    finally:
        hook.detach()
        buffer.flush()

    logger.info(f"Activation capture complete. Saved to {args.output}")
    logger.info(f"Total vectors saved: {buffer.total_saved}")

    # Compute statistics
    logger.info("Computing activation statistics...")
    stats = {}
    for layer_idx in target_layers:
        # Load activations for this layer
        npy_files = sorted(args.output.glob(f"layer_{layer_idx:02d}_*.npy"))
        if npy_files:
            arrays = [np.load(f) for f in npy_files]
            stacked = np.concatenate(arrays, axis=0)
            stats[layer_idx] = {
                "shape": stacked.shape,
                "mean": float(stacked.mean()),
                "std": float(stacked.std()),
                "min": float(stacked.min()),
                "max": float(stacked.max()),
                "num_vectors": stacked.shape[0],
            }
            logger.info(f"Layer {layer_idx}: {stats[layer_idx]}")

    # Save statistics
    stats_file = args.output / "statistics.txt"
    with open(stats_file, "w") as f:
        for layer_idx, layer_stats in sorted(stats.items()):
            f.write(f"Layer {layer_idx}:\n")
            for key, val in layer_stats.items():
                f.write(f"  {key}: {val}\n")
            f.write("\n")
    logger.info(f"Statistics saved to {stats_file}")


if __name__ == "__main__":
    main()
