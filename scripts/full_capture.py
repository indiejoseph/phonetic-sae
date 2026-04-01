#!/usr/bin/env python3
"""Full-scale activation capture: 50K+ sentences for SAE training.

This script:
1. Loads a TTS model (Qwen3-TTS or CosyVoice2)
2. Attaches activation hooks to phonetic layers
3. Runs a large dataset through the model
4. Saves activations efficiently to disk (FP16)

Usage:
    python scripts/full_capture.py \
        --model qwen3tts \
        --dataset libritts \
        --output data/activations/qwen3tts \
        --num-samples 50000

    python scripts/full_capture.py \
        --model cosyvoice2 \
        --dataset custom \
        --dataset-file data/my_dataset.csv \
        --output data/activations/cosyvoice2 \
        --num-samples 50000
"""

import argparse
import logging
from pathlib import Path

import torch

from src.hooks import ActivationHook
from src.data.activation_buffer import ActivationBuffer
from src.data.dataset_prep import (
    LibriTTSRDataset,
    CustomDataset,
    create_pilot_dataset,
)
from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper
from src.models.cosyvoice2_wrapper import CosyVoice2Wrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Full-scale activation capture")
    parser.add_argument(
        "--model",
        choices=["qwen3tts", "cosyvoice2"],
        default="qwen3tts",
        help="Which model to use",
    )
    parser.add_argument(
        "--dataset",
        choices=["pilot", "libritts", "custom"],
        default="libritts",
        help="Which dataset to use",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help="Path to custom dataset file (CSV or JSONL, required if --dataset=custom)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/activations"),
        help="Output directory for activations",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50000,
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for activation buffer flushing",
    )
    args = parser.parse_args()

    logger.info(
        f"Full-scale capture: {args.model} on {args.device}, "
        f"dataset={args.dataset}, num_samples={args.num_samples}"
    )

    # Load model
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    if args.model == "qwen3tts":
        model_wrapper = Qwen3TTSWrapper(device=args.device, dtype=torch_dtype)
        target_layers = model_wrapper.get_target_layers()
        model = model_wrapper.model
        tokenizer = model_wrapper.tokenizer
    else:  # cosyvoice2
        model_wrapper = CosyVoice2Wrapper(device=args.device, dtype=torch_dtype)
        target_layers = model_wrapper.get_target_layers()
        model = model_wrapper.model
        tokenizer = None

    if model is None:
        logger.error("Model failed to load")
        return

    logger.info(f"Model loaded. Target layers: {target_layers}")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    if args.dataset == "pilot":
        dataset_obj = None
        dataset = create_pilot_dataset(num_samples=args.num_samples)
        logger.info(f"Created pilot dataset with {len(dataset)} samples")
    elif args.dataset == "libritts":
        try:
            dataset_obj = LibriTTSRDataset("data/datasets/LibriTTS_R")
            dataset_iter = dataset_obj.iterator(batch_size=1)
            # Limit to num_samples
            samples = []
            for batch in dataset_iter:
                samples.extend(batch)
                if len(samples) >= args.num_samples:
                    break
            dataset = samples[: args.num_samples]
            logger.info(f"Loaded {len(dataset)} LibriTTS-R samples")
        except FileNotFoundError:
            logger.error("LibriTTS-R not found at data/datasets/LibriTTS_R")
            logger.info("Falling back to pilot dataset")
            dataset = create_pilot_dataset(num_samples=args.num_samples)
    elif args.dataset == "custom":
        if args.dataset_file is None:
            logger.error("--dataset-file required for custom dataset")
            return
        try:
            dataset_obj = CustomDataset(args.dataset_file)
            dataset_iter = dataset_obj.iterator(batch_size=1)
            samples = []
            for batch in dataset_iter:
                samples.extend(batch)
                if len(samples) >= args.num_samples:
                    break
            dataset = samples[: args.num_samples]
            logger.info(f"Loaded {len(dataset)} custom samples")
        except Exception as e:
            logger.error(f"Failed to load custom dataset: {e}")
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
        batch_size=args.batch_size,
        dtype=args.dtype,
    )

    # Capture activations
    logger.info("Starting activation capture...")
    hook.attach("mlp")

    try:
        for batch_idx, sample in enumerate(dataset):
            if batch_idx % 100 == 0:
                logger.info(f"Processing sample {batch_idx}/{len(dataset)}")

            # Run inference
            try:
                with torch.no_grad():
                    if args.model == "qwen3tts":
                        # Tokenize and pass to model
                        input_ids = tokenizer.encode(
                            sample.text, return_tensors="pt"
                        ).to(args.device)
                        _ = model(input_ids)
                    else:  # cosyvoice2
                        # Use generate method
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
    logger.info(f"Total vectors saved per layer: {buffer.total_saved}")

    # Summary
    logger.info("=" * 60)
    logger.info("CAPTURE SUMMARY")
    logger.info("=" * 60)
    for layer_idx, count in sorted(buffer.total_saved.items()):
        logger.info(f"Layer {layer_idx:02d}: {count:,} vectors")
    total = sum(buffer.total_saved.values())
    logger.info(f"Total: {total:,} vectors across {len(buffer.total_saved)} layers")
    logger.info(f"Output directory: {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
