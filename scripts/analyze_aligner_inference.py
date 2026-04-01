#!/usr/bin/env python3
"""Analyze Qwen3-ForcedAligner model internals and inference flow.

This script:
1. Loads the forced aligner model
2. Inspects architecture and layers
3. Profiles inference performance
4. Documents the inference pipeline

Usage:
    python scripts/analyze_aligner_inference.py
    python scripts/analyze_aligner_inference.py --lang zh
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def analyze_model_architecture(model, language: str) -> Dict[str, Any]:
    """Analyze model architecture and layer composition."""
    info = {
        "language": language,
        "model_type": type(model).__name__,
        "device": next(model.parameters()).device,
        "dtype": next(model.parameters()).dtype,
    }

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info["total_parameters"] = total_params
    info["trainable_parameters"] = trainable_params
    info["size_mb"] = (total_params * 4) / (1024 * 1024)  # Assuming float32

    # Inspect model structure
    if hasattr(model, "config"):
        config = model.config
        logger.info(f"\nModel Config:")
        for key in list(config.__dict__.keys())[:15]:
            val = getattr(config, key)
            if not callable(val) and not key.startswith("_"):
                logger.info(f"  {key}: {val}")
                info[f"config_{key}"] = str(val)

    # List major modules
    logger.info(f"\nModel Modules:")
    major_modules = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and isinstance(module, torch.nn.Module):
            if any(p in name.lower() for p in ["encoder", "decoder", "attention", "linear"]):
                major_modules.append(name)
                if len(major_modules) <= 20:
                    logger.info(f"  {name}: {type(module).__name__}")

    info["major_modules"] = major_modules[:20]

    return info


def profile_inference(model, language: str, num_samples: int = 10) -> Dict[str, Any]:
    """Profile inference performance."""
    logger.info(f"\n{'='*70}")
    logger.info(f"INFERENCE PROFILING ({language})")
    logger.info(f"{'='*70}")

    # Mock inputs (audio + phonemes)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Create dummy inputs
    audio_length = 1000  # ~10 seconds at 100 fps
    hidden_dim = 768
    phoneme_length = 50

    dummy_audio = torch.randn(
        1, audio_length, 80,  # (batch, time, mel_freq)
        device=device,
        dtype=dtype
    )
    dummy_phonemes = torch.randint(0, 40, (1, phoneme_length), device=device)

    # Warmup
    logger.info(f"Warming up...")
    with torch.no_grad():
        for _ in range(3):
            try:
                _ = model(dummy_audio, dummy_phonemes)
            except Exception:
                pass

    # Profile
    logger.info(f"Profiling {num_samples} inference runs...")
    times = []
    max_memory = 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for i in range(num_samples):
            start_time = time.time()
            try:
                output = model(dummy_audio, dummy_phonemes)
                elapsed = time.time() - start_time
                times.append(elapsed * 1000)  # milliseconds

                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / 1024 / 1024
                    max_memory = max(max_memory, current_memory)

                if (i + 1) % (num_samples // 3) == 0:
                    logger.info(f"  Completed {i + 1}/{num_samples} runs")

            except Exception as e:
                logger.warning(f"  Run {i} failed: {e}")

    torch.cuda.reset_peak_memory_stats()

    # Compute statistics
    times = np.array(times)
    profile_info = {
        "num_samples": num_samples,
        "mean_latency_ms": float(np.mean(times)),
        "median_latency_ms": float(np.median(times)),
        "min_latency_ms": float(np.min(times)),
        "max_latency_ms": float(np.max(times)),
        "std_latency_ms": float(np.std(times)),
        "peak_memory_mb": max_memory,
        "throughput_samples_per_sec": 1000.0 / float(np.mean(times)),
    }

    logger.info(f"\nInference Performance:")
    logger.info(f"  Mean latency: {profile_info['mean_latency_ms']:.2f} ms")
    logger.info(f"  Median latency: {profile_info['median_latency_ms']:.2f} ms")
    logger.info(f"  Std dev: {profile_info['std_latency_ms']:.2f} ms")
    logger.info(f"  Throughput: {profile_info['throughput_samples_per_sec']:.1f} samples/sec")
    if max_memory > 0:
        logger.info(f"  Peak memory: {max_memory:.1f} MB")

    return profile_info


def analyze_model_layers(model) -> Dict[str, Any]:
    """Analyze layer-by-layer composition."""
    logger.info(f"\n{'='*70}")
    logger.info(f"LAYER ANALYSIS")
    logger.info(f"{'='*70}")

    layer_info = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            layer_info[name] = {
                "type": "Linear",
                "in_features": module.in_features,
                "out_features": module.out_features,
                "params": module.weight.numel() + (module.bias.numel() if module.bias is not None else 0),
            }

        elif isinstance(module, torch.nn.TransformerEncoderLayer):
            layer_info[name] = {
                "type": "TransformerEncoderLayer",
                "d_model": module.self_attn.embed_dim if hasattr(module, "self_attn") else "unknown",
            }

        elif isinstance(module, torch.nn.TransformerDecoderLayer):
            layer_info[name] = {
                "type": "TransformerDecoderLayer",
                "d_model": module.self_attn.embed_dim if hasattr(module, "self_attn") else "unknown",
            }

        elif isinstance(module, torch.nn.MultiheadAttention):
            layer_info[name] = {
                "type": "MultiheadAttention",
                "embed_dim": module.embed_dim,
                "num_heads": module.num_heads,
            }

    # Print major layers
    logger.info(f"Found {len(layer_info)} analyzable layers\n")
    for name, info in list(layer_info.items())[:10]:
        logger.info(f"  {name}: {info}")

    return layer_info


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Qwen3-ForcedAligner internals"
    )
    parser.add_argument(
        "--lang",
        choices=["en", "zh", "yue"],
        default="en",
        help="Language to analyze",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile inference performance",
    )

    args = parser.parse_args()

    logger.info(f"\n{'='*70}")
    logger.info(f"QWEN3-FORCEDALIGNER INFERENCE ANALYSIS")
    logger.info(f"{'='*70}\n")

    try:
        from src.alignment import QwenForcedAligner

        # Load model
        logger.info(f"Loading Qwen3-ForcedAligner for language: {args.lang}")
        aligner = QwenForcedAligner(device=args.device, language=args.lang)

        # Analyze architecture
        arch_info = analyze_model_architecture(aligner.model, args.lang)

        logger.info(f"\nModel Summary:")
        logger.info(f"  Type: {arch_info['model_type']}")
        logger.info(f"  Total params: {arch_info['total_parameters']:,}")
        logger.info(f"  Trainable params: {arch_info['trainable_parameters']:,}")
        logger.info(f"  Size on disk: {arch_info['size_mb']:.1f} MB")

        # Analyze layers
        layer_info = analyze_model_layers(aligner.model)

        # Profile inference (optional)
        profile_info = {}
        if args.profile:
            profile_info = profile_inference(aligner.model, args.lang)

        # Save analysis
        output_file = Path(f"aligner_analysis_{args.lang}.json")
        analysis = {
            "language": args.lang,
            "architecture": arch_info,
            "num_layers": len(layer_info),
            "major_modules": arch_info.get("major_modules", []),
        }
        if profile_info:
            analysis["inference_profile"] = profile_info

        with open(output_file, "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        logger.info(f"\n✅ Analysis saved to: {output_file}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
