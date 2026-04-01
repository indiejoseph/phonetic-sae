#!/usr/bin/env python3
"""SAE Training script: train a Top-K Sparse Autoencoder on captured activations.

Usage:
    # Train Qwen3-TTS SAE on activations
    python scripts/train_sae.py \
        --config configs/sae_qwen3tts.yaml \
        --activation-dir data/activations/qwen3tts \
        --output checkpoints/sae_qwen3tts

    # Train with custom settings
    python scripts/train_sae.py \
        --config configs/sae_qwen3tts.yaml \
        --activation-dir data/activations/qwen3tts \
        --output checkpoints/sae_qwen3tts \
        --batch-size 4096 \
        --max-steps 100000 \
        --wandb-project phonetic-sae \
        --device cuda:0
"""

import argparse
import logging
from pathlib import Path

import torch
import yaml

from src.sae.topk_sae import TopKSAE, SAEConfig
from src.sae.trainer import SAETrainer
from src.data.shuffled_activation_buffer import ShuffledActivationBuffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder on TTS activations")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to SAE config YAML file",
    )
    parser.add_argument(
        "--activation-dir",
        type=Path,
        required=True,
        help="Directory containing saved activations (.npy files)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/sae"),
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max training steps (overrides config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project name for logging",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="W&B run name",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation",
    )
    args = parser.parse_args()

    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Override config with command-line args
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.max_steps:
        config["max_steps"] = args.max_steps
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.wandb_project:
        config["wandb_project"] = args.wandb_project
    if args.wandb_run_name:
        config["wandb_run_name"] = args.wandb_run_name

    logger.info(f"SAE Config: d_in={config['d_in']}, d_sae={config['d_sae']}, k=32")
    logger.info(f"Training Config: batch_size={config['batch_size']}, max_steps={config['max_steps']}")

    # Load activations
    logger.info(f"Loading activations from {args.activation_dir}")
    train_loader = ShuffledActivationBuffer(
        activation_dir=args.activation_dir,
        batch_size=config["batch_size"],
        shuffle=True,
        device=args.device,
        dtype=torch.float32,
    )

    logger.info(f"Loaded {train_loader.total_vectors} activation vectors")
    logger.info(f"Num batches per epoch: {len(train_loader)}")

    # Create SAE
    sae_config = SAEConfig(
        d_in=config["d_in"],
        d_sae=config["d_sae"],
        k=32,
        dtype=torch.float32,
        device=args.device,
    )
    sae = TopKSAE(sae_config)
    logger.info(f"Created SAE: {sae}")

    # Create trainer
    trainer = SAETrainer(
        sae=sae,
        train_loader=train_loader,
        learning_rate=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 0.01),
        max_steps=config["max_steps"],
        warmup_steps=config.get("warmup_steps", 1000),
        log_interval=config.get("log_interval", 100),
        checkpoint_interval=config.get("checkpoint_interval", 5000),
        dead_feature_threshold=config.get("dead_feature_threshold", 100),
        dead_feature_resampling=config.get("dead_feature_resampling", True),
        device=args.device,
        use_amp=config.get("mixed_precision", True),
        wandb_project=args.wandb_project or config.get("wandb_project"),
        wandb_run_name=args.wandb_run_name or config.get("wandb_run_name"),
    )

    # Train
    logger.info("Starting training...")
    results = trainer.train(output_dir=args.output)

    logger.info(f"Training complete!")
    logger.info(f"Final step: {results['final_step']}")
    logger.info(f"Final loss: {results['final_loss']:.6f}")
    logger.info(f"Checkpoints saved to: {args.output}")


if __name__ == "__main__":
    main()
