"""SAE Training loop with monitoring, checkpointing, and dead feature management.

Trains a Top-K Sparse Autoencoder on captured LLM activations using:
- AdamW optimizer with cosine learning rate schedule
- Mixed precision training (AMP)
- W&B logging for experiment tracking
- Periodic checkpointing
- Dead feature detection and resampling
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.sae.topk_sae import TopKSAE, SAEConfig

logger = logging.getLogger(__name__)


class SAETrainer:
    """Trainer for Top-K Sparse Autoencoder.

    Parameters
    ----------
    sae : TopKSAE
        The SAE model to train
    train_loader : iterable
        DataLoader yielding dict[int, Tensor] batches of activations
    val_loader : iterable, optional
        Validation DataLoader
    learning_rate : float
        Initial learning rate
    weight_decay : float
        L2 regularization coefficient
    max_steps : int
        Total training steps
    warmup_steps : int
        Steps to linearly increase LR from 0
    log_interval : int
        Log metrics every N steps
    checkpoint_interval : int
        Save checkpoint every N steps
    dead_feature_threshold : int
        Count a feature as dead if activated < this many times
    dead_feature_resampling : bool
        Resample dead features during training
    device : str
        Device to train on
    use_amp : bool
        Use automatic mixed precision
    wandb_project : str, optional
        W&B project name for logging
    wandb_run_name : str, optional
        W&B run name
    """

    def __init__(
        self,
        sae: TopKSAE,
        train_loader: iterable,
        val_loader: Optional[iterable] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        max_steps: int = 500000,
        warmup_steps: int = 1000,
        log_interval: int = 100,
        checkpoint_interval: int = 5000,
        dead_feature_threshold: int = 100,
        dead_feature_resampling: bool = True,
        device: str = "cuda",
        use_amp: bool = True,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        self.sae = sae.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.dead_feature_threshold = dead_feature_threshold
        self.dead_feature_resampling = dead_feature_resampling
        self.device = device
        self.use_amp = use_amp

        # Optimizer
        self.optimizer = optim.AdamW(sae.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # LR scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_steps, eta_min=1e-5)

        # Warmup: linear increase from 0
        def warmup_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0

        # Combine warmup + cosine
        from torch.optim.lr_scheduler import LambdaLR

        self.warmup_scheduler = LambdaLR(self.optimizer, warmup_lambda)

        # AMP
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # Logging
        self.wandb_enabled = wandb_project is not None
        if self.wandb_enabled:
            try:
                import wandb

                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name or "sae-training",
                    config={
                        "d_in": sae.config.d_in,
                        "d_sae": sae.config.d_sae,
                        "k": sae.config.k,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "max_steps": max_steps,
                    },
                )
                logger.info(f"W&B logging enabled: {wandb_run_name}")
            except ImportError:
                logger.warning("W&B not installed, skipping logging")
                self.wandb_enabled = False

        # Metrics
        self.metrics_history = {
            "step": [],
            "loss": [],
            "explained_variance": [],
            "sparsity": [],
            "dead_features": [],
            "learning_rate": [],
        }

    def train(self, output_dir: str | Path = "checkpoints/sae") -> dict:
        """Train the SAE.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save checkpoints and final model

        Returns
        -------
        metrics : dict
            Final training metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Starting SAE training: max_steps={self.max_steps}, "
            f"device={self.device}, use_amp={self.use_amp}"
        )

        step = 0
        epoch = 0

        while step < self.max_steps:
            epoch += 1

            for batch in self.train_loader:
                if step >= self.max_steps:
                    break

                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # For now, use the first layer's activations
                # (in production, could extend to multi-layer training)
                first_layer = list(batch.keys())[0]
                x = batch[first_layer]

                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        loss, metrics = self.sae(x)
                    # Backward pass
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.sae.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss, metrics = self.sae(x)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.sae.parameters(), max_norm=1.0)
                    self.optimizer.step()

                # Update learning rate
                self.warmup_scheduler.step()

                # Logging
                if step % self.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    self._log_metrics(step, loss.item(), metrics, lr)

                # Dead feature resampling
                if (
                    self.dead_feature_resampling
                    and step > 0
                    and step % 1000 == 0
                    and metrics["dead_features"] > 0
                ):
                    logger.info(f"Resampling {metrics['dead_features']} dead features")
                    with torch.no_grad():
                        z_sparse, _ = self.sae.encode(x)
                        x_residual = x - self.sae.decode(z_sparse)
                    self.sae.resample_dead_features(z_sparse, x_residual)

                # Checkpointing
                if step > 0 and step % self.checkpoint_interval == 0:
                    ckpt_path = output_dir / f"checkpoint_step_{step:06d}.pt"
                    self._save_checkpoint(ckpt_path)
                    logger.info(f"Saved checkpoint to {ckpt_path}")

                # Validation (optional)
                if self.val_loader is not None and step % 5000 == 0:
                    val_loss = self._validate()
                    if self.wandb_enabled:
                        try:
                            import wandb

                            wandb.log({"val_loss": val_loss, "step": step})
                        except:
                            pass

                step += 1

        logger.info(f"Training complete after {step} steps")

        # Save final model
        final_path = output_dir / "final_model.pt"
        self._save_checkpoint(final_path)
        logger.info(f"Saved final model to {final_path}")

        # Return final metrics
        return {
            "final_step": step,
            "final_loss": loss.item(),
            "final_metrics": metrics,
        }

    def _log_metrics(self, step: int, loss: float, metrics: dict, lr: float):
        """Log metrics to console and W&B."""
        logger.info(
            f"[Step {step:06d}] "
            f"Loss: {loss:.6f}, "
            f"EV: {metrics['explained_variance']:.4f}, "
            f"Sparsity: {metrics['sparsity']:.4f}, "
            f"Dead: {metrics['dead_features']}, "
            f"LR: {lr:.2e}"
        )

        # Store in history
        self.metrics_history["step"].append(step)
        self.metrics_history["loss"].append(loss)
        self.metrics_history["explained_variance"].append(metrics["explained_variance"])
        self.metrics_history["sparsity"].append(metrics["sparsity"])
        self.metrics_history["dead_features"].append(metrics["dead_features"])
        self.metrics_history["learning_rate"].append(lr)

        # Log to W&B
        if self.wandb_enabled:
            try:
                import wandb

                wandb.log(
                    {
                        "loss": loss,
                        "explained_variance": metrics["explained_variance"],
                        "sparsity": metrics["sparsity"],
                        "dead_features": metrics["dead_features"],
                        "learning_rate": lr,
                        "step": step,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to log to W&B: {e}")

    def _validate(self) -> float:
        """Run validation and return mean loss."""
        self.sae.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                first_layer = list(batch.keys())[0]
                x = batch[first_layer]
                loss, _ = self.sae(x)
                total_loss += loss.item()
                num_batches += 1

        mean_loss = total_loss / max(num_batches, 1)
        logger.info(f"Validation loss: {mean_loss:.6f}")
        self.sae.train()
        return mean_loss

    def _save_checkpoint(self, path: str | Path):
        """Save model checkpoint."""
        path = Path(path)
        torch.save(
            {
                "step": self.metrics_history["step"][-1] if self.metrics_history["step"] else 0,
                "model_state_dict": self.sae.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.sae.config,
                "metrics_history": self.metrics_history,
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str = "cuda") -> TopKSAE:
        """Load a trained SAE from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        sae = TopKSAE(config).to(device)
        sae.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded SAE from {path}")
        return sae
