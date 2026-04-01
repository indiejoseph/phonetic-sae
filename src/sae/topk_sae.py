"""Top-K Sparse Autoencoder for phonetic feature discovery.

Implements a sparse autoencoder with Top-K activation enforcing exact sparsity.
Architecture:
    z = TopK(W_enc @ (x - b_dec) + b_enc, k=K)
    x_hat = W_dec @ z + b_dec
    Loss = ||x - x_hat||^2

Key features:
    - TopK activation (exactly K features active per sample)
    - Weight normalization for better interpretability
    - Dead feature detection and monitoring
    - Explained variance computation
    - Full support for encoder/decoder access (for interventions)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SAEConfig:
    """Configuration for TopK SAE."""

    d_in: int  # Input dimension (d_model)
    d_sae: int  # SAE latent dimension (usually 16x or 32x d_in)
    k: int = 32  # Number of active features (sparsity)
    dtype: torch.dtype = torch.float32
    device: str = "cpu"


class TopKSAE(nn.Module):
    """Top-K Sparse Autoencoder.

    Parameters
    ----------
    config : SAEConfig
        Configuration dict or object with d_in, d_sae, k.
    """

    def __init__(self, config: SAEConfig | dict):
        super().__init__()
        if isinstance(config, dict):
            config = SAEConfig(**config)
        self.config = config

        # Encoder: R^d_in → R^d_sae
        self.encoder = nn.Linear(config.d_in, config.d_sae, bias=True, dtype=config.dtype)
        # Decoder: R^d_sae → R^d_in
        self.decoder = nn.Linear(config.d_sae, config.d_in, bias=True, dtype=config.dtype)

        # Optional: separate pre-bias (centering input before encoding)
        self.register_buffer("pre_bias", torch.zeros(config.d_in, dtype=config.dtype))

        self.to(config.device)
        self._initialize_weights()
        self._dead_features_count = 0

    def _initialize_weights(self):
        """Initialize encoder and decoder weights."""
        # Encoder: small random initialization
        nn.init.kaiming_normal_(self.encoder.weight, nonlinearity="linear")
        nn.init.zeros_(self.encoder.bias)

        # Decoder: small random, then normalize rows to unit norm
        nn.init.kaiming_normal_(self.decoder.weight, nonlinearity="linear")
        nn.init.zeros_(self.decoder.bias)

        # Normalize decoder rows to unit norm (standard in SAE literature)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=1, p=2)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to sparse latent representation.

        Parameters
        ----------
        x : Tensor
            Input of shape (batch, d_in) or (N, d_in)

        Returns
        -------
        z_sparse : Tensor
            Sparse activation (batch, d_sae) with exactly k non-zero entries per sample
        z_full : Tensor
            Full latent (batch, d_sae) before thresholding (for analysis)
        """
        # Center input
        x_centered = x - self.pre_bias
        # Encode
        z_full = self.encoder(x_centered)
        # Top-K sparsity: keep only top k activations, zero out rest
        z_sparse = self._topk_activation(z_full)
        return z_sparse, z_full

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse latent back to input space.

        Parameters
        ----------
        z : Tensor
            Sparse latent (batch, d_sae)

        Returns
        -------
        x_hat : Tensor
            Reconstruction (batch, d_in)
        """
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Forward pass: encode → sparse → decode → compute loss.

        Parameters
        ----------
        x : Tensor
            Input (batch, d_in)

        Returns
        -------
        loss : Tensor
            Reconstruction loss (scalar)
        metrics : dict
            Diagnostics: reconstruction_mse, sparsity, dead_features, etc.
        """
        z_sparse, z_full = self.encode(x)
        x_hat = self.decode(z_sparse)
        mse_loss = F.mse_loss(x, x_hat)

        # Compute metrics
        with torch.no_grad():
            metrics = self._compute_metrics(x, x_hat, z_sparse, z_full)

        return mse_loss, metrics

    @staticmethod
    def _topk_activation(z: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """Apply Top-K sparsity: keep only top k values, zero out rest.

        Parameters
        ----------
        z : Tensor
            Full latent (batch, d_sae)
        k : int, optional
            Number of top values to keep per sample. If None, uses self.config.k.

        Returns
        -------
        z_sparse : Tensor
            Sparse latent with exactly k active features per sample
        """
        # Get top-k values and indices
        topk_vals, topk_idx = torch.topk(z.abs(), k=min(k or 32, z.shape[-1]), dim=-1)
        # Create sparse tensor
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_idx, z.gather(-1, topk_idx))
        return z_sparse

    def _topk_activation(self, z: torch.Tensor) -> torch.Tensor:
        """Apply Top-K sparsity using config.k."""
        return self.__class__._topk_activation(z, k=self.config.k)

    def _compute_metrics(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        z_sparse: torch.Tensor,
        z_full: torch.Tensor,
    ) -> dict:
        """Compute diagnostic metrics."""
        batch_size = x.shape[0]

        # Reconstruction error
        mse = F.mse_loss(x, x_hat)

        # Explained variance (fraction of input variance captured by reconstruction)
        x_var = (x - x.mean(dim=0)).pow(2).mean()
        residual_var = (x - x_hat).pow(2).mean()
        explained_var = 1.0 - (residual_var / (x_var + 1e-8))

        # Sparsity: fraction of non-zero features
        num_active = (z_sparse != 0).sum(dim=-1).float().mean()
        sparsity = num_active / z_sparse.shape[-1]

        # Dead features: features that never activate
        num_batches_seen = getattr(self, "_batches_seen", 0)
        num_batches_seen += 1
        self._batches_seen = num_batches_seen

        feature_activity = (z_sparse != 0).float().mean(dim=0)
        dead_features = (feature_activity == 0).sum().item()
        dead_features_frac = dead_features / z_sparse.shape[-1]

        # Cosine similarity between encoder and decoder
        cos_sim = F.cosine_similarity(
            self.encoder.weight.data, self.decoder.weight.data.T, dim=1
        ).mean()

        return {
            "mse": mse.item(),
            "explained_variance": explained_var.item(),
            "sparsity": sparsity.item(),
            "num_active_features": num_active.item(),
            "dead_features": dead_features,
            "dead_features_frac": dead_features_frac,
            "encoder_decoder_cosine_sim": cos_sim.item(),
        }

    def get_dead_features(self, z_sparse: torch.Tensor, threshold: int = 0) -> list[int]:
        """Get indices of features that never (or rarely) activate.

        Parameters
        ----------
        z_sparse : Tensor
            Sparse latents from a batch or dataset (N, d_sae)
        threshold : int
            Count a feature as "dead" if it activates fewer than this many times

        Returns
        -------
        dead_indices : list[int]
            Indices of dead features
        """
        feature_counts = (z_sparse != 0).sum(dim=0)
        dead_indices = (feature_counts <= threshold).nonzero(as_tuple=True)[0].tolist()
        return dead_indices

    def resample_dead_features(
        self,
        z_sparse: torch.Tensor,
        x_residual: torch.Tensor,
        threshold: int = 0,
        num_samples: int = 100,
    ):
        """Resample dead features from high-loss examples.

        Parameters
        ----------
        z_sparse : Tensor
            Sparse latents (N, d_sae)
        x_residual : Tensor
            Reconstruction residual (N, d_in) = x - x_hat
        threshold : int
            Feature activation count below this is considered dead
        num_samples : int
            How many high-loss examples to use for resampling
        """
        dead_idx = self.get_dead_features(z_sparse, threshold=threshold)
        if not dead_idx:
            logger.info("No dead features to resample")
            return

        # Get examples with highest reconstruction loss
        residual_norms = x_residual.pow(2).sum(dim=1)
        top_loss_idx = torch.topk(residual_norms, k=min(num_samples, len(residual_norms)))[1]
        high_loss_examples = x_residual[top_loss_idx]  # (num_samples, d_in)

        # Resample each dead feature from high-loss examples
        with torch.no_grad():
            for feat_idx in dead_idx:
                # Randomly pick one of the high-loss examples
                sample_idx = torch.randint(0, high_loss_examples.shape[0], (1,)).item()
                new_weight = high_loss_examples[sample_idx]  # (d_in,)

                # Update decoder and encoder
                self.decoder.weight.data[feat_idx] = F.normalize(new_weight, p=2)
                self.encoder.bias.data[feat_idx] = 0.0

        logger.info(f"Resampled {len(dead_idx)} dead features")

    def state_dict_with_config(self) -> dict:
        """Return state dict including config (useful for saving)."""
        return {
            "config": self.config,
            "state_dict": self.state_dict(),
        }

    @classmethod
    def from_config_and_state(cls, config: SAEConfig, state_dict: dict) -> TopKSAE:
        """Load from saved config and state dict."""
        model = cls(config)
        model.load_state_dict(state_dict)
        return model

    def __repr__(self) -> str:
        return (
            f"TopKSAE(d_in={self.config.d_in}, d_sae={self.config.d_sae}, "
            f"k={self.config.k}, device={self.config.device})"
        )
