"""Demo script showing `ActivationPatcher` usage with a dummy model and SAE stub.

This file runs a tiny PyTorch model, attaches `ActivationPatcher`, and demonstrates
modifying latent features (zeroing top-k activations) during forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.intervention import ActivationPatcher


class DummySAE(nn.Module):
    def __init__(self, d_in=32, d_latent=64):
        super().__init__()
        self.enc = nn.Linear(d_in, d_latent)
        self.dec = nn.Linear(d_latent, d_in)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, d_in) -> z: (N, d_latent)
        return torch.relu(self.enc(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)


class TinyModel(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
                )
                for _ in range(6)
            ]
        )

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


def zero_topk(z: torch.Tensor, k: int = 4) -> torch.Tensor:
    # z: (N, d_latent)
    topk = torch.topk(z.abs(), k, dim=-1).indices
    mask = torch.ones_like(z)
    for i in range(z.shape[0]):
        mask[i, topk[i]] = 0.0
    return z * mask


def run_demo():
    device = torch.device("cpu")
    model = TinyModel(d_model=32).to(device)
    sae = DummySAE(d_in=32, d_latent=64).to(device)

    patcher = ActivationPatcher(
        sae, model, layer_idx=0, hook_point="mlp", device=str(device)
    )
    patcher.set_modify_fn(lambda z: zero_topk(z, k=4))
    patcher.attach()

    # Create dummy input: batch=2, seq_len=4, d_model=32
    x = torch.randn(2, 4, 32)
    out = model(x)
    print("Forward completed. Output shape:", out.shape)

    patcher.detach()


if __name__ == "__main__":
    run_demo()
