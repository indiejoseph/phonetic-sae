"""Activation patching scaffold: encode -> modify -> decode -> replace activation in forward pass.

This provides a minimal `ActivationPatcher` class that can load an SAE-like model
(expects `encode` and `decode` methods) and attach a forward hook to a target
layer to perform runtime interventions.

Usage:
    patcher = ActivationPatcher(sae, model, layer_idx=3, hook_point='mlp')
    patcher.attach()
    # inside SAE modify_fn will be applied to latent vectors z
    patcher.detach()
"""

from __future__ import annotations

from typing import Callable, Literal
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
HookPoint = Literal["mlp", "residual"]


class ActivationPatcher:
    def __init__(
        self,
        sae_model,
        model: nn.Module,
        layer_idx: int,
        layer_accessor: Callable | None = None,
        hook_point: HookPoint = "mlp",
        device: str = "cpu",
    ):
        """s a e_model: object with `encode(x)` and `decode(z)` methods.
        modify_fn: callable(z: Tensor) -> Tensor will be applied in latent space.
        """
        self.sae = sae_model
        self.model = model
        self.layer_idx = layer_idx
        self.layer_accessor = layer_accessor or self._default_layer_accessor
        self.hook_point = hook_point
        self.device = device

        self._handle = None
        self.modify_fn = None  # to be set by user

    @staticmethod
    def _default_layer_accessor(model: nn.Module, layer_idx: int) -> nn.Module:
        if hasattr(model, "talker") and hasattr(model.talker, "layers"):
            return model.talker.layers[layer_idx]
        if hasattr(model, "llm") and hasattr(model.llm, "layers"):
            return model.llm.layers[layer_idx]
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer_idx]
        if hasattr(model, "layers"):
            return model.layers[layer_idx]
        raise AttributeError("Cannot find layer; provide a custom layer_accessor")

    def _get_hook_target(self, layer: nn.Module):
        if self.hook_point == "mlp":
            for attr in ("mlp", "feed_forward", "ffn"):
                if hasattr(layer, attr):
                    return getattr(layer, attr)
            raise AttributeError("MLP submodule not found on layer")
        return layer

    def _hook_fn(self, module: nn.Module, input: tuple, output):
        # Extract tensor
        tensor = output[0] if isinstance(output, tuple) else output

        # Expect shape (batch, seq_len, d_model)
        device = tensor.device
        x = tensor.detach()

        # Merge batch and seq dims
        orig_shape = x.shape
        flat = x.reshape(-1, x.shape[-1]).to(self.device)

        # Encode -> modify -> decode
        with torch.no_grad():
            z = self.sae.encode(flat)
            if self.modify_fn is not None:
                z = self.modify_fn(z)
            x_hat = self.sae.decode(z)

        x_hat = x_hat.reshape(orig_shape).to(device)

        # Need to return the modified output; most PyTorch hooks don't support
        # replacing outputs directly via forward_hook return value, so we mutate in-place
        try:
            if isinstance(output, tuple):
                out_list = list(output)
                out_list[0] = x_hat
                return tuple(out_list)
            else:
                return x_hat
        except Exception:
            # If mutation fails, try in-place copy
            tensor.copy_(x_hat)
            return None

    def attach(self):
        layer = self.layer_accessor(self.model, self.layer_idx)
        target = self._get_hook_target(layer)
        self._handle = target.register_forward_hook(self._hook_fn)
        logger.info(f"Attached patcher to layer {self.layer_idx} ({self.hook_point})")

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
            logger.info("Detached patcher")

    def set_modify_fn(self, fn: Callable):
        """Set function applied in latent space: fn(z: Tensor) -> Tensor"""
        self.modify_fn = fn

    def save_sae(self, path: Path):
        torch.save(self.sae.state_dict(), path)
