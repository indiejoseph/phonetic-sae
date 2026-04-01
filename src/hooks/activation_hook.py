"""Generic activation hook for capturing LLM internal activations.

Attaches PyTorch forward hooks to specified modules and collects
per-token activation vectors with metadata. Supports both MLP
post-activation and residual stream hook points.

Usage:
    hook = ActivationHook(model, layer_indices=[1, 2, 3, 4, 5, 6, 7])
    hook.attach("mlp")       # or "residual"
    output = model(input_ids)
    activations = hook.collect()  # dict[int, Tensor]
    hook.detach()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

HookPoint = Literal["mlp", "residual"]


@dataclass
class CapturedActivation:
    """Container for a single layer's captured activations."""

    layer_idx: int
    hook_point: HookPoint
    # (batch, seq_len, d_model) — detached, moved to CPU
    tensor: torch.Tensor
    # Optional metadata
    batch_id: int | None = None


class ActivationHook:
    """Attach forward hooks to Transformer layers and capture activations.

    Parameters
    ----------
    model : nn.Module
        The LLM or TTS model containing Transformer layers.
    layer_indices : list[int]
        Which layer indices to hook (0-indexed).
    layer_accessor : callable, optional
        Function that takes (model, layer_idx) and returns the nn.Module
        for that layer. Defaults to ``model.layers[i]``.
    device : str
        Device to move captured activations to. Default "cpu".
    dtype : torch.dtype | None
        If set, cast captured activations to this dtype (e.g. torch.float16).
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: list[int],
        layer_accessor: callable | None = None,
        device: str = "cpu",
        dtype: torch.dtype | None = torch.float16,
    ):
        self.model = model
        self.layer_indices = sorted(layer_indices)
        self.layer_accessor = layer_accessor or self._default_layer_accessor
        self.device = device
        self.dtype = dtype

        # Storage
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._buffers: dict[int, list[torch.Tensor]] = {
            i: [] for i in self.layer_indices
        }
        self._hook_point: HookPoint | None = None
        self._batch_counter = 0

    @staticmethod
    def _default_layer_accessor(model: nn.Module, layer_idx: int) -> nn.Module:
        """Try common layer accessor patterns."""
        # Qwen3-TTS Talker: model.talker.layers[i]
        if hasattr(model, "talker") and hasattr(model.talker, "layers"):
            return model.talker.layers[layer_idx]
        # CosyVoice2 LLM: model.llm.layers[i]
        if hasattr(model, "llm") and hasattr(model.llm, "layers"):
            return model.llm.layers[layer_idx]
        # Generic: model.model.layers[i] (HuggingFace style)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer_idx]
        # Direct: model.layers[i]
        if hasattr(model, "layers"):
            return model.layers[layer_idx]
        raise AttributeError(
            f"Cannot find layer {layer_idx} on model {type(model).__name__}. "
            "Provide a custom layer_accessor function."
        )

    def _get_hook_target(self, layer: nn.Module, hook_point: HookPoint) -> nn.Module:
        """Return the specific submodule to hook within a Transformer layer."""
        if hook_point == "mlp":
            # Try common MLP attribute names
            for attr in ("mlp", "feed_forward", "ffn"):
                if hasattr(layer, attr):
                    return getattr(layer, attr)
            raise AttributeError(
                f"Cannot find MLP submodule on {type(layer).__name__}. "
                "Expected one of: mlp, feed_forward, ffn."
            )
        elif hook_point == "residual":
            # Hook the entire layer to capture the residual stream output
            return layer
        else:
            raise ValueError(
                f"Unknown hook_point: {hook_point!r}. Use 'mlp' or 'residual'."
            )

    def _make_hook_fn(self, layer_idx: int):
        """Create a forward hook function for a specific layer."""

        def hook_fn(module: nn.Module, input: tuple, output):
            # output may be a tuple (e.g., (hidden_states, ...)) or a tensor
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output

            # Detach, optionally cast, move to storage device
            captured = tensor.detach()
            if self.dtype is not None:
                captured = captured.to(dtype=self.dtype)
            captured = captured.to(self.device)

            self._buffers[layer_idx].append(captured)

        return hook_fn

    def attach(self, hook_point: HookPoint = "mlp") -> ActivationHook:
        """Attach hooks to all target layers.

        Parameters
        ----------
        hook_point : "mlp" or "residual"
            Where within each layer to capture activations.

        Returns
        -------
        self
            For method chaining.
        """
        self.detach()  # Remove any existing hooks
        self._hook_point = hook_point

        for layer_idx in self.layer_indices:
            layer = self.layer_accessor(self.model, layer_idx)
            target = self._get_hook_target(layer, hook_point)
            handle = target.register_forward_hook(self._make_hook_fn(layer_idx))
            self._handles.append(handle)
            logger.debug(f"Attached {hook_point} hook to layer {layer_idx}")

        logger.info(
            f"Attached {len(self._handles)} hooks ({hook_point}) "
            f"to layers {self.layer_indices}"
        )
        return self

    def detach(self) -> None:
        """Remove all hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._hook_point = None

    def collect(self, clear: bool = True) -> dict[int, torch.Tensor]:
        """Collect all captured activations.

        Parameters
        ----------
        clear : bool
            If True, clear buffers after collecting.

        Returns
        -------
        dict[int, Tensor]
            Mapping from layer_idx to concatenated activation tensor
            of shape (total_tokens, d_model).
        """
        result = {}
        for layer_idx, tensors in self._buffers.items():
            if tensors:
                # Concatenate along batch dimension, then flatten (batch, seq, d) → (N, d)
                stacked = torch.cat(tensors, dim=0)
                if stacked.dim() == 3:
                    # (batch, seq_len, d_model) → (batch * seq_len, d_model)
                    stacked = stacked.reshape(-1, stacked.shape[-1])
                result[layer_idx] = stacked
            else:
                result[layer_idx] = torch.empty(0)

        if clear:
            self.clear()

        self._batch_counter += 1
        return result

    def flush_to_disk(
        self, out_dir: str | Path, prefix: str = "activations", clear: bool = True
    ) -> dict[int, Path]:
        """Write current buffers to disk as .pt files and optionally clear buffers.

        Returns a mapping layer_idx -> Path written.
        """
        from pathlib import Path

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = {}
        for layer_idx, tensors in self._buffers.items():
            if not tensors:
                continue
            stacked = torch.cat(tensors, dim=0)
            if stacked.dim() == 3:
                stacked = stacked.reshape(-1, stacked.shape[-1])
            path = out_dir / f"{prefix}_layer{layer_idx}_batch{self._batch_counter}.pt"
            torch.save(stacked.cpu(), path)
            written[layer_idx] = path

        if clear:
            self.clear()

        self._batch_counter += 1
        return written

    def clear(self) -> None:
        """Clear all activation buffers without detaching hooks."""
        for layer_idx in self._buffers:
            self._buffers[layer_idx].clear()

    @property
    def num_captured(self) -> dict[int, int]:
        """Number of tensors captured per layer (before collect)."""
        return {k: len(v) for k, v in self._buffers.items()}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.detach()

    def __repr__(self) -> str:
        status = "attached" if self._handles else "detached"
        return (
            f"ActivationHook(layers={self.layer_indices}, "
            f"hook_point={self._hook_point!r}, status={status})"
        )
