"""Streaming activation buffer for capturing and storing LLM activations.

Collects activation tensors from hooks, batches them, and flushes to disk
as FP16 .npy files. Designed for memory-efficient capture of 50M+ vectors.

Usage:
    buffer = ActivationBuffer(
        output_dir="data/activations/qwen3tts",
        layer_indices=[1, 2, 3, 4, 5, 6, 7],
        batch_size=512,
    )
    # Inside training loop:
    hook = ActivationHook(model, layer_indices=[...])
    hook.attach("mlp")
    output = model(input_ids)
    activations = hook.collect()  # dict[int, Tensor]
    buffer.add_batch(activations)
    # At the end:
    buffer.flush()
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ActivationBuffer:
    """Streaming buffer for collecting and saving activations to disk.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save activation files
    layer_indices : list[int]
        Which layers are being captured
    batch_size : int
        How many activation batches to accumulate before flushing to disk
    dtype : str
        Data type to save ("float32", "float16", "int8")
    """

    def __init__(
        self,
        output_dir: str | Path,
        layer_indices: list[int],
        batch_size: int = 512,
        dtype: str = "float16",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.layer_indices = sorted(layer_indices)
        self.batch_size = batch_size
        self.dtype = dtype

        # Buffers: accumulate activation tensors per layer
        self._buffers = {i: [] for i in self.layer_indices}
        self._total_saved = {i: 0 for i in self.layer_indices}
        self._batch_count = 0

        # Map dtype string to numpy dtype
        self._dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
            "int8": np.int8,
        }
        if dtype not in self._dtype_map:
            raise ValueError(f"Unknown dtype: {dtype}. Use float32, float16, or int8.")

    def add_batch(self, activations: dict[int, torch.Tensor]) -> None:
        """Add a batch of activations (from hook.collect()).

        Parameters
        ----------
        activations : dict[int, Tensor]
            Mapping from layer_idx to activation tensor (N, d_model).
            Typically from ActivationHook.collect().
        """
        for layer_idx in self.layer_indices:
            if layer_idx in activations:
                tensor = activations[layer_idx]
                # Ensure it's a CPU tensor
                if tensor.device.type != "cpu":
                    tensor = tensor.cpu()
                self._buffers[layer_idx].append(tensor.numpy())

        self._batch_count += 1

        # Auto-flush if buffer is large enough
        if self._batch_count >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Flush accumulated activations to disk as .npy files."""
        if self._batch_count == 0:
            logger.info("Buffer is empty, nothing to flush")
            return

        for layer_idx in self.layer_indices:
            if self._buffers[layer_idx]:
                # Concatenate all batches
                stacked = np.concatenate(self._buffers[layer_idx], axis=0)

                # Convert dtype if needed
                np_dtype = self._dtype_map[self.dtype]
                if stacked.dtype != np_dtype:
                    stacked = stacked.astype(np_dtype)

                # Save
                file_path = self.output_dir / f"layer_{layer_idx:02d}_batch_{self._total_saved[layer_idx]:06d}.npy"
                np.save(file_path, stacked)

                self._total_saved[layer_idx] += stacked.shape[0]
                logger.debug(
                    f"Saved {stacked.shape[0]} activations to {file_path.name} "
                    f"(total: {self._total_saved[layer_idx]})"
                )

                # Clear buffer
                self._buffers[layer_idx].clear()

        self._batch_count = 0
        logger.info(
            f"Flushed activations. Total saved per layer: {self._total_saved}"
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.flush()

    @property
    def total_saved(self) -> dict[int, int]:
        """Total activation vectors saved per layer."""
        return self._total_saved.copy()

    def __repr__(self) -> str:
        return (
            f"ActivationBuffer(output_dir={self.output_dir}, "
            f"layers={self.layer_indices}, "
            f"batch_size={self.batch_size}, "
            f"dtype={self.dtype})"
        )


def load_activations_from_dir(
    activation_dir: str | Path,
    layer_indices: list[int] | None = None,
    limit: int | None = None,
    dtype: str = "float16",
) -> dict[int, np.ndarray]:
    """Load all saved activations from directory.

    Parameters
    ----------
    activation_dir : str or Path
        Directory containing .npy activation files
    layer_indices : list[int], optional
        Which layers to load. If None, loads all found.
    limit : int, optional
        Maximum number of vectors to load per layer
    dtype : str
        Expected data type of saved files

    Returns
    -------
    activations : dict[int, ndarray]
        Mapping from layer_idx to (N, d_model) array
    """
    activation_dir = Path(activation_dir)
    activations = {}

    # Find all .npy files grouped by layer
    files_by_layer = {}
    for npy_file in sorted(activation_dir.glob("layer_*.npy")):
        # Parse layer index from filename: layer_01_batch_000000.npy
        parts = npy_file.stem.split("_")
        if len(parts) >= 2:
            try:
                layer_idx = int(parts[1])
                if layer_indices is None or layer_idx in layer_indices:
                    if layer_idx not in files_by_layer:
                        files_by_layer[layer_idx] = []
                    files_by_layer[layer_idx].append(npy_file)
            except ValueError:
                continue

    # Load and concatenate per layer
    for layer_idx in sorted(files_by_layer.keys()):
        arrays = []
        for npy_file in files_by_layer[layer_idx]:
            arr = np.load(npy_file)
            arrays.append(arr)

        stacked = np.concatenate(arrays, axis=0)

        # Optionally limit
        if limit is not None and stacked.shape[0] > limit:
            stacked = stacked[:limit]

        activations[layer_idx] = stacked
        logger.info(f"Loaded layer {layer_idx}: {stacked.shape}")

    return activations
