"""Shuffled activation buffer for efficient SAE training.

Loads activation tensors from disk in random order and yields
mini-batches with inter- and intra-file shuffling to prevent
overfitting to sentence-level structure.

Designed for memory-efficient training on 50M+ activation vectors.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ShuffledActivationBuffer:
    """Shuffled loader for activation tensors from disk.

    Parameters
    ----------
    activation_dir : str or Path
        Directory containing saved activation .npy files (layer_XX_batch_YYYYYY.npy)
    layer_indices : list[int]
        Which layers to load. If None, loads all found.
    batch_size : int
        Number of activation vectors per mini-batch
    shuffle : bool
        Whether to shuffle within and across files
    seed : int
        Random seed for reproducibility
    device : str
        Device to load tensors to ("cpu", "cuda", etc.)
    dtype : torch.dtype
        Data type for tensors (torch.float32, torch.float16, etc.)
    """

    def __init__(
        self,
        activation_dir: str | Path,
        layer_indices: list[int] | None = None,
        batch_size: int = 8192,
        shuffle: bool = True,
        seed: int = 42,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.activation_dir = Path(activation_dir)
        self.layer_indices = layer_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.device = device
        self.dtype = dtype

        # Find and organize files by layer
        self._files_by_layer = self._organize_files()
        self._validate_layer_coverage()

        # Load all activations (or set up iterator if too large)
        self._activations = {}
        self._load_activations()

        self._current_idx = 0
        self._num_batches = 0

    def _organize_files(self) -> dict[int, list[Path]]:
        """Find all .npy files and organize by layer."""
        files_by_layer = {}

        for npy_file in sorted(self.activation_dir.glob("layer_*.npy")):
            # Parse filename: layer_01_batch_000000.npy
            try:
                parts = npy_file.stem.split("_")
                if len(parts) >= 2 and parts[0] == "layer":
                    layer_idx = int(parts[1])
                    if (
                        self.layer_indices is None
                        or layer_idx in self.layer_indices
                    ):
                        if layer_idx not in files_by_layer:
                            files_by_layer[layer_idx] = []
                        files_by_layer[layer_idx].append(npy_file)
            except (ValueError, IndexError):
                logger.warning(f"Could not parse layer from {npy_file}")
                continue

        logger.info(
            f"Found {sum(len(v) for v in files_by_layer.values())} "
            f"activation files for layers {sorted(files_by_layer.keys())}"
        )
        return files_by_layer

    def _validate_layer_coverage(self):
        """Ensure all requested layers have files."""
        if self.layer_indices is None:
            return

        missing = set(self.layer_indices) - set(self._files_by_layer.keys())
        if missing:
            logger.warning(f"No files found for layers: {missing}")

    def _load_activations(self):
        """Load all activation files into memory."""
        for layer_idx, file_list in sorted(self._files_by_layer.items()):
            arrays = []
            total_size = 0

            for npy_file in file_list:
                arr = np.load(npy_file, allow_pickle=False)
                arrays.append(arr)
                total_size += arr.shape[0]

            # Concatenate all files for this layer
            stacked = np.concatenate(arrays, axis=0)

            # Convert to tensor and move to device
            tensor = torch.from_numpy(stacked).to(
                device=self.device, dtype=self.dtype
            )

            self._activations[layer_idx] = tensor
            logger.info(
                f"Loaded layer {layer_idx}: {total_size} vectors, "
                f"shape {tensor.shape}, dtype {tensor.dtype}"
            )

        # Shuffle indices for sampling
        if self.shuffle:
            random.seed(self.seed)
            self._shuffle_indices()

    def _shuffle_indices(self):
        """Shuffle activation ordering within each layer."""
        for layer_idx in self._activations:
            num_vectors = self._activations[layer_idx].shape[0]
            indices = list(range(num_vectors))
            random.shuffle(indices)
            # Reorder tensor along batch dimension
            self._activations[layer_idx] = self._activations[layer_idx][indices]

    def __iter__(self) -> Iterator[dict[int, torch.Tensor]]:
        """Iterate over mini-batches of activations."""
        if not self._activations:
            raise RuntimeError("No activations loaded")

        # Get length from first layer
        first_layer = list(self._activations.keys())[0]
        num_samples = self._activations[first_layer].shape[0]

        self._current_idx = 0
        self._num_batches = 0

        while self._current_idx < num_samples:
            batch_end = min(self._current_idx + self.batch_size, num_samples)
            batch = {}

            for layer_idx, tensor in self._activations.items():
                batch[layer_idx] = tensor[self._current_idx : batch_end]

            self._current_idx = batch_end
            self._num_batches += 1

            yield batch

    def __len__(self) -> int:
        """Total number of mini-batches."""
        if not self._activations:
            return 0
        first_layer = list(self._activations.keys())[0]
        num_samples = self._activations[first_layer].shape[0]
        return (num_samples + self.batch_size - 1) // self.batch_size

    def num_samples(self) -> dict[int, int]:
        """Total number of activation vectors per layer."""
        return {k: v.shape[0] for k, v in self._activations.items()}

    @property
    def total_vectors(self) -> int:
        """Total activation vectors across all layers."""
        if not self._activations:
            return 0
        return list(self._activations.values())[0].shape[0]

    def __repr__(self) -> str:
        num_vectors = self.total_vectors
        num_layers = len(self._activations)
        return (
            f"ShuffledActivationBuffer("
            f"activation_dir={self.activation_dir}, "
            f"layers={num_layers}, "
            f"vectors={num_vectors}, "
            f"batch_size={self.batch_size})"
        )
