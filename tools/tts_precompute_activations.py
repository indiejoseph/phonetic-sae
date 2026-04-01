"""Convert captured activation tensors into concatenated .npy datasets for MSAE training.

Usage:
    python tools/tts_precompute_activations.py --input-dir data/activations/ --out-train data/tts_train.npy --out-val data/tts_val.npy --val-split 0.05

The script expects either .pt or .npy files in `--input-dir`. It will load them, optionally convert to float16,
shuffle rows, and write out two .npy files suitable for `SAEDataset` in the MSAE repo.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random

import numpy as np
import torch


def load_tensor(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix in (".pt", ".pth"):
        t = torch.load(path, map_location="cpu")
        if isinstance(t, dict) and "activations" in t:
            t = t["activations"]
        if isinstance(t, torch.Tensor):
            return t.numpy()
        # try to convert
        return np.array(t)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def collect_rows(input_dir: Path, max_rows: int | None = None) -> np.ndarray:
    files = sorted(
        [p for p in input_dir.iterdir() if p.suffix in (".npy", ".pt", ".pth")]
    )
    if not files:
        raise FileNotFoundError(f"No activation files found in {input_dir}")

    rows = []
    total = 0
    for p in files:
        arr = load_tensor(p)
        if arr.ndim == 3:
            # (batch, seq_len, d) -> flatten to (N, d)
            arr = arr.reshape(-1, arr.shape[-1])
        rows.append(arr)
        total += arr.shape[0]
        if max_rows and total >= max_rows:
            break

    all_rows = np.concatenate(rows, axis=0)
    if max_rows and all_rows.shape[0] > max_rows:
        indices = np.random.choice(all_rows.shape[0], size=max_rows, replace=False)
        all_rows = all_rows[indices]
    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--out-train", required=True, type=Path)
    parser.add_argument("--out-val", required=True, type=Path)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    input_dir = args.input_dir
    input_dir = Path(input_dir)
    all_rows = collect_rows(input_dir, max_rows=args.max_rows)
    print(f"Collected {all_rows.shape[0]} vectors with dim {all_rows.shape[1]}")

    # Shuffle
    rng = np.random.default_rng()
    rng.shuffle(all_rows)

    # Split
    n_val = int(all_rows.shape[0] * args.val_split)
    val = all_rows[:n_val]
    train = all_rows[n_val:]

    # Cast
    dtype = np.float16 if args.dtype == "float16" else np.float32
    train = train.astype(dtype)
    val = val.astype(dtype)

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    args.out_val.parent.mkdir(parents=True, exist_ok=True)

    np.save(args.out_train, train)
    np.save(args.out_val, val)

    print(
        f"Wrote train: {args.out_train} ({train.nbytes/1e9:.2f} GB), val: {args.out_val}"
    )


if __name__ == "__main__":
    main()
