"""Wrapper to launch MSAE training in the `third_party/MSAE` submodule with sensible defaults.

This script constructs a command line calling the MSAE `train.py` inside the submodule.
It validates that the submodule exists and that data files are present before launching.

Example:
  python tools/run_msae_train.py --train data/tts_train.npy --val data/tts_val.npy --expansion 16 --topk 32 --out-dir results/msae_tts
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    parser.add_argument("--expansion", type=int, default=16)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=Path("results/msae_tts"))
    parser.add_argument("--msae-path", type=Path, default=Path("third_party/MSAE"))
    args = parser.parse_args()

    msae_path = args.msae_path
    if not msae_path.exists():
        print(
            f"MSAE submodule not found at {msae_path}. Did you initialize submodules?"
        )
        sys.exit(1)

    if not args.train.exists() or not args.val.exists():
        print(
            "Train/val files not found. Run tools/tts_precompute_activations.py first."
        )
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(msae_path / "train.py"),
        "-dt",
        str(args.train),
        "-ds",
        str(args.val),
        "-m",
        "MSAE_UW",
        "-a",
        f"TopKReLU_{args.topk}",
        "--expansion_factor",
        str(args.expansion),
        "--epochs",
        str(args.epochs),
        "--outdir",
        str(args.out_dir),
    ]

    print("Launching MSAE train with command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
