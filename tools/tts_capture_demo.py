"""Demo: capture activations from a TTS model using `ActivationHook`.

This is a scaffold. Supply a Python module that exposes a `load_model()`
function returning `(model, tokenizer)` or `model` alone. The script attaches
hooks, runs a few example sentences, and writes activation files to `--out-dir`.

Example:
  python tools/tts_capture_demo.py --model-module my_tts_loader:load_model --sentences examples/sentences.txt --out-dir data/activations --layers 3 4 5 --hook mlp

Notes:
- The loader must return a model object usable for `.generate()` or `.forward()`.
- This script is intentionally defensive and prints instructions where manual
  wiring is required for your specific model.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys
from typing import Optional

import torch

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.hooks.activation_hook import ActivationHook


def load_user_model(module_spec: str):
    """Import user module and call load_model()

    module_spec: 'module:callable'
    """
    if ":" not in module_spec:
        raise ValueError("--model-module must be MODULE:callable")
    mod_name, func_name = module_spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    if not hasattr(mod, func_name):
        raise AttributeError(f"Module {mod_name} has no attribute {func_name}")
    fn = getattr(mod, func_name)
    return fn()


def _save_activations(acts: dict[int, torch.Tensor], out_dir: Path, batch_id: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer_idx, tensor in acts.items():
        path = out_dir / f"layer_{layer_idx}_batch_{batch_id}.pt"
        torch.save(tensor.cpu(), path)
    print(f"Saved activations for batch {batch_id} to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-module",
        required=True,
        help="MODULE:callable returning model or (model, tokenizer)",
    )
    parser.add_argument(
        "--sentences", required=True, help="Text file with one sentence per line"
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[3], help="Layer indices to hook"
    )
    parser.add_argument("--hook", choices=["mlp", "residual"], default="mlp")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    print("Loading user model via:", args.model_module)
    model_return = load_user_model(args.model_module)
    if isinstance(model_return, tuple):
        model, tokenizer = model_return
    else:
        model = model_return
        tokenizer = None

    model.to(args.device)
    model.eval()

    hook = ActivationHook(
        model, layer_indices=args.layers, device="cpu", dtype=torch.float16
    )
    hook.attach(hook_point=args.hook)

    with open(args.sentences, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    batch = []
    batch_id = 0
    for i, sent in enumerate(lines):
        batch.append(sent)
        if len(batch) >= args.batch_size or i == len(lines) - 1:
            # Run inference. This part is model-specific.
            try:
                if tokenizer is not None and hasattr(tokenizer, "__call__"):
                    tokens = tokenizer(batch, return_tensors="pt", padding=True).to(
                        args.device
                    )
                    out = model(**tokens)
                elif hasattr(model, "generate"):
                    # try simple generate with text input; user loader should implement prepare_inputs
                    if hasattr(model, "tokenize"):
                        toks = model.tokenize(batch).to(args.device)
                        _ = model.generate(toks)
                    else:
                        # Best-effort: call generate with raw strings (some wrappers accept this)
                        _ = model.generate(batch)
                else:
                    raise RuntimeError(
                        "Model does not support automatic tokenization/generation in demo. Provide a loader returning (model, tokenizer)."
                    )
            except Exception as e:
                print(
                    "Model inference failed in demo. This demo requires a loader that returns (model, tokenizer) or a model with .generate()."
                )
                print("Error:", e)

            acts = hook.collect()
            _save_activations(acts, args.out_dir, batch_id)
            batch.clear()
            batch_id += 1

    hook.detach()


if __name__ == "__main__":
    main()
