#!/usr/bin/env python3
"""Inspect the actual Qwen3-TTS model structure to find where layers are stored."""

import torch
from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper

def inspect_model_structure(model, prefix="", max_depth=3, current_depth=0):
    """Recursively inspect model structure."""
    if current_depth > max_depth:
        return

    # Get all attributes
    attrs = dir(model)

    # Filter for meaningful attributes (not private, not methods)
    meaningful_attrs = []
    for attr in attrs:
        if attr.startswith("_"):
            continue
        try:
            val = getattr(model, attr)
            # Skip methods and properties
            if callable(val) and not isinstance(val, torch.nn.Module):
                continue
            meaningful_attrs.append((attr, type(val).__name__))
        except:
            pass

    print(f"\n{prefix}{type(model).__name__}:")
    for attr, type_name in sorted(meaningful_attrs)[:20]:
        print(f"  {prefix}{attr}: {type_name}")

    # Dive deeper into module children
    if hasattr(model, "_modules"):
        print(f"\n{prefix}_modules (children):")
        for name, child_module in list(model._modules.items())[:20]:
            if child_module is not None:
                print(f"  {prefix}{name}: {type(child_module).__name__}")
                if current_depth < max_depth - 1:
                    inspect_model_structure(child_module, prefix + "  ", max_depth, current_depth + 1)

def main():
    print("Loading Qwen3-TTS model...")
    wrapper = Qwen3TTSWrapper(device="cuda", dtype=torch.float16)

    print("\n" + "="*70)
    print("OUTER WRAPPER (qwen_tts.Qwen3TTSModel)")
    print("="*70)
    inspect_model_structure(wrapper.model, max_depth=2)

    if hasattr(wrapper.model, "model"):
        print("\n" + "="*70)
        print("INNER MODEL (wrapper.model.model)")
        print("="*70)
        inspect_model_structure(wrapper.model.model, max_depth=2)

        if hasattr(wrapper.model.model, "base_model"):
            print("\n" + "="*70)
            print("BASE MODEL (unwrapped from PEFT)")
            print("="*70)
            inspect_model_structure(wrapper.model.model.base_model, max_depth=3)

if __name__ == "__main__":
    main()
