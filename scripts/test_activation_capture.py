#!/usr/bin/env python3
"""Test activation capture pipeline with hooks attached.

Minimal test to verify:
1. Model loads correctly
2. Hooks attach to correct layers
3. Activations are captured when model runs
"""

import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Testing Activation Capture Pipeline")
    logger.info("=" * 60)

    try:
        # Step 1: Load model
        logger.info("\n[1/5] Loading Qwen3-TTS model...")
        from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper

        wrapper = Qwen3TTSWrapper(device="cpu", dtype=torch.bfloat16)
        model = wrapper.model
        logger.info(f"✅ Model loaded: {type(model).__name__}")

        # Step 2: Create hook
        logger.info("\n[2/5] Creating activation hook...")
        from src.hooks import ActivationHook

        target_layers = wrapper.get_target_layers()
        layer_accessor = wrapper.get_layer_accessor()

        hook = ActivationHook(
            model,
            layer_indices=target_layers,
            layer_accessor=layer_accessor,
            device="cpu",
            dtype=torch.float16,
        )
        logger.info(f"✅ Hook created for layers: {target_layers}")

        # Step 3: Attach hooks
        logger.info("\n[3/5] Attaching hooks to MLP post-activations...")
        hook.attach("mlp")
        logger.info(f"✅ Hooks attached")

        # Step 4: Run model forward pass using generate()
        logger.info("\n[4/5] Running model forward pass with text...")
        test_text = "Hello, this is a test sentence."
        logger.info(f"  Text: '{test_text}'")

        try:
            with torch.no_grad():
                result = wrapper.generate(text=test_text)
            logger.info(f"✅ Forward pass successful")
            logger.info(f"  Output type: {type(result)}")
            if isinstance(result, dict) and "waveform" in result:
                logger.info(f"  Waveform shape: {result['waveform'].shape}")
                logger.info(f"  Sample rate: {result['sample_rate']}")
        except Exception as e:
            logger.error(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            hook.detach()
            return False

        # Step 5: Collect activations
        logger.info("\n[5/5] Collecting activations...")
        activations = hook.collect()
        hook.detach()

        if activations:
            logger.info(f"✅ Activations collected!")
            logger.info(f"  Number of layers: {len(activations)}")
            for layer_idx, act_tensor in sorted(activations.items()):
                logger.info(f"  Layer {layer_idx:02d}: shape {act_tensor.shape}, dtype {act_tensor.dtype}")
            logger.info("\n✅ ACTIVATION CAPTURE PIPELINE WORKS!")
            return True
        else:
            logger.warning("⚠️  No activations captured")
            logger.warning("  Hooks may not have fired correctly")
            return False

    except Exception as e:
        logger.error(f"Failed to test pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
