#!/usr/bin/env python3
"""Inspect Qwen3TTSModel forward pass capabilities.

Determines the correct way to invoke the model for activation capture.
"""

import logging
import torch
import inspect
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def inspect_model_methods(model, name="Model"):
    """Inspect callable methods and signatures."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Inspecting {name}")
    logger.info(f"{'='*60}")

    logger.info(f"Type: {type(model).__name__}")
    logger.info(f"Module: {type(model).__module__}")

    # Check __call__ method
    logger.info("\n__call__ method:")
    if hasattr(model, '__call__'):
        logger.info(f"  Callable: Yes")
        try:
            sig = inspect.signature(model.__call__)
            logger.info(f"  Signature: {sig}")
        except Exception as e:
            logger.info(f"  Could not get signature: {e}")
    else:
        logger.info(f"  Callable: No")

    # Check forward method
    logger.info("\nforward() method:")
    if hasattr(model, 'forward'):
        logger.info(f"  Has forward: Yes")
        try:
            sig = inspect.signature(model.forward)
            logger.info(f"  Signature: {sig}")
        except Exception as e:
            logger.info(f"  Could not get signature: {e}")
    else:
        logger.info(f"  Has forward: No")

    # List all public methods
    logger.info("\nPublic methods (generate*, forward*, __call__):")
    for attr in dir(model):
        if not attr.startswith('_'):
            obj = getattr(model, attr)
            if callable(obj):
                if any(x in attr for x in ['generate', 'forward', 'call']):
                    logger.info(f"  - {attr}")

    # List generate methods specifically
    logger.info("\nGenerate methods:")
    for attr in dir(model):
        if 'generate' in attr.lower():
            obj = getattr(model, attr)
            if callable(obj):
                logger.info(f"  - {attr}")
                try:
                    sig = inspect.signature(obj)
                    logger.info(f"    Signature: {sig}")
                except Exception as e:
                    logger.info(f"    Could not get signature: {e}")


def main():
    logger.info("Loading Qwen3-TTS model...")

    try:
        from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper

        # Load model with bfloat16
        wrapper = Qwen3TTSWrapper(device="cpu", dtype=torch.bfloat16)
        model = wrapper.model

        logger.info(f"✅ Model loaded: {type(model).__name__}")

        # Inspect wrapper
        inspect_model_methods(wrapper, "Qwen3TTSWrapper")

        # Inspect underlying model
        inspect_model_methods(model, "Qwen3TTSModel (wrapper.model)")

        # Inspect inner model if accessible
        if hasattr(model, 'model'):
            inspect_model_methods(model.model, "model.model (Qwen3TTSForConditionalGeneration)")

        # Inspect talker if accessible
        if hasattr(model, 'model') and hasattr(model.model, 'talker'):
            inspect_model_methods(model.model.talker, "model.model.talker (Qwen3TTSTalkerForConditionalGeneration)")

        # Test different invocation methods
        logger.info(f"\n{'='*60}")
        logger.info("Testing different invocation methods")
        logger.info(f"{'='*60}")

        # Create dummy input_ids
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

        # Test 1: Direct call on wrapper.model
        logger.info("\n[Test 1] Calling model(input_ids):")
        try:
            with torch.no_grad():
                output = model(input_ids)
            logger.info(f"  ✅ Success! Output type: {type(output)}")
        except Exception as e:
            logger.info(f"  ❌ Failed: {type(e).__name__}: {str(e)[:100]}")

        # Test 2: Direct forward call
        logger.info("\n[Test 2] Calling model.forward(input_ids):")
        try:
            with torch.no_grad():
                output = model.forward(input_ids)
            logger.info(f"  ✅ Success! Output type: {type(output)}")
        except Exception as e:
            logger.info(f"  ❌ Failed: {type(e).__name__}: {str(e)[:100]}")

        # Test 3: Using generate method with text
        logger.info("\n[Test 3] Calling model.generate(text='hello'):")
        try:
            with torch.no_grad():
                output = model.generate(text="hello")
            logger.info(f"  ✅ Success! Output type: {type(output)}")
        except Exception as e:
            logger.info(f"  ❌ Failed: {type(e).__name__}: {str(e)[:100]}")

        # Test 4: Check if we can access the actual talker model
        logger.info("\n[Test 4] Accessing internal talker model:")
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'talker'):
                talker = model.model.talker
                logger.info(f"  ✅ Found talker: {type(talker).__name__}")

                # Try calling talker directly
                logger.info("  [Test 4a] Calling talker(input_ids):")
                try:
                    with torch.no_grad():
                        output = talker(input_ids)
                    logger.info(f"    ✅ Success! Output type: {type(output)}")
                except Exception as e:
                    logger.info(f"    ❌ Failed: {type(e).__name__}: {str(e)[:100]}")
            else:
                logger.info("  ❌ Cannot access talker")
        except Exception as e:
            logger.info(f"  ❌ Failed: {type(e).__name__}: {str(e)[:100]}")

        # Test 5: Check wrapper's tokenizer
        logger.info("\n[Test 5] Tokenizer check:")
        if hasattr(wrapper, 'tokenizer') and wrapper.tokenizer:
            logger.info(f"  ✅ Tokenizer available: {type(wrapper.tokenizer).__name__}")
            # Try tokenizing
            try:
                tokens = wrapper.tokenizer.encode("hello world", return_tensors="pt")
                logger.info(f"  ✅ Tokenized successfully: shape {tokens.shape}")
            except Exception as e:
                logger.info(f"  ❌ Tokenization failed: {e}")
        else:
            logger.info(f"  ❌ No tokenizer available")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
