#!/usr/bin/env python3
"""Inspect the actual Qwen3TTSModel API to understand what generation methods exist."""

import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Loading model to inspect API...")

    try:
        # Try importing from qwen_tts directly
        logger.info("\nAttempting to load Qwen3TTSModel from qwen_tts...")
        try:
            from qwen_tts import Qwen3TTSModel

            model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
            logger.info(f"✅ Loaded: {type(model).__name__}")

            # Check all public methods
            logger.info("\nPublic methods on Qwen3TTSModel:")
            methods = [m for m in dir(model) if not m.startswith('_')]

            generate_methods = []
            other_methods = []

            for method in sorted(methods):
                attr = getattr(model, method)
                if callable(attr):
                    if 'generate' in method.lower():
                        generate_methods.append(method)
                    else:
                        other_methods.append(method)

            if generate_methods:
                logger.info("\n🎯 Generation methods found:")
                for method in generate_methods:
                    logger.info(f"  • {method}")
            else:
                logger.warning("\n⚠️  NO GENERATION METHODS FOUND")

            logger.info(f"\nOther public methods ({len(other_methods)} total):")
            for method in other_methods[:15]:  # Show first 15
                logger.info(f"  • {method}")
            if len(other_methods) > 15:
                logger.info(f"  ... and {len(other_methods) - 15} more")

            # Try to check method signatures
            logger.info("\n" + "="*60)
            logger.info("Method signatures:")
            logger.info("="*60)

            import inspect

            for method_name in generate_methods:
                try:
                    method = getattr(model, method_name)
                    sig = inspect.signature(method)
                    logger.info(f"\n{method_name}{sig}")
                except Exception as e:
                    logger.warning(f"Could not get signature for {method_name}: {e}")

            # Check class hierarchy
            logger.info("\n" + "="*60)
            logger.info("Class hierarchy:")
            logger.info("="*60)
            for cls in type(model).__mro__[:5]:
                logger.info(f"  {cls}")

            # Try to understand the model structure
            logger.info("\n" + "="*60)
            logger.info("Model structure:")
            logger.info("="*60)

            # Check for model attribute (inner model)
            if hasattr(model, 'model'):
                logger.info("Has .model attribute")
                inner = model.model
                logger.info(f"  Type: {type(inner).__name__}")

                if hasattr(inner, 'generate'):
                    logger.info("  └─ Has .generate() method")
                if hasattr(inner, 'talker'):
                    logger.info("  └─ Has .talker attribute")

            # Check for generate method at any level
            logger.info("\nSearching for 'generate' across hierarchy:")
            if hasattr(model, 'generate'):
                logger.info("  ✅ model.generate exists")
            else:
                logger.info("  ❌ model.generate NOT found")

            if hasattr(model, 'model') and hasattr(model.model, 'generate'):
                logger.info("  ✅ model.model.generate exists")
            else:
                logger.info("  ❌ model.model.generate NOT found")

        except ImportError as e:
            logger.error(f"Could not import from qwen_tts: {e}")
            logger.info("\nTrying alternative import...")

            from transformers import AutoModel
            model = AutoModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
            logger.info(f"Loaded via transformers: {type(model).__name__}")

            # Check methods
            generate_methods = [m for m in dir(model) if 'generate' in m.lower() and not m.startswith('_')]
            logger.info(f"Generation methods: {generate_methods if generate_methods else 'None found'}")

    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
