#!/usr/bin/env python3
"""Find available speakers in Qwen3-TTS model."""

import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Loading Qwen3-TTS model...")

    try:
        from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper

        wrapper = Qwen3TTSWrapper(device="cpu")
        model = wrapper.model

        logger.info("\n" + "="*60)
        logger.info("Searching for speaker information...")
        logger.info("="*60)

        # Check various attributes that might contain speakers
        speaker_attrs = [
            'speakers',
            'speaker_list',
            'available_speakers',
            'speaker_names',
            '_speakers',
            '_speaker_list',
        ]

        found_speakers = False

        for attr in speaker_attrs:
            if hasattr(model, attr):
                speakers = getattr(model, attr)
                logger.info(f"\n✅ Found: model.{attr}")
                logger.info(f"   Value: {speakers}")
                logger.info(f"   Type: {type(speakers)}")
                found_speakers = True

        # Check nested model
        if hasattr(model, 'model'):
            logger.info("\nChecking model.model...")
            for attr in speaker_attrs:
                if hasattr(model.model, attr):
                    speakers = getattr(model.model, attr)
                    logger.info(f"\n✅ Found: model.model.{attr}")
                    logger.info(f"   Value: {speakers}")
                    found_speakers = True

        if not found_speakers:
            logger.warning("\n⚠️  No speaker list found in standard locations")

        # Try to test different speaker names
        logger.info("\n" + "="*60)
        logger.info("Testing speaker names...")
        logger.info("="*60)

        test_speakers = ["default", "", "Vivian", "Alice", "speaker_0", "speaker_1"]

        for speaker in test_speakers:
            logger.info(f"\nTesting speaker: '{speaker}'")
            try:
                result = wrapper.generate(
                    text="test",
                    speaker=speaker,
                    instruct=""
                )
                logger.info(f"  ✅ SUCCESS! Speaker '{speaker}' works")
                logger.info(f"     Output type: {type(result)}")
                if isinstance(result, dict) and "waveform" in result:
                    logger.info(f"     Waveform shape: {result['waveform'].shape if hasattr(result['waveform'], 'shape') else 'N/A'}")
                return True
            except Exception as e:
                error_msg = str(e)[:100]
                logger.info(f"  ❌ Failed: {error_msg}")

        logger.error("\n❌ No speakers found that work")
        return False

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
