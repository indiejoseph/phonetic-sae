#!/usr/bin/env python3
"""Install and test fallback phoneme converters.

Run this if Qwen3-ForcedAligner inspection fails or doesn't expose text_to_phonemes.

Usage:
    python scripts/setup_phoneme_converters.py
    python scripts/setup_phoneme_converters.py --test-only
"""

import subprocess
import sys
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def install_packages():
    """Install required packages for phoneme converters."""
    packages = [
        "g2p_en",      # English grapheme-to-phoneme
        "pypinyin",    # Mandarin Chinese
        "tojyutping",  # Cantonese (excellent support)
    ]

    logger.info("\n" + "=" * 70)
    logger.info("INSTALLING PHONEME CONVERTER DEPENDENCIES")
    logger.info("=" * 70 + "\n")

    installed = []
    failed = []

    for package in packages:
        logger.info(f"Installing {package}...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package, "--break-system-packages"],
                capture_output=True,
                text=True,
                timeout=60,
                check=True
            )
            logger.info(f"  ✅ {package} installed")
            installed.append(package)
        except subprocess.CalledProcessError as e:
            logger.error(f"  ❌ Failed to install {package}")
            logger.error(f"     Error: {e.stderr}")
            failed.append(package)
        except subprocess.TimeoutExpired:
            logger.error(f"  ⏱️  Installation timed out for {package}")
            failed.append(package)

    logger.info("\n" + "=" * 70)
    logger.info(f"INSTALLATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"✅ Installed: {len(installed)}/{len(packages)}")
    for pkg in installed:
        logger.info(f"   - {pkg}")

    if failed:
        logger.warning(f"❌ Failed: {len(failed)}/{len(packages)}")
        for pkg in failed:
            logger.warning(f"   - {pkg}")

    return len(failed) == 0


def test_converters():
    """Test phoneme converters."""
    logger.info("\n" + "=" * 70)
    logger.info("TESTING PHONEME CONVERTERS")
    logger.info("=" * 70 + "\n")

    test_cases = {
        "en": ("hello world", ["h", "eh", "l", "ow"]),
        "zh": ("你好", ["ni", "hao"]),
        "yue": ("你好", None),  # No expected output, just verify it works
    }

    all_passed = True

    try:
        from src.alignment.phoneme_converter import get_converter
    except ImportError as e:
        logger.error(f"Cannot import phoneme_converter: {e}")
        return False

    for lang, (text, expected) in test_cases.items():
        try:
            converter = get_converter(lang)
            phonemes = converter.text_to_phonemes(text)

            logger.info(f"✅ {lang}: '{text}' → {phonemes}")

            if expected:
                if phonemes[:len(expected)] == expected:
                    logger.info(f"   Output matches expected prefix ✓")
                else:
                    logger.warning(f"   Output differs from expected: {expected}")
                    all_passed = False

        except Exception as e:
            logger.error(f"❌ {lang}: {e}")
            all_passed = False

    logger.info("\n" + "=" * 70)
    if all_passed:
        logger.info("✅ ALL TESTS PASSED")
    else:
        logger.warning("⚠️  SOME TESTS FAILED - See above for details")
    logger.info("=" * 70)

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Install and test fallback phoneme converters"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test converters (don't install)"
    )
    parser.add_argument(
        "--install-only",
        action="store_true",
        help="Only install packages (don't test)"
    )

    args = parser.parse_args()

    success = True

    if not args.test_only:
        success = install_packages() and success

    if not args.install_only and success:
        success = test_converters() and success

    if success:
        logger.info("\n✅ Setup complete! Fallback converters are ready.")
        logger.info("\nUsage:")
        logger.info("  from src.alignment.phoneme_converter import get_converter")
        logger.info("  converter = get_converter('en')")
        logger.info("  phonemes = converter.text_to_phonemes('hello')")
        sys.exit(0)
    else:
        logger.error("\n❌ Setup failed. See above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
