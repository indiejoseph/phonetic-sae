"""Fallback text-to-phoneme conversion when model API doesn't expose it.

This module provides external G2P (Grapheme-to-Phoneme) converters for
English, Mandarin, and Cantonese, to be used when Qwen3-ForcedAligner
doesn't expose a text_to_phonemes method.

Usage:
    from src.alignment.phoneme_converter import get_converter

    converter = get_converter("en")
    phonemes = converter.text_to_phonemes("hello world")
    # ['h', 'ah', 'l', 'ow', 'w', 'er', 'l', 'd']
"""

import logging
from typing import List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PhonemeConverter(ABC):
    """Base class for text-to-phoneme conversion."""

    @abstractmethod
    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert text to list of phonemes.

        Args:
            text: Input text

        Returns:
            List of phoneme strings (lowercase)
        """
        pass


class G2pEnConverter(PhonemeConverter):
    """English text-to-phonemes using g2p_en library.

    Converts English text to ARPAbet phonemes (e.g., 'hello' -> ['h', 'eh', 'l', 'ow']).

    Requires: pip install g2p_en
    """

    def __init__(self):
        """Initialize G2P converter."""
        try:
            from g2p_en.g2p import G2p
            self.g2p = G2p()
            logger.info("✅ G2P English converter initialized")
        except ImportError as e:
            raise ImportError(
                "g2p_en not installed. Install with:\n"
                "  pip install g2p_en --break-system-packages"
            ) from e

    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert English text to ARPAbet phonemes.

        Args:
            text: English text to convert

        Returns:
            List of ARPAbet phonemes (lowercase, no stress markers)

        Example:
            >>> converter = G2pEnConverter()
            >>> converter.text_to_phonemes("hello")
            ['h', 'eh', 'l', 'ow']
        """
        phonemes = self.g2p(text)
        # Remove stress markers (0, 1, 2) and convert to lowercase
        result = []
        for p in phonemes:
            if isinstance(p, str):
                # Strip trailing digits (stress markers)
                clean_p = p.rstrip('0123456789').lower()
                if clean_p:  # Only add non-empty phonemes
                    result.append(clean_p)
        return result


class PinYinConverter(PhonemeConverter):
    """Mandarin text-to-phonemes using pypinyin library.

    Converts Mandarin Chinese text to pinyin phonemes
    (e.g., '你好' -> ['ni', 'hao']).

    Requires: pip install pypinyin
    """

    def __init__(self):
        """Initialize Pinyin converter."""
        try:
            from pypinyin import pinyin, NORMAL
            self.pinyin_fn = pinyin
            self.NORMAL = NORMAL
            logger.info("✅ Pinyin converter initialized")
        except ImportError as e:
            raise ImportError(
                "pypinyin not installed. Install with:\n"
                "  pip install pypinyin --break-system-packages"
            ) from e

    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert Mandarin text to pinyin phonemes.

        Args:
            text: Mandarin Chinese text to convert

        Returns:
            List of pinyin phonemes (lowercase, no tone marks)

        Example:
            >>> converter = PinYinConverter()
            >>> converter.text_to_phonemes("你好")
            ['ni', 'hao']
        """
        try:
            phonemes = self.pinyin_fn(text, style=self.NORMAL)
            # Flatten list of lists and lowercase
            result = []
            for p_list in phonemes:
                if p_list:  # Handle empty entries
                    result.append(p_list[0].lower())
            return result
        except Exception as e:
            logger.warning(f"Error converting text '{text}': {e}")
            # Fallback: return empty for error cases
            return []


class ToJyutpingConverter(PhonemeConverter):
    """Cantonese text-to-phonemes using ToJyutping library.

    Converts Cantonese text to jyutping romanization with excellent character coverage
    (e.g., '你好' -> ['nei5', 'hou2']).

    ToJyutping is specifically designed for Cantonese and has superior coverage
    compared to basic jyutping.

    Requires: pip install ToJyutping
    """

    def __init__(self):
        """Initialize ToJyutping converter."""
        try:
            import ToJyutping
            self.ToJyutping = ToJyutping
            logger.info("✅ ToJyutping converter initialized")
        except ImportError as e:
            raise ImportError(
                "ToJyutping not installed. Install with:\n"
                "  pip install ToJyutping --break-system-packages"
            ) from e

    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert Cantonese text to jyutping phonemes.

        Args:
            text: Cantonese Chinese text to convert

        Returns:
            List of jyutping romanizations (lowercase, with tone numbers)

        Note:
            - Excellent character coverage for Cantonese
            - Returns most common pronunciation for each character
            - Tone numbers (1-9) are included in output
            - Non-Chinese characters (punctuation) are skipped

        Example:
            >>> converter = ToJyutpingConverter()
            >>> converter.text_to_phonemes("你好")
            ['nei5', 'hou2']
        """
        phonemes = []
        try:
            # Use get_jyutping_list to get (character, jyutping) pairs
            jyutping_list = self.ToJyutping.get_jyutping_list(text)

            # Extract jyutping values, skip None (punctuation/unknown chars)
            for char, jyutping_val in jyutping_list:
                if jyutping_val is not None:
                    # Convert to lowercase for consistency
                    phonemes.append(jyutping_val.lower())

            return phonemes
        except Exception as e:
            logger.warning(f"Error converting text '{text}': {e}")
            # Fallback: return empty list
            return []


def get_converter(language: str) -> PhonemeConverter:
    """Get appropriate phoneme converter for language.

    Args:
        language: Language code ('en', 'english', 'zh', 'mandarin', 'yue', 'cantonese')

    Returns:
        PhonemeConverter instance for the specified language

    Raises:
        ValueError: If language not supported
        ImportError: If required library not installed

    Example:
        >>> converter = get_converter("en")
        >>> converter.text_to_phonemes("test")
        ['t', 'eh', 's', 't']
    """
    converters = {
        "en": G2pEnConverter,
        "english": G2pEnConverter,
        "zh": PinYinConverter,
        "mandarin": PinYinConverter,
        "yue": ToJyutpingConverter,
        "cantonese": ToJyutpingConverter,
    }

    lang_key = language.lower().strip()
    converter_class = converters.get(lang_key)

    if not converter_class:
        supported = ", ".join(set(converters.keys()))
        raise ValueError(
            f"Unsupported language: '{language}'\n"
            f"Supported: {supported}"
        )

    return converter_class()


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_cases = {
        "en": "hello world",
        "zh": "你好世界",
        "yue": "你好",
    }

    print("\n" + "=" * 70)
    print("PHONEME CONVERTER TEST")
    print("=" * 70 + "\n")

    for lang, text in test_cases.items():
        try:
            converter = get_converter(lang)
            phonemes = converter.text_to_phonemes(text)
            print(f"✅ {lang}: '{text}' → {phonemes}")
        except ImportError as e:
            print(f"⚠️  {lang}: Missing library")
            print(f"   {e}")
        except Exception as e:
            print(f"❌ {lang}: Error - {e}")

    print("\n" + "=" * 70)
