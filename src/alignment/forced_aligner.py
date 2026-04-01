"""Forced alignment for phoneme-to-frame mapping using Qwen3-ForcedAligner.

This module provides tools to align text/phonemes with audio frames,
enabling frame-level phoneme labels for SAE feature analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from qwen_asr import Qwen3ForcedAligner as QwenAligner

logger = logging.getLogger(__name__)


@dataclass
class PhonemeAlignment:
    """Result of phoneme alignment."""

    phonemes: list[str]  # List of phonemes in sequence
    frame_boundaries: list[tuple[int, int]]  # [(start_frame, end_frame), ...]
    frame_to_phoneme: list[int]  # frame_to_phoneme[i] = phoneme_index
    duration_ms: float  # Total duration in milliseconds


class QwenForcedAligner:
    """Wrapper for Qwen3-ForcedAligner-0.6B from HuggingFace.

    Performs automatic alignment of text/phonemes to audio frames.
    Supports: English (en), Mandarin (zh), Cantonese (yue).

    Model: https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B
    """

    # Default phoneme inventories per language
    # NOTE: These are based on standard phoneme sets but should be verified
    # against the actual Qwen3-ForcedAligner model output.
    # The model may use different phoneme inventories—check model outputs.
    DEFAULT_PHONEME_SETS = {
        "en": [
            # ARPAbet (standard English TTS/ASR)
            "aa", "ae", "ah", "ao", "aw", "ay", "b", "ch", "d", "dh",
            "eh", "er", "ey", "f", "g", "hh", "ih", "iy", "jh", "k",
            "l", "m", "n", "ng", "oh", "ow", "oy", "p", "r", "s",
            "sh", "t", "th", "uh", "uw", "v", "w", "y", "z", "zh",
            "pau", "sil",  # silence markers
        ],
        "zh": [
            # Mandarin pinyin initials + finals
            # NOTE: Verify against actual model output
            "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
            "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s",
            "a", "o", "e", "i", "u", "ü",
            "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "ong",
            "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "iong",
            "ua", "uo", "uai", "ui", "uan", "un", "uang", "ueng",
            "üe", "üan", "ün",
            "pau", "sil",
        ],
        "yue": [
            # Cantonese Jyutping (simplified)
            # NOTE: Verify against actual model output
            "p", "ph", "m", "f", "t", "th", "n", "l", "k", "kh", "ng", "h",
            "gw", "kw", "z", "c", "s", "j",
            "a", "e", "i", "o", "u", "oe",
            "ai", "ei", "oi", "ou", "au",
            "an", "en", "on", "un", "in",
            "ang", "eng", "ong", "ung", "ing",
            "am", "em", "im", "om", "um",
            "ap", "ep", "ip", "op", "up",
            "ak", "ek", "ik", "ok", "uk",
            "pau", "sil",
        ],
    }

    # Will be populated from model if available
    PHONEME_SETS = DEFAULT_PHONEME_SETS.copy()

    def __init__(self, device: str = "cuda", language: str = "en"):
        """Initialize Qwen3-ForcedAligner.

        Args:
            device: torch device ("cuda" or "cpu")
            language: language code ("en", "zh", "yue")
        """
        self.device = device
        self.language = language

        # Map language codes to Qwen API language names
        self.language_names = {
            "en": "English",
            "zh": "Chinese",
            "yue": "Cantonese",
        }

        if language not in self.PHONEME_SETS:
            raise ValueError(f"Unsupported language: {language}. Choose from: {list(self.PHONEME_SETS.keys())}")

        try:
            logger.info(f"Loading Qwen3-ForcedAligner for {language}...")

            # Use qwen_asr's Qwen3ForcedAligner directly
            device_map = "cuda:0" if device == "cuda" else "cpu"
            self.model = QwenAligner.from_pretrained(
                "Qwen/Qwen3-ForcedAligner-0.6B",
                dtype=torch.bfloat16,
                device_map=device_map,
            )
            logger.info("✅ Forced aligner loaded successfully")

            # Use default phoneme sets (model doesn't expose them directly)
            self.phoneme_set = self.PHONEME_SETS[language]
            logger.info(f"✅ Using {len(self.phoneme_set)} phonemes for {language}")

        except ImportError as e:
            logger.error(f"qwen_asr library required: pip install qwen-asr. Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load Qwen3-ForcedAligner: {e}")
            raise

    def align(
        self,
        text: str,
        audio_path: Optional[str] = None,
        audio_tensor: Optional[torch.Tensor] = None,
        sample_rate: int = 16000,
    ) -> PhonemeAlignment:
        """Align text/phonemes to audio frames using Qwen3-ForcedAligner.

        Args:
            text: Input text (transcript)
            audio_path: Path to audio file (alternative to audio_tensor)
            audio_tensor: Audio waveform tensor of shape (1, T) or (T,)
            sample_rate: Audio sample rate (default 16000)

        Returns:
            PhonemeAlignment object with phoneme sequence and frame boundaries
        """
        # Prepare audio input
        if audio_path is not None:
            audio_input = str(audio_path)
            audio_tensor, sr = torchaudio.load(audio_path)
            if sr != sample_rate:
                resample = torchaudio.transforms.Resample(sr, sample_rate)
                audio_tensor = resample(audio_tensor)
        elif audio_tensor is not None:
            audio_input = audio_tensor
            # Ensure audio is mono
            if audio_tensor.dim() > 1:
                audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
        else:
            raise ValueError("Either audio_path or audio_tensor must be provided")

        # Get language name for Qwen API
        language_name = self.language_names.get(self.language, self.language)

        logger.info(f"Aligning text (len={len(text)}) in {language_name}")

        try:
            # Call Qwen3-ForcedAligner.align()
            # Returns: list of lists of objects with .text, .start_time, .end_time
            results = self.model.align(
                audio=audio_input,
                text=text,
                language=language_name,
            )
        except Exception as e:
            logger.error(f"Alignment failed: {e}")
            raise

        # Parse results: results[0] is list of phoneme objects
        # Each object has .text (phoneme), .start_time, .end_time (in seconds)
        if not results or not results[0]:
            logger.warning("No alignment results returned")
            return PhonemeAlignment(
                phonemes=[],
                frame_boundaries=[],
                frame_to_phoneme=[],
                duration_ms=0,
            )

        alignment_list = results[0]
        phonemes = []
        boundaries = []

        # Convert time boundaries to frames
        for item in alignment_list:
            phoneme_text = item.text
            start_time = float(item.start_time)  # in seconds
            end_time = float(item.end_time)      # in seconds

            phonemes.append(phoneme_text)

            # Convert to frame indices (assuming sample_rate frames per second)
            start_frame = int(start_time * sample_rate)
            end_frame = int(end_time * sample_rate)
            boundaries.append((start_frame, end_frame))

        # Get total duration from last boundary or from tensor
        if boundaries:
            duration_ms = boundaries[-1][1] / sample_rate * 1000
        else:
            duration_ms = 0

        # Create frame-to-phoneme mapping
        if boundaries:
            num_frames = boundaries[-1][1]
            frame_to_phoneme = [0] * num_frames

            for phoneme_idx, (start_frame, end_frame) in enumerate(boundaries):
                for frame_idx in range(start_frame, min(end_frame, num_frames)):
                    frame_to_phoneme[frame_idx] = phoneme_idx
        else:
            frame_to_phoneme = []

        logger.info(
            f"✅ Aligned {len(phonemes)} phonemes "
            f"(duration: {duration_ms:.0f}ms)"
        )

        return PhonemeAlignment(
            phonemes=phonemes,
            frame_boundaries=boundaries,
            frame_to_phoneme=frame_to_phoneme,
            duration_ms=duration_ms,
        )

    def get_phoneme_inventory(self) -> list[str]:
        """Return the phoneme inventory for the current language."""
        return self.phoneme_set

    def validate_phonemes(self, phonemes: list[str]) -> bool:
        """Check if all phonemes are in the inventory."""
        return all(p in self.phoneme_set for p in phonemes)


class LanguageSpecificAligner:
    """Manages aligners for multiple languages."""

    def __init__(self, languages: list[str] = None, device: str = "cuda"):
        """Initialize aligners for specified languages.

        Args:
            languages: List of language codes ("en", "zh", "yue")
            device: torch device
        """
        if languages is None:
            languages = ["en", "zh", "yue"]

        self.aligners = {}
        for lang in languages:
            try:
                self.aligners[lang] = QwenForcedAligner(device=device, language=lang)
            except Exception as e:
                logger.warning(f"Failed to load aligner for {lang}: {e}")

    def align(self, text: str, audio_path: str, language: str) -> Optional[PhonemeAlignment]:
        """Align using language-specific aligner.

        Args:
            text: Input text
            audio_path: Path to audio file
            language: Language code ("en", "zh", "yue")

        Returns:
            PhonemeAlignment or None if aligner not available
        """
        if language not in self.aligners:
            logger.error(f"Aligner not available for {language}")
            return None

        return self.aligners[language].align(text=text, audio_path=audio_path)

    def get_available_languages(self) -> list[str]:
        """Return list of available language aligners."""
        return list(self.aligners.keys())
