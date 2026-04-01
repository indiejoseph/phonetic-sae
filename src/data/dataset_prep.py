"""Dataset preparation utilities for activation mining.

Handles loading and preparing text-speech datasets for activation capture.
Supports LibriTTS-R, TIMIT, and custom multilingual datasets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import torch

logger = logging.getLogger(__name__)


@dataclass
class TextSpeechPair:
    """A single text-speech pair for synthesis."""

    text: str
    language: str = "English"
    speaker_id: Optional[str] = None
    ref_audio_path: Optional[str] = None
    ref_text: Optional[str] = None


class DatasetIterator:
    """Generic iterator for text-speech pairs."""

    def __init__(self, pairs: list[TextSpeechPair], batch_size: int = 1):
        self.pairs = pairs
        self.batch_size = batch_size
        self.current_idx = 0

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> list[TextSpeechPair]:
        if self.current_idx >= len(self.pairs):
            raise StopIteration
        batch = self.pairs[self.current_idx : self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        return batch

    def __len__(self) -> int:
        return len(self.pairs)


class LibriTTSRDataset:
    """LibriTTS-R corpus (English, multi-speaker TTS data).

    To use:
    1. Download from: https://www.openslr.org/141/
    2. Extract to data/datasets/LibriTTS_R/
    """

    def __init__(self, root_dir: str | Path, subset: str = "clean"):
        self.root_dir = Path(root_dir)
        self.subset = subset  # "clean" or "other"

        if not self.root_dir.exists():
            raise FileNotFoundError(f"LibriTTS-R not found at {self.root_dir}")

        self.pairs = self._load_pairs()
        logger.info(f"Loaded {len(self.pairs)} LibriTTS-R samples")

    def _load_pairs(self) -> list[TextSpeechPair]:
        """Load text-speech pairs from LibriTTS-R directory structure."""
        pairs = []
        # LibriTTS-R structure: train-clean-{360,100}/... or train-other-500/...
        pattern = f"train-{self.subset}-*"
        for split_dir in self.root_dir.glob(pattern):
            if not split_dir.is_dir():
                continue
            # Iterate speakers
            for speaker_dir in split_dir.iterdir():
                if not speaker_dir.is_dir():
                    continue
                # Iterate chapters (books)
                for chapter_dir in speaker_dir.iterdir():
                    if not chapter_dir.is_dir():
                        continue
                    # Find .wav and corresponding .normalized.txt
                    for wav_file in chapter_dir.glob("*.wav"):
                        txt_file = wav_file.with_suffix(".normalized.txt")
                        if txt_file.exists():
                            with open(txt_file, "r") as f:
                                text = f.read().strip()
                            pairs.append(
                                TextSpeechPair(
                                    text=text,
                                    language="English",
                                    speaker_id=speaker_dir.name,
                                    ref_audio_path=str(wav_file),
                                )
                            )
        return pairs

    def iterator(self, batch_size: int = 1) -> DatasetIterator:
        """Get an iterator over the dataset."""
        return DatasetIterator(self.pairs, batch_size=batch_size)


class CustomDataset:
    """Custom multilingual dataset (Mandarin/Cantonese/English).

    Expected format:
    - CSV/JSONL file with columns: text, lang (or language), audio_path, (optional) speaker_id

    Supported language codes:
    - "en" or "English" → English
    - "zh" or "Mandarin" → Mandarin Chinese
    - "yue" or "Cantonese" → Cantonese
    """

    # Language code mappings
    LANGUAGE_CODE_MAP = {
        "en": "English",
        "zh": "Mandarin",
        "yue": "Cantonese",
        "english": "English",
        "mandarin": "Mandarin",
        "cantonese": "Cantonese",
    }

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV/JSONL not found at {self.csv_path}")

        self.pairs = self._load_from_csv()
        logger.info(f"Loaded {len(self.pairs)} custom dataset samples")
        logger.info(f"Language distribution: {self.get_language_distribution()}")

    def _normalize_language(self, lang: str) -> str:
        """Normalize language code/name to full language name."""
        if not lang:
            return "English"

        lang_normalized = lang.strip().lower()
        return self.LANGUAGE_CODE_MAP.get(lang_normalized, lang.strip())

    def _load_from_csv(self) -> list[TextSpeechPair]:
        """Load from CSV or JSONL with columns: text, lang/language, audio_path, [speaker_id]."""
        import csv
        import json

        pairs = []

        # Detect file format
        if self.csv_path.suffix.lower() == ".jsonl":
            # Load JSONL format
            with open(self.csv_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON at line {line_num}: {e}")
                        continue

                    pair = self._process_row(row)
                    if pair:
                        pairs.append(pair)
        else:
            # Load CSV format
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pair = self._process_row(row)
                    if pair:
                        pairs.append(pair)

        return pairs

    def _process_row(self, row: dict) -> Optional[TextSpeechPair]:
        """Process a single row from CSV or JSONL."""
        text = row.get("text", "").strip() if isinstance(row.get("text"), str) else ""

        # Handle both "lang" and "language" columns
        language = row.get("lang") or row.get("language", "")
        language = self._normalize_language(language)

        # For JSONL format, audio_path might be optional if we only care about text
        audio_path = row.get("audio_path", "").strip() if isinstance(row.get("audio_path"), str) else ""
        speaker_id = row.get("speaker_id")

        if not text:
            logger.warning(f"Skipping row with missing text: {row}")
            return None

        # Allow missing audio_path for text-only datasets
        pair = TextSpeechPair(
            text=text,
            language=language,
            speaker_id=speaker_id,
            ref_audio_path=audio_path if audio_path else None,
        )
        return pair

    def iterator(self, batch_size: int = 1) -> DatasetIterator:
        """Get an iterator over the dataset."""
        return DatasetIterator(self.pairs, batch_size=batch_size)

    def filter_by_language(self, language: str) -> list[TextSpeechPair]:
        """Get all pairs for a specific language."""
        return [p for p in self.pairs if p.language == language]

    def get_language_distribution(self) -> dict[str, int]:
        """Get count of pairs per language."""
        dist = {}
        for pair in self.pairs:
            dist[pair.language] = dist.get(pair.language, 0) + 1
        return dist


def create_pilot_dataset(num_samples: int = 100) -> list[TextSpeechPair]:
    """Create a small pilot dataset for testing (simple English sentences)."""
    pilot_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, how are you doing today?",
        "Machine learning is fascinating.",
        "Natural language processing enables AI.",
        "This is a test sentence for speech synthesis.",
    ]

    pairs = []
    for i in range(num_samples):
        text = pilot_texts[i % len(pilot_texts)]
        pairs.append(
            TextSpeechPair(
                text=f"{text} [sample {i}]",
                language="English",
            )
        )
    return pairs
