"""Wrapper for Qwen3-TTS-0.6B with activation hook support.

Provides an interface to load and run Qwen3-TTS inference with
automatic activation capture from the Talker's phonetic layers.

Architecture:
    - Talker: 28-layer Qwen3-TTS (d_model=1024)
    - Code Predictor: 5-layer refinement for acoustic details
    - Speech Decoder: ConvNet → waveform

Target layers for phonetic SAE: Layers 1–7 of Talker (first 25%)
Hook point: model.talker.layers[i].mlp (post-activation)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Qwen3TTSWrapper:
    """Wrapper for Qwen3-TTS-0.6B with activation hook support.

    Parameters
    ----------
    model_path : str or Path
        Path to pretrained Qwen3-TTS model directory or HuggingFace ID
    device : str
        Device to load model on ("cuda", "cpu", etc.)
    dtype : torch.dtype
        Model dtype (torch.float16 for memory efficiency on consumer GPUs)
    """

    def __init__(
        self,
        model_path: str | Path = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        # Keep model_path as string if it's a HuggingFace ID (contains '/'),
        # only convert local paths to Path objects
        if isinstance(model_path, str):
            # HuggingFace model ID contains '/', local paths don't
            self.model_path = model_path if "/" in model_path else Path(model_path)
        else:
            self.model_path = model_path

        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load Qwen3-TTS model using qwen_tts package.

        Note: Use qwen_tts.Qwen3TTSModel, NOT transformers.AutoModelForCausalLM
        The qwen_tts package handles the specialized Qwen3-TTS architecture.
        """
        try:
            from qwen_tts import Qwen3TTSModel

            logger.info(f"Loading Qwen3-TTS from: {self.model_path}")

            # Convert device format: "cuda" → "cuda:0"
            device_map = "cuda:0" if self.device == "cuda" else self.device

            self.model = Qwen3TTSModel.from_pretrained(
                str(self.model_path),
                device_map=device_map,
                dtype=self.dtype,
                # Uncomment if flash_attention_2 is available and desired
                # attn_implementation="flash_attention_2",
            )

            logger.info(f"✅ Qwen3-TTS model loaded on {device_map} with dtype {self.dtype}")

            # Load tokenizer for text encoding
            # Try loading tokenizer from model repo (which has vocab.json and tokenizer.json)
            from transformers import AutoTokenizer

            self.tokenizer = None

            # First, try loading from the same model repo as TTS
            try:
                loaded = AutoTokenizer.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=True,
                )
                logger.debug(f"Loaded object type: {type(loaded).__name__}")
                self.tokenizer = loaded
                logger.info(f"✅ Tokenizer loaded from {self.model_path} (type: {type(self.tokenizer).__name__})")
            except Exception as e:
                logger.debug(f"Could not load tokenizer from model repo: {e}")

            # Fallback: Try other Qwen tokenizers
            if self.tokenizer is None:
                tokenizer_candidates = [
                    "Qwen/Qwen2.5-7B",      # Qwen2.5 tokenizer
                    "Qwen/Qwen2-7B",        # Qwen2 tokenizer
                    "Qwen/Qwen-7B",         # Original Qwen tokenizer
                ]

                for tokenizer_id in tokenizer_candidates:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            tokenizer_id,
                            trust_remote_code=True,
                        )
                        logger.info(f"✅ Tokenizer loaded from {tokenizer_id}")
                        break
                    except Exception as e:
                        logger.debug(f"Could not load tokenizer from {tokenizer_id}: {e}")
                        continue

            # Last resort: try processor from model
            if self.tokenizer is None:
                logger.warning("Could not load tokenizer from standard sources. Trying processor from model...")
                if hasattr(self.model, "processor"):
                    self.tokenizer = self.model.processor
                    logger.info("✅ Using model's processor as tokenizer")
                else:
                    logger.warning("⚠️ No tokenizer or processor available. Activation capture will not work.")

        except ImportError as e:
            logger.error(f"qwen_tts library required: pip install qwen-tts. Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load Qwen3-TTS: {e}")
            raise

    def get_talker_layer(self, layer_idx: int) -> nn.Module:
        """Get a specific layer from the Talker."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if not hasattr(self.model, "talker"):
            raise AttributeError("Model does not have 'talker' attribute")
        return self.model.talker.layers[layer_idx]

    def get_target_layers(self) -> list[int]:
        """Get target layer indices for phonetic SAE (layers 1-7 of Talker)."""
        return list(range(1, 8))  # Layers 1-7 (first 25% of 28-layer model)

    def get_layer_accessor(self):
        """Return a custom layer accessor function for this model.

        Qwen3-TTS layer structure (discovered via inspection):
        Qwen3TTSModel (qwen_tts wrapper)
          └── .model → Qwen3TTSForConditionalGeneration
              └── .talker → Qwen3TTSTalkerForConditionalGeneration (PEFT-wrapped)
                  └── .model → Qwen3TTSTalkerModel
                      └── .layers[i] ← TARGET LAYERS
        """
        def accessor(model, layer_idx: int) -> nn.Module:
            """Access Qwen3-TTS talker layers through the deep wrapper structure."""
            try:
                # Navigate through the known structure
                # Step 1: model.model (unwrap qwen_tts wrapper)
                if not hasattr(model, "model"):
                    raise AttributeError(f"{type(model).__name__} has no 'model' attribute")
                inner = model.model
                logger.debug(f"  Step 1: model.model → {type(inner).__name__}")

                # Step 2: model.model.talker (access talker component)
                if not hasattr(inner, "talker"):
                    raise AttributeError(f"{type(inner).__name__} has no 'talker' attribute")
                talker = inner.talker
                logger.debug(f"  Step 2: model.talker → {type(talker).__name__}")

                # Step 3: Unwrap PEFT if present (model.model.talker.base_model)
                if hasattr(talker, "base_model"):
                    talker = talker.base_model
                    logger.debug(f"  Step 3: unwrap PEFT → {type(talker).__name__}")

                # Step 4: model.model.talker.model (access talker model)
                if not hasattr(talker, "model"):
                    raise AttributeError(f"{type(talker).__name__} has no 'model' attribute")
                talker_model = talker.model
                logger.debug(f"  Step 4: talker.model → {type(talker_model).__name__}")

                # Step 5: Access layers (model.model.talker.model.layers[i])
                if not hasattr(talker_model, "layers"):
                    raise AttributeError(f"{type(talker_model).__name__} has no 'layers' attribute")

                layer = talker_model.layers[layer_idx]
                logger.info(f"✓ Found layer {layer_idx} via model.model.talker.model.layers[{layer_idx}]")
                return layer

            except (AttributeError, IndexError, TypeError) as e:
                logger.error(f"Cannot access layer {layer_idx}: {e}")
                raise

        return accessor

    def generate(
        self,
        text: str,
        language: str = "English",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
    ) -> dict:
        """Generate speech from text using Qwen3-TTS.

        Supports two modes:
        1. Voice Cloning (ref_audio + ref_text required)
        2. Custom Voice (speaker + instruct required)

        Parameters
        ----------
        text : str
            Text to synthesize
        language : str
            Language ("English", "Chinese", "Japanese", "Spanish", etc.)
        ref_audio : str, optional
            Path to reference audio for voice cloning
        ref_text : str, optional
            Transcription of reference audio
        speaker : str, optional
            Speaker name for custom voice (e.g., "Vivian")
        instruct : str, optional
            Voice instruction/style for custom voice

        Returns
        -------
        output : dict
            Dictionary with keys:
            - "waveform": numpy array of synthesized audio (shape: [1, num_samples])
            - "sample_rate": 24000
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            # Voice Cloning mode: requires ref_audio and ref_text
            if ref_audio is not None and ref_text is not None:
                logger.info(f"Generating voice clone for: {text[:50]}...")
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
                return {"waveform": wavs[0] if isinstance(wavs, list) else wavs, "sample_rate": sr}

            # Custom Voice mode: requires speaker and optionally instruct
            elif speaker is not None:
                logger.info(f"Generating custom voice ({speaker}) for: {text[:50]}...")
                wavs, sr = self.model.generate_custom_voice(
                    text=text,
                    language=language,
                    speaker=speaker,
                    instruct=instruct or "",
                )
                return {"waveform": wavs[0] if isinstance(wavs, list) else wavs, "sample_rate": sr}

            # Base mode: just text-to-speech without voice cloning
            else:
                logger.info(f"Generating base TTS for: {text[:50]}...")
                # Use the model's standard generate method for base mode
                if hasattr(self.model, "generate"):
                    wavs, sr = self.model.generate(text=text, language=language)
                    return {"waveform": wavs[0] if isinstance(wavs, list) else wavs, "sample_rate": sr}
                else:
                    raise ValueError("Model does not support base generation mode")

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def get_config(self) -> dict:
        """Get model configuration."""
        if self.model is None or not hasattr(self.model, "config"):
            return {}
        config = self.model.config
        return {
            "model_type": config.model_type if hasattr(config, "model_type") else "unknown",
            "hidden_size": config.hidden_size if hasattr(config, "hidden_size") else None,
            "num_layers": config.num_hidden_layers if hasattr(config, "num_hidden_layers") else None,
            "vocab_size": config.vocab_size if hasattr(config, "vocab_size") else None,
        }

    def __repr__(self) -> str:
        return (
            f"Qwen3TTSWrapper(model_path={self.model_path}, "
            f"device={self.device}, dtype={self.dtype})"
        )
