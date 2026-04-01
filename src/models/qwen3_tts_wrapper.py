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
        dtype: torch.dtype = torch.float16,
    ):
        self.model_path = Path(model_path) if isinstance(model_path, str) else model_path
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

        Qwen3-TTS complex wrapper structure:
        - Outer: Qwen3TTSModel (from qwen_tts)
        - Middle: model.model may be PEFT-wrapped
        - Base: model.model.base_model is actual Qwen3TTSForConditionalGeneration
        - Layers: base_model.transformer.h[i] or similar
        """
        def accessor(model, layer_idx: int) -> nn.Module:
            """Access Qwen3-TTS talker layers through wrapper layers."""
            try:
                # Unwrap the model step by step
                model_to_inspect = model
                unwrap_steps = [
                    ("model", "unwrap qwen_tts wrapper"),
                    ("base_model", "unwrap PEFT wrapper"),
                ]

                for attr, desc in unwrap_steps:
                    if hasattr(model_to_inspect, attr):
                        model_to_inspect = getattr(model_to_inspect, attr)
                        logger.debug(f"  → {desc}: {type(model_to_inspect).__name__}")

                # Try different possible layer structures on unwrapped model
                # Qwen2-style: model.transformer.h[i]
                if hasattr(model_to_inspect, "transformer") and hasattr(model_to_inspect.transformer, "h"):
                    layer = model_to_inspect.transformer.h[layer_idx]
                    logger.info(f"✓ Accessing layers via model.transformer.h[{layer_idx}]")
                    return layer

                # HuggingFace style: model.model.layers[i]
                elif hasattr(model_to_inspect, "model") and hasattr(model_to_inspect.model, "layers"):
                    layer = model_to_inspect.model.layers[layer_idx]
                    logger.info(f"✓ Accessing layers via model.model.layers[{layer_idx}]")
                    return layer

                # Direct access: model.layers[i]
                elif hasattr(model_to_inspect, "layers"):
                    layer = model_to_inspect.layers[layer_idx]
                    logger.info(f"✓ Accessing layers via model.layers[{layer_idx}]")
                    return layer

                else:
                    # As last resort, check _modules dict
                    if hasattr(model_to_inspect, "_modules"):
                        layer_modules = [k for k in model_to_inspect._modules.keys()
                                       if k.startswith("layer") or k.startswith("h")]
                        if layer_modules:
                            logger.warning(f"Found layer-like modules: {layer_modules[:5]}")

                    raise AttributeError(
                        f"Cannot find layer structure in {type(model_to_inspect).__name__}. "
                        f"Tried: transformer.h, model.layers, layers"
                    )

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
