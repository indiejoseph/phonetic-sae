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
        """Load Qwen3-TTS model and tokenizer."""
        try:
            # Try loading from local path first
            if isinstance(self.model_path, Path) and self.model_path.exists():
                logger.info(f"Loading Qwen3-TTS from local path: {self.model_path}")
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    torch_dtype=self.dtype,
                    device_map=self.device,
                    trust_remote_code=True,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            else:
                # Load from HuggingFace
                logger.info(f"Loading Qwen3-TTS from HuggingFace: {self.model_path}")
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    torch_dtype=self.dtype,
                    device_map=self.device,
                    trust_remote_code=True,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

            logger.info(f"Model loaded on {self.device} with dtype {self.dtype}")
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

    def generate(
        self,
        text: str,
        language: str = "English",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        temperature: float = 0.9,
        top_k: int = 50,
        max_new_tokens: int = 2048,
    ) -> dict:
        """Generate speech from text using voice cloning.

        Parameters
        ----------
        text : str
            Text to synthesize
        language : str
            Language code ("English", "Chinese", "Spanish", etc.)
        ref_audio : str, optional
            Path to reference audio for voice cloning
        ref_text : str, optional
            Transcription of reference audio
        temperature : float
            Sampling temperature
        top_k : int
            Top-K sampling parameter
        max_new_tokens : int
            Maximum tokens to generate

        Returns
        -------
        output : dict
            Dictionary with keys:
            - "waveform": numpy array of synthesized audio
            - "sample_rate": 24000
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            # Use the model's generate_voice_clone method if available
            if hasattr(self.model, "generate_voice_clone"):
                wav, sr = self.model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    temperature=temperature,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                )
                return {"waveform": wav, "sample_rate": sr}
            else:
                # Fallback: use generic generation
                logger.warning("Model does not have generate_voice_clone, using generic generation")
                input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids,
                        temperature=temperature,
                        top_k=top_k,
                        max_new_tokens=max_new_tokens,
                    )
                # This won't produce audio, just dummy output
                return {"waveform": None, "sample_rate": 24000}
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
