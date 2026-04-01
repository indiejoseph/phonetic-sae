"""Wrapper for CosyVoice2-0.5B with activation hook support.

Provides an interface to load and run CosyVoice2 inference with
automatic activation capture from the LLM backbone's phonetic layers.

Architecture:
    - LLM Backbone: Qwen2.5-0.5B (d_model=896, 24 layers)
    - Causal Flow Matching: Diffusion-based mel-spectrogram generation
    - HiFT Vocoder: GAN-based waveform synthesis with NSF

Target layers for phonetic SAE: Layers 1–6 of LLM (first 25%)
Hook point: model.llm.layers[i].mlp (post-activation)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CosyVoice2Wrapper:
    """Wrapper for CosyVoice2-0.5B with activation hook support.

    Parameters
    ----------
    model_dir : str or Path
        Path to CosyVoice2 model directory
    device : str
        Device to load model on ("cuda", "cpu", etc.)
    dtype : torch.dtype
        Model dtype (torch.float16 for memory efficiency)
    """

    def __init__(
        self,
        model_dir: str | Path = "pretrained_models/CosyVoice2-0.5B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir
        self.device = device
        self.dtype = dtype
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load CosyVoice2 model."""
        try:
            # CosyVoice2 uses a custom model class, so we need to import it
            logger.info(f"Loading CosyVoice2 from: {self.model_dir}")

            # Try importing from third_party/CosyVoice2
            import sys

            cosyvoice_path = Path(__file__).parent.parent.parent / "third_party" / "CosyVoice2"
            if cosyvoice_path.exists():
                sys.path.insert(0, str(cosyvoice_path))

            try:
                from cosyvoice.utils.cosyvoice_model import CosyVoiceModel

                self.model = CosyVoiceModel.from_pretrained(
                    str(self.model_dir),
                    device=self.device,
                )
                logger.info(f"Model loaded on {self.device}")
            except ImportError:
                logger.warning(
                    "CosyVoiceModel not found. Make sure third_party/CosyVoice2 is set up."
                )
                self.model = None

        except Exception as e:
            logger.error(f"Failed to load CosyVoice2: {e}")
            raise

    def get_llm_layer(self, layer_idx: int) -> nn.Module:
        """Get a specific layer from the LLM backbone."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if not hasattr(self.model, "llm"):
            raise AttributeError("Model does not have 'llm' attribute")
        return self.model.llm.layers[layer_idx]

    def get_target_layers(self) -> list[int]:
        """Get target layer indices for phonetic SAE (layers 1-6 of LLM)."""
        return list(range(1, 7))  # Layers 1-6 (first 25% of 24-layer model)

    def get_layer_accessor(self):
        """Return a custom layer accessor function for this model.

        CosyVoice2 has layers at: model.llm.layers[i].mlp
        """
        def accessor(model, layer_idx: int) -> nn.Module:
            """Access CosyVoice2 LLM layers."""
            try:
                # CosyVoice2 structure
                if hasattr(model, "llm") and hasattr(model.llm, "layers"):
                    return model.llm.layers[layer_idx]
                else:
                    raise AttributeError(
                        f"Model {type(model).__name__} does not have 'llm.layers'. "
                        f"Expected: CosyVoice2Model with llm.layers[{layer_idx}]"
                    )
            except (AttributeError, IndexError) as e:
                logger.error(
                    f"Cannot access layer {layer_idx}: {e}. "
                    f"Available attributes: {dir(model)}"
                )
                raise

        return accessor

    def generate(
        self,
        tts_text: str,
        prompt_text: str,
        prompt_wav: Optional[str] = None,
        zero_shot_spk_id: str = "",
        stream: bool = False,
        speed: float = 1.0,
    ) -> dict:
        """Generate speech from text using zero-shot voice cloning.

        Parameters
        ----------
        tts_text : str
            Text to synthesize
        prompt_text : str
            Speaker prompt text (for zero-shot cloning)
        prompt_wav : str, optional
            Path to speaker reference audio
        zero_shot_spk_id : str
            Pre-saved speaker ID (if empty, uses prompt_wav)
        stream : bool
            Whether to use streaming mode
        speed : float
            Speaking rate multiplier

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
            # Use the model's inference_zero_shot method if available
            if hasattr(self.model, "inference_zero_shot"):
                # CosyVoice2 returns a generator of dict objects
                output_gen = self.model.inference_zero_shot(
                    tts_text=tts_text,
                    prompt_text=prompt_text,
                    prompt_wav=prompt_wav or "",
                    zero_shot_spk_id=zero_shot_spk_id,
                    stream=stream,
                    speed=speed,
                )
                # Collect output
                output_list = list(output_gen)
                if output_list:
                    last_output = output_list[-1]
                    return {
                        "waveform": last_output.get("tts_speech"),
                        "sample_rate": 24000,
                    }
                else:
                    raise RuntimeError("Generation produced no output")
            else:
                logger.error("Model does not have inference_zero_shot method")
                raise RuntimeError("Model method not available")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def get_config(self) -> dict:
        """Get model configuration."""
        if self.model is None:
            return {}
        # Extract LLM config if available
        try:
            if hasattr(self.model, "llm") and hasattr(self.model.llm, "config"):
                config = self.model.llm.config
                return {
                    "model_type": config.model_type if hasattr(config, "model_type") else "unknown",
                    "hidden_size": config.hidden_size if hasattr(config, "hidden_size") else None,
                    "num_layers": config.num_hidden_layers if hasattr(config, "num_hidden_layers") else None,
                    "vocab_size": config.vocab_size if hasattr(config, "vocab_size") else None,
                }
        except Exception as e:
            logger.warning(f"Could not extract config: {e}")
        return {}

    def __repr__(self) -> str:
        return (
            f"CosyVoice2Wrapper(model_dir={self.model_dir}, "
            f"device={self.device}, dtype={self.dtype})"
        )
