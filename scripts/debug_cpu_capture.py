#!/usr/bin/env python3
"""CPU debug script for activation capture.

Validates the full pipeline on CPU with a small number of samples:
1. Load model on CPU (float32)
2. Load combined dataset (audio + codec from save_to_disk())
3. Run generate_voice_clone() with proper ref_audio
4. Verify hooks capture activations from Talker layers

Usage:
    PYTHONPATH="." python scripts/debug_cpu_capture.py \
        --model qwen3tts \
        --dataset data/combined_yue \
        --num-samples 1 \
        --max-tokens 64
"""

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_samples(dataset_path: str, num_samples: int):
    """Load samples from combined dataset (created by merge_dataset.py).

    Dataset was saved with ds.save_to_disk(), fields:
        audio       - dict with "array" (np.ndarray) and "sampling_rate" (16000)
        text        - transcription string
        codec       - list of frames, each frame is 16 codebook values
        lang        - "en", "zh", "yue"
        phone       - phoneme string (from original HF dataset)
        spk_emb     - speaker embedding (from original HF dataset)
        + other original fields
    """
    from datasets import load_from_disk

    logger.info(f"Loading dataset from {dataset_path}...")
    ds = load_from_disk(dataset_path)
    logger.info(f"Dataset loaded: {len(ds)} samples")

    # Pick num_samples + 1 (first as reference speaker)
    n = min(num_samples + 1, len(ds))
    samples = []

    tmp_dir = Path(tempfile.gettempdir()) / "phonetic_sae_debug"
    tmp_dir.mkdir(exist_ok=True)

    for i in range(n):
        row = ds[i]
        audio = row["audio"]

        # Save audio to temp wav for generate_voice_clone()
        wav_path = str(tmp_dir / f"sample_{i}.wav")
        sf.write(wav_path, audio["array"], audio["sampling_rate"])

        samples.append({
            "audio_path": wav_path,
            "text": row["text"],
            "lang": row.get("lang", ""),
            "codec": row.get("codec"),
            "phone": row.get("phone", ""),
            "sample_rate": audio["sampling_rate"],
        })

    logger.info(f"Prepared {len(samples)} samples (1 ref + {len(samples)-1} synthesis)")
    for i, s in enumerate(samples):
        role = "REF" if i == 0 else "SYN"
        logger.info(f"  [{role}] lang={s['lang']} text={s['text'][:50]}...")

    return samples


def debug_qwen3tts(args):
    """Debug Qwen3-TTS activation capture on CPU."""
    from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper
    from src.hooks.activation_hook import ActivationHook

    # Step 1: Load model on CPU
    logger.info("=" * 60)
    logger.info("Loading Qwen3-TTS on CPU (this may take a minute)...")
    logger.info("=" * 60)

    wrapper = Qwen3TTSWrapper(
        device="cpu",
        dtype=torch.float32,
    )
    model = wrapper.model
    target_layers = wrapper.get_target_layers()
    layer_accessor = wrapper.get_layer_accessor()

    logger.info(f"Model loaded. Target layers: {target_layers}")

    # Step 2: Load samples from combined dataset
    logger.info("=" * 60)
    logger.info("Loading samples from combined dataset...")
    logger.info("=" * 60)

    samples = load_samples(args.dataset, args.num_samples)

    # First sample = reference speaker for voice cloning
    ref = samples[0]
    ref_audio_path = ref["audio_path"]
    ref_text = ref["text"]
    logger.info(f"Reference: {ref_audio_path} ({ref['sample_rate']}Hz)")
    logger.info(f"  text: {ref_text[:80]}...")

    # Remaining = synthesis targets (if only 1 sample, reuse as both)
    synth_samples = samples[1:] if len(samples) > 1 else samples[:1]

    # Step 3: Attach hooks
    logger.info("=" * 60)
    logger.info("Attaching activation hooks...")
    logger.info("=" * 60)

    hook = ActivationHook(
        model,
        layer_indices=target_layers,
        layer_accessor=layer_accessor,
        device="cpu",
        dtype=torch.float32,
    )
    hook.attach("mlp")

    # Step 4: Run inference
    # Map lang code to Qwen3-TTS language name
    # Note: Qwen3-TTS has no "Cantonese" — yue falls under "Chinese"
    lang_map = {"en": "English", "zh": "Chinese", "yue": "Chinese"}

    logger.info("=" * 60)
    logger.info(f"Running inference (max_tokens={args.max_tokens})...")
    logger.info("=" * 60)

    for i, sample in enumerate(synth_samples):
        text = sample["text"]
        if len(text) > 200:
            text = text[:200]

        language = lang_map.get(sample["lang"], "Auto")
        logger.info(f"\n[{i+1}/{len(synth_samples)}] lang={language} text={text[:60]}...")

        try:
            with torch.no_grad():
                wavs, sr = wrapper.model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=ref_audio_path,
                    ref_text=ref_text,
                    x_vector_only_mode=True,
                    max_new_tokens=args.max_tokens,
                    do_sample=True,
                    temperature=0.9,
                    top_k=50,
                )

            # Collect activations
            activations = hook.collect()

            logger.info(f"  Generation OK, sr={sr}")
            logger.info(f"  Activations from {len(activations)} layers:")
            for layer_idx, tensor in sorted(activations.items()):
                logger.info(
                    f"    Layer {layer_idx}: shape={tuple(tensor.shape)}, "
                    f"mean={tensor.float().mean():.4f}, std={tensor.float().std():.4f}"
                )

            # Save debug output
            if args.output:
                out_dir = Path(args.output)
                out_dir.mkdir(parents=True, exist_ok=True)

                for layer_idx, tensor in activations.items():
                    torch.save(tensor, out_dir / f"debug_layer{layer_idx}_sample{i}.pt")

                if isinstance(wavs, list) and len(wavs) > 0:
                    sf.write(str(out_dir / f"debug_sample{i}.wav"), wavs[0], sr)

                logger.info(f"  Saved to {out_dir}")

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

            # Check if hooks captured anything before failure
            activations = hook.collect()
            if any(t.numel() > 0 for t in activations.values()):
                logger.info("  (Hooks captured activations before failure:)")
                for layer_idx, tensor in sorted(activations.items()):
                    if tensor.numel() > 0:
                        logger.info(f"    Layer {layer_idx}: shape={tuple(tensor.shape)}")

    hook.detach()

    logger.info("\n" + "=" * 60)
    logger.info("CPU Debug Complete")
    logger.info("=" * 60)


def debug_cosyvoice2(args):
    """Debug CosyVoice2 activation capture on CPU."""
    logger.info("CosyVoice2 CPU debug not yet implemented")


def main():
    parser = argparse.ArgumentParser(description="CPU debug for activation capture")
    parser.add_argument("--model", choices=["qwen3tts", "cosyvoice2"], default="qwen3tts")
    parser.add_argument("--dataset", type=str, default="data/combined_yue",
                        help="Path to combined dataset (from save_to_disk())")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of synthesis samples (keep small for CPU)")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Max new tokens to generate (keep small for CPU)")
    parser.add_argument("--output", type=Path, default=Path("data/debug_activations"))
    args = parser.parse_args()

    logger.info(f"CPU Debug: model={args.model}, dataset={args.dataset}, samples={args.num_samples}")

    if args.model == "qwen3tts":
        debug_qwen3tts(args)
    else:
        debug_cosyvoice2(args)


if __name__ == "__main__":
    main()
