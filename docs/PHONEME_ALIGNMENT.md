# Phoneme Alignment with Qwen3-ForcedAligner

This guide explains how to use the Qwen3-ForcedAligner to perform phoneme-to-frame alignment for enhanced SAE feature analysis.

---

## Overview

**Why Phoneme Alignment?**

Phoneme alignment maps text/phoneme sequences to audio frames, enabling:
- ✅ Frame-level phoneme labels
- ✅ Direct correlation between SAE features and phonemes
- ✅ Identification of monosemantic phonetic features
- ✅ Language-specific feature analysis

**Before Alignment:**
```
Audio frames: [frame_0, frame_1, ..., frame_N]
             ↓
Activations: [act_0, act_1, ..., act_N]
             ↓
SAE Features: [features_0, features_1, ..., features_N]
             ↓
Question: Which phoneme does feature_i encode?
```

**After Alignment:**
```
Audio frames: [frame_0, frame_1, ..., frame_N]
             ↓
Phonemes:     ["m",     "ə",      ..., "l"]        ← from aligner
             ↓
Activations: [act_0, act_1, ..., act_N]
             ↓
SAE Features: [features_0, features_1, ..., features_N]
             ↓
Answer: feature_i activates most for phoneme "m"
```

---

## Installation

The Qwen3-ForcedAligner requires `transformers`:

```bash
pip install transformers torchaudio
```

---

## Quick Start

### Step 1: Run Aligned Capture

```bash
python scripts/capture_with_alignment.py \
  --model qwen3tts \
  --dataset-file data/out.jsonl \
  --lang en \
  --output data/activations_aligned \
  --num-samples 100 \
  --device cuda
```

**Output Structure:**
```
data/activations_aligned/en/
├── layer_01/
│   ├── phoneme_m.npy          # Activations when phoneme="m"
│   ├── phoneme_ə.npy
│   ├── phoneme_l.npy
│   └── ...
├── layer_02/
│   └── ...
├── phoneme_inventory.json      # {"m": 150, "ə": 200, "l": 120, ...}
└── frame_labels.jsonl          # Sample IDs + phoneme sequences
```

### Step 2: Analyze Per-Phoneme Activations

```python
import numpy as np
from pathlib import Path

# Load phoneme-organized activations
aligned_dir = Path("data/activations_aligned/en/layer_01")

phoneme_activations = {}
for phoneme_file in aligned_dir.glob("phoneme_*.npy"):
    phoneme = phoneme_file.stem.replace("phoneme_", "")
    activations = np.load(phoneme_file)  # shape: (n_frames, d_model)
    phoneme_activations[phoneme] = activations
    print(f"Phoneme '{phoneme}': {activations.shape[0]} frames")

# Output:
# Phoneme 'm': 150 frames
# Phoneme 'ə': 200 frames
# Phoneme 'l': 120 frames
```

### Step 3: Correlate SAE Features with Phonemes

```python
from src.sae import TopKSAE
import torch

# Load trained SAE
sae = TopKSAE.load("checkpoints/sae_qwen3tts/sae_final.pt")

# For each phoneme, compute which SAE features activate most
feature_phoneme_correlation = {}

for phoneme, activations_np in phoneme_activations.items():
    acts = torch.from_numpy(activations_np).float().to("cuda")

    # Encode activations
    z_sparse, z_full = sae.encode(acts)

    # z_sparse: (n_frames, d_sae) with exactly K non-zero per frame
    # Compute mean activation per feature
    mean_activation = z_sparse.mean(dim=0)

    feature_phoneme_correlation[phoneme] = mean_activation.cpu().numpy()

# Find which feature activates most for each phoneme
for feature_idx in range(sae.config.d_sae):
    top_phoneme = max(
        feature_phoneme_correlation.items(),
        key=lambda x: x[1][feature_idx]
    )[0]
    print(f"Feature {feature_idx} → phoneme '{top_phoneme}'")

# Output:
# Feature 0 → phoneme 'm'
# Feature 1 → phoneme 'ə'
# Feature 2 → phoneme 'l'
# ...
```

---

## Supported Languages

The Qwen3-ForcedAligner supports three languages with specific phoneme inventories.

**⚠️ Important Note on Phoneme Sets:**

The phoneme sets listed below are based on standard inventories for each language. **The actual Qwen3-ForcedAligner model may use different phoneme sets.**

The implementation automatically attempts to extract the actual phoneme inventory from the loaded model. If extraction fails, it falls back to the standard sets below.

**To verify the actual phoneme set your model uses:**
```python
from src.alignment import QwenForcedAligner

aligner = QwenForcedAligner(language="en")
actual_phonemes = aligner.get_phoneme_inventory()
print(f"Model uses {len(actual_phonemes)} phonemes: {actual_phonemes}")
```

---

### English ("en")
**Phoneme Set:** ARPAbet (43 phonemes + silence)
```
Consonants: b, ch, d, dh, f, g, hh, jh, k, l, m, n, ng, p, r, s, sh, t, th, v, w, y, z, zh
Vowels: aa, ae, ah, ao, aw, ay, eh, er, ey, ih, iy, oh, ow, oy, uh, uw
Silence: pau, sil
```

**Example:**
```python
aligner = QwenForcedAligner(language="en")
alignment = aligner.align(
    text="hello",
    audio_path="audio/hello.wav"
)
# phonemes: ["hh", "eh", "l", "ow"]
```

### Mandarin Chinese ("zh")
**Phoneme Set:** Pinyin initials + finals (40+ phonemes)
```
Initials: b, p, m, f, d, t, n, l, g, k, h, j, q, x, zh, ch, sh, r, z, c, s
Finals: a, e, i, o, u, ü, ai, ei, ao, ou, an, en, ang, eng, ong, ...
```

**Example:**
```python
aligner = QwenForcedAligner(language="zh")
alignment = aligner.align(
    text="你好",  # "ni hao"
    audio_path="audio/nihao.wav"
)
# phonemes: ["n", "i", "h", "ao"]
```

### Cantonese ("yue")
**Phoneme Set:** Jyutping initials + finals (40+ phonemes)
```
Initials: p, ph, m, f, t, th, n, l, k, kh, ng, h, gw, kw, z, c, s, j
Finals: a, e, i, o, u, oe, ai, ei, oi, ou, au, an, en, on, ...
```

**Example:**
```python
aligner = QwenForcedAligner(language="yue")
alignment = aligner.align(
    text="你好",  # "nei hou"
    audio_path="audio/nehou.wav"
)
# phonemes: ["n", "ei", "h", "ou"]
```

---

## API Reference

### QwenForcedAligner

```python
from src.alignment import QwenForcedAligner

aligner = QwenForcedAligner(device="cuda", language="en")
```

**Methods:**

```python
# Perform alignment
alignment = aligner.align(
    text: str,                      # Input text
    audio_path: Optional[str],      # Path to audio file
    audio_tensor: Optional[torch.Tensor],  # Or raw audio tensor
    sample_rate: int = 16000        # Audio sample rate
) -> PhonemeAlignment

# Get phoneme inventory
phonemes = aligner.get_phoneme_inventory()  # Returns: list[str]

# Validate phonemes
valid = aligner.validate_phonemes(["m", "ə", "l"])  # Returns: bool
```

### PhonemeAlignment (Return Object)

```python
@dataclass
class PhonemeAlignment:
    phonemes: list[str]                    # ["m", "ə", "l", ...]
    frame_boundaries: list[tuple[int, int]]  # [(0, 10), (10, 25), (25, 35), ...]
    frame_to_phoneme: list[int]            # [0, 0, ..., 1, 1, ..., 2, 2, ...]
    duration_ms: float                     # Total duration in milliseconds
```

### LanguageSpecificAligner

For batch processing multiple languages:

```python
from src.alignment import LanguageSpecificAligner

aligner_multi = LanguageSpecificAligner(
    languages=["en", "zh", "yue"],
    device="cuda"
)

# Get available languages
langs = aligner_multi.get_available_languages()  # ["en", "zh", "yue"]

# Align with auto-selected aligner
alignment = aligner_multi.align(
    text="你好",
    audio_path="audio/hello.wav",
    language="zh"
)
```

---

## Full Workflow Example

```python
from src.alignment import QwenForcedAligner
from src.hooks import ActivationHook
from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper
from src.sae import TopKSAE
import torch
import numpy as np

# 1. Initialize aligner
aligner = QwenForcedAligner(device="cuda", language="en")

# 2. Load model
wrapper = Qwen3TTSWrapper(device="cuda")
model = wrapper.model
hook = ActivationHook(model, layer_indices=[1, 2, 3])
hook.attach("mlp")

# 3. Process single sample
text = "hello world"
audio_path = "audio/hello.wav"

# Get phoneme alignment
alignment = aligner.align(text=text, audio_path=audio_path)
print(f"Phonemes: {alignment.phonemes}")
# Output: ["hh", "eh", "l", "ow", "w", "er", "l", "d"]

# Get activations
input_ids = wrapper.tokenizer.encode(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    _ = model(input_ids)

activations = hook.collect()  # dict[layer_idx, tensor]

# 4. Organize by phoneme
phoneme_acts = {}
for layer_idx, acts in activations.items():
    if acts.dim() == 1:
        acts = acts.unsqueeze(0)

    for frame_idx, phoneme_idx in enumerate(alignment.frame_to_phoneme):
        phoneme = alignment.phonemes[phoneme_idx]
        if phoneme not in phoneme_acts:
            phoneme_acts[phoneme] = {}
        if layer_idx not in phoneme_acts[phoneme]:
            phoneme_acts[phoneme][layer_idx] = []

        phoneme_acts[phoneme][layer_idx].append(acts[frame_idx].detach().cpu().numpy())

# 5. Train SAE and correlate
sae = TopKSAE.load("checkpoints/sae_qwen3tts/sae_final.pt")

for phoneme, layer_acts in phoneme_acts.items():
    acts_np = np.concatenate(layer_acts.get(1, []), axis=0)
    acts = torch.from_numpy(acts_np).float().to("cuda")
    z_sparse, _ = sae.encode(acts)

    # Which features activate for this phoneme?
    top_features = np.argsort(-z_sparse.mean(dim=0).cpu().numpy())[:5]
    print(f"Phoneme '{phoneme}' → Top features: {top_features}")
```

---

## Troubleshooting

### Alignment Fails for Cantonese

**Issue:** Low alignment accuracy for Cantonese samples

**Solutions:**
1. Verify audio quality (16kHz mono recommended)
2. Check text matches audio (no pauses, background noise, etc.)
3. Try English ("en") or Mandarin ("zh") first to verify setup
4. Use fallback to standard capture (without alignment)

### Memory Issues

**Issue:** CUDA out of memory during alignment

**Solutions:**
```bash
# Use CPU for alignment, GPU for model
python scripts/capture_with_alignment.py \
  --model qwen3tts \
  --device cuda \
  --num-samples 50  # Reduce batch size
```

### Model Download Hangs

**Issue:** HuggingFace model download times out

**Solutions:**
```bash
# Pre-download model manually
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B

# Or set cache directory
export HF_HOME=/path/to/cache
python scripts/capture_with_alignment.py ...
```

---

## Next Steps

1. **Run aligned capture** on your full dataset
2. **Analyze per-phoneme activations** (see Phase 3)
3. **Train per-language SAEs** (optional)
4. **Correlate SAE features with phonemes** (Phase 3 notebook)
5. **Intervene on specific phonemic features** (Phase 4)

---

## References

- [Qwen3-ForcedAligner on HuggingFace](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
- [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) (alternative)
- [TIMIT Phoneme Set](https://en.wikipedia.org/wiki/TIMIT)
- [Pinyin (Mandarin)](https://en.wikipedia.org/wiki/Pinyin)
- [Jyutping (Cantonese)](https://en.wikipedia.org/wiki/Jyutping)
