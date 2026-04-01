# Phase 1 Quick Start: Activation Capture with Forced Alignment

This guide walks you through **your first activation capture run** in under 30 minutes, validating the entire pipeline end-to-end.

---

## Prerequisites Checklist

Before starting, ensure:

```bash
# 1. Repository is cloned with submodules
git submodule status
# All should show ✓ status (no '-' prefix)

# 2. Environment is set up
source .venv/bin/activate
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')"

# 3. Models can be downloaded (requires internet)
python -c "from transformers import AutoModel; print('✅ HuggingFace access OK')"
```

If any step fails, see [Prerequisites](../README.md#prerequisites) and [Troubleshooting](../README.md#troubleshooting).

---

## Step 1: Verify Qwen3-ForcedAligner API (5 minutes)

The forced aligner is critical for phoneme alignment. Let's verify it works with your setup.

### Run API Inspection

```bash
python scripts/inspect_aligner_api.py --device cuda
```

**Expected output:**
```
======================================================================
QWEN3-FORCEDALIGNER ACTUAL API INSPECTION
======================================================================

Loading model...
✅ Model loaded!

======================================================================
MODEL METHODS & FUNCTIONS
======================================================================

Looking for key methods:
  ✅ align: (text, audio, language, sample_rate)
  ...

======================================================================
TESTING ACTUAL MODEL INPUT/OUTPUT
======================================================================

Testing inference...
  Testing: text + audio...
    ✅ Success!
    Output type: ModelOutput
    Output keys: [...align_boundaries, frame_to_phoneme...]
```

**Save the output:**
```bash
python scripts/inspect_aligner_api.py --device cuda > aligner_api_verification.txt
```

If you see errors, check:
- PyTorch version: `python -c "import torch; print(torch.__version__)"` (should be 2.1+)
- CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- GPU memory: `nvidia-smi` (should show ≥8GB free)

---

## Step 2: Extract Phoneme Inventory (3 minutes)

Verify what phonemes the model recognizes for your language:

```bash
# For English
python scripts/inspect_aligner.py --lang en

# For Mandarin
python scripts/inspect_aligner.py --lang zh

# For Cantonese
python scripts/inspect_aligner.py --lang yue
```

**Expected output:**
```
Extracted phoneme inventory for language: en
Found 43 phonemes:
['aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', ...]

Saved to: phoneme_inventory_en.json
```

This tells you:
- How many phonemes the model knows
- What phoneme names it uses (ARPAbet for English, Pinyin for Mandarin, etc.)
- Whether your dataset phonemes match the model's vocabulary

---

## Step 3: Prepare a Tiny Dataset (5 minutes)

Let's test with just **10 samples** before running on your full dataset.

### Create a test dataset in JSONL format:

```bash
mkdir -p data/test_dataset
```

**`data/test_dataset/test.jsonl`:**
```jsonl
{"text": "hello", "lang": "en", "audio_path": "data/test_dataset/sample_01.wav"}
{"text": "world", "lang": "en", "audio_path": "data/test_dataset/sample_02.wav"}
{"text": "你好", "lang": "zh", "audio_path": "data/test_dataset/sample_03.wav"}
```

**Get sample audio files:**

You can generate synthetic audio for testing (no real data needed yet):

```python
import torch
import torchaudio

# Generate 10 synthetic audio files (1 second each at 16kHz)
for i in range(10):
    # Random noise (replace with real audio when ready)
    audio = torch.randn(16000)  # 1 second at 16kHz
    torchaudio.save(f"data/test_dataset/sample_{i:02d}.wav", audio.unsqueeze(0), 16000)
```

Or use publicly available speech samples (e.g., from LibriSpeech or Common Voice).

---

## Step 4: Run Pilot Capture (10 minutes)

Now run activation capture on your 10-sample test set:

```bash
python scripts/capture_with_alignment.py \
  --model qwen3tts \
  --dataset custom \
  --dataset-csv data/test_dataset/test.jsonl \
  --lang en \
  --output data/pilot_activations \
  --num-samples 10 \
  --device cuda \
  --verbose
```

**What this does:**
1. Loads Qwen3-TTS model
2. Loads your 10 audio samples + text
3. Records activations from layers 1–7 during inference
4. Uses Qwen3-ForcedAligner to align phonemes to frames
5. Organizes output: `data/pilot_activations/layer_XX/phoneme_*.npy`

**Expected output:**
```
Loading Qwen3-TTS (0.6B)...
Initializing Qwen3-ForcedAligner (language: en)...
Processing 10 samples...
  [1/10] hello (156 frames)
    Phonemes: ['h', 'eh', 'l', 'ow']
    Activations captured: 7 layers × 1024 dims
  [2/10] world (142 frames)
    ...
Organizing activations by phoneme...
  layer_01/phoneme_h: 45 activations
  layer_01/phoneme_eh: 52 activations
  ...
✅ Capture complete! Output: data/pilot_activations/
```

**Check the output:**

```bash
# List captured phoneme activations
ls -lah data/pilot_activations/layer_01/

# Should show:
# phoneme_h.npy (shape: (N, 1024))
# phoneme_eh.npy (shape: (M, 1024))
# ... (one file per unique phoneme)

# View metadata
cat data/pilot_activations/phoneme_inventory.json
# { "h": 45, "eh": 52, "l": 58, "ow": 40 }

# View frame-level labels
head -5 data/pilot_activations/frame_labels.jsonl
```

**Inspect activation shapes:**

```python
import numpy as np

# Load one phoneme's activations
h_activations = np.load("data/pilot_activations/layer_01/phoneme_h.npy")
print(f"Shape: {h_activations.shape}")  # Should be (N, 1024)
print(f"Mean: {h_activations.mean():.4f}, Std: {h_activations.std():.4f}")
```

---

## Step 5: Verify Output Structure (2 minutes)

Check that everything looks correct:

```bash
# Verify directory structure
find data/pilot_activations -type f | head -20

# Should show:
# data/pilot_activations/layer_01/phoneme_h.npy
# data/pilot_activations/layer_01/phoneme_eh.npy
# data/pilot_activations/layer_01/phoneme_l.npy
# data/pilot_activations/layer_01/phoneme_ow.npy
# data/pilot_activations/layer_02/phoneme_h.npy
# ...
# data/pilot_activations/phoneme_inventory.json
# data/pilot_activations/frame_labels.jsonl

# Verify phoneme inventory is populated
python -c "
import json
with open('data/pilot_activations/phoneme_inventory.json') as f:
    inv = json.load(f)
    print(f'Total unique phonemes: {len(inv)}')
    print(f'Top 10: {dict(sorted(inv.items(), key=lambda x: x[1], reverse=True)[:10])}')
"

# Verify frame labels are aligned
python -c "
import json
with open('data/pilot_activations/frame_labels.jsonl') as f:
    lines = f.readlines()[:3]
    for line in lines:
        label = json.loads(line)
        print(f\"Sample: {label['text']}, Frames: {label['num_frames']}, Phonemes: {len(label['phoneme_indices'])}\")
"
```

---

## Step 6: Next: Scale to Full Dataset

Once the pilot succeeds, you're ready to capture your full dataset:

```bash
# For 50,000 samples (estimated: 2–3 hours on RTX 3090)
python scripts/capture_with_alignment.py \
  --model qwen3tts \
  --dataset custom \
  --dataset-csv data/your_full_dataset.jsonl \
  --lang en \
  --output data/activations/qwen3tts \
  --num-samples 50000 \
  --device cuda \
  --batch-size 4
```

**Monitor progress:**
```bash
# In another terminal, watch output size
watch -n 5 'du -sh data/activations/qwen3tts'

# Should grow: 0 → 100 MB → 1 GB → ... → ~100 GB
```

**Troubleshooting capture failures:**

See [Troubleshooting: Activation Capture](../docs/PHONEME_ALIGNMENT.md#troubleshooting).

---

## Step 7: Validate for Phase 2 (SAE Training)

Before training a SAE, verify the activation data is suitable:

```python
import numpy as np
import json
from pathlib import Path

activation_dir = Path("data/pilot_activations")

# 1. Count total samples per layer
for layer_dir in sorted(activation_dir.glob("layer_*")):
    npy_files = list(layer_dir.glob("phoneme_*.npy"))
    total_samples = sum(np.load(f).shape[0] for f in npy_files)
    print(f"{layer_dir.name}: {len(npy_files)} phonemes, {total_samples} total samples")

# 2. Check activation statistics
with open(activation_dir / "phoneme_inventory.json") as f:
    inventory = json.load(f)
    print(f"\nPhoneme coverage: {len(inventory)} unique phonemes")
    print(f"Total observations: {sum(inventory.values())}")

    # Check for imbalanced phonemes
    max_count = max(inventory.values())
    min_count = min(inventory.values())
    print(f"Max/min observations: {max_count}/{min_count} ({max_count/min_count:.1f}x imbalance)")

# 3. Verify frame-to-phoneme alignment
with open(activation_dir / "frame_labels.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        import json
        label = json.loads(line)
        text = label["text"]
        phonemes = label["phoneme_indices"]
        frames = label["num_frames"]
        print(f"\n{text}: {frames} frames → {len(set(phonemes))} unique phonemes")
```

**Expected results:**
- ✅ Each layer has activations for multiple phonemes
- ✅ Total samples ≥ 1,000 (more is better)
- ✅ Imbalance ratio < 10:1 (some imbalance is OK)
- ✅ Frame-to-phoneme mappings are non-trivial (multiple phonemes per sample)

---

## Troubleshooting Quick Reference

| Issue | Check | Fix |
|-------|-------|-----|
| **Model download timeout** | `HF_HOME=/path/to/cache python scripts/inspect_aligner_api.py` | Set HF_HOME, check internet |
| **CUDA out of memory** | `nvidia-smi` | Reduce `--batch-size`, use `--device cpu` for testing |
| **No phonemes detected** | Check phoneme_inventory.json is non-empty | Verify audio quality, text encoding |
| **Misaligned boundaries** | Check frame_labels.jsonl for phoneme_indices | Run `analyze_aligner_inference.py` to debug |
| **Wrong phoneme set** | Compare `phoneme_inventory.json` with `docs/PHONEME_SETS_SOURCES.md` | May need custom phoneme mapping |

---

## Next Steps

Once pilot capture succeeds:

1. **Scale to full dataset** — Run full capture with all 50K+ samples
2. **Phase 2: SAE Training** — See [SAE Training Guide](../README.md#training)
3. **Phase 3: Feature Mapping** — Analyze which SAE features correlate with specific phonemes

For detailed setup & API verification, see:
- [Qwen3-ForcedAligner Inference](QWEN3_FORCEDALIGNER_INFERENCE.md)
- [Phoneme Alignment Guide](PHONEME_ALIGNMENT.md)
- [API Discovery Guide](ACTUAL_API_DISCOVERY.md)

---

## Questions?

- **API issues?** → `python scripts/inspect_aligner_api.py --device cpu` (test on CPU first)
- **Alignment problems?** → Check `frame_labels.jsonl` structure
- **Performance?** → Profile with `scripts/analyze_aligner_inference.py --profile`

Happy capturing! 🎯
