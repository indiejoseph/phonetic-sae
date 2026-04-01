# Phase 1 Activation Capture — Quick Start

**Status:** ✅ Ready to run
**Time to first result:** ~10 minutes (test) to ~3 hours (full)

---

## 1️⃣ Validate Installation (10 minutes)

```bash
cd phonetic-sae
source .venv/bin/activate
python scripts/test_activation_capture.py
```

**Expected output:** `✅ ACTIVATION CAPTURE PIPELINE WORKS!`

If this works, move to step 2. If not, see `docs/ACTIVATION_CAPTURE_TROUBLESHOOTING.md`

---

## 2️⃣ Quick Test (30 minutes)

Capture 100 samples to verify the full pipeline:

```bash
python scripts/full_capture.py \
    --model qwen3tts \
    --dataset pilot \
    --output data/activations_test \
    --num-samples 100 \
    --device cuda
```

**Output:** Files in `data/activations_test/layer_01.npy`, etc.

---

## 3️⃣ Full Capture (2-3 hours)

Capture 50,000 samples for SAE training:

```bash
python scripts/full_capture.py \
    --model qwen3tts \
    --dataset libritts \
    --output data/activations_qwen3tts \
    --num-samples 50000 \
    --device cuda
```

**Output:** ~100 GB in `data/activations_qwen3tts/`

---

## 4️⃣ With Phoneme Alignment (Optional, slower)

Capture 50,000 samples with phoneme alignment:

```bash
python scripts/capture_with_alignment.py \
    --model qwen3tts \
    --dataset-file data/out.jsonl \
    --lang en \
    --output data/activations_aligned \
    --num-samples 50000 \
    --device cuda
```

**Output:** Phoneme-organized activations in `data/activations_aligned/en/`

---

## Available Options

### Model
```
--model qwen3tts      # Qwen3-TTS (default)
--model cosyvoice2    # CosyVoice2 (alternative)
```

### Dataset
```
--dataset pilot       # 50 test samples
--dataset libritts    # LibriTTS-R (auto-download)
--dataset custom      # Custom CSV/JSONL
  --dataset-file data/my_dataset.csv
```

### Output
```
--output data/activations     # Output directory
--num-samples 50000           # How many samples
--batch-size 512              # Buffer batch size
```

### Device
```
--device cuda         # GPU (default, must have CUDA)
--device cpu          # CPU (slower, works anywhere)
```

### Data Type
```
--dtype bfloat16      # Default (recommended)
--dtype float16       # Alternative
--dtype float32       # Not recommended (uses 2x memory)
```

---

## Common Commands

### Minimal Test (No GPU required, 1 minute)
```bash
python scripts/test_activation_capture.py
```

### Fast Validation (5 GPU minutes)
```bash
python scripts/full_capture.py \
    --num-samples 10 --device cuda
```

### Medium Run (30 GPU minutes)
```bash
python scripts/full_capture.py \
    --dataset pilot --num-samples 100 --device cuda
```

### Full Production Run (3 GPU hours)
```bash
python scripts/full_capture.py \
    --dataset libritts --num-samples 50000 --device cuda
```

---

## Check Results

```bash
# List activation files
ls -lh data/activations_*/layer_*.npy

# Check file size
du -sh data/activations_*/

# Inspect activation data
python << 'EOF'
import numpy as np
act = np.load("data/activations_qwen3tts/layer_01.npy")
print(f"Shape: {act.shape}")
print(f"Dtype: {act.dtype}")
print(f"Min: {act.min():.3f}, Max: {act.max():.3f}")
EOF
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `No module named 'torch'` | Run `pip install -r requirements.txt` |
| `CUDA out of memory` | Use `--device cpu` or reduce `--num-samples` |
| `No activations captured` | Run validation test first |
| `Model download timeout` | Set `export HF_HOME=/path/to/cache` |
| `No such file: libritts` | Falls back to pilot automatically |

**Full troubleshooting:** See `docs/ACTIVATION_CAPTURE_TROUBLESHOOTING.md`

---

## Hardware Requirements

| Scenario | GPU | Time |
|----------|-----|------|
| Test only | None (CPU OK) | 1 min |
| 100 samples | RTX 3080+ | 5 min |
| 50K samples | RTX 4090/A100 | 2-3 hours |

**Storage:** ~100 GB for 50K samples

---

## What Happens Next?

After capture completes:

1. **Phase 2 (SAE Training):** Train sparse autoencoder on activations
2. **Phase 3 (Feature Analysis):** Map SAE features to phonemes
3. **Phase 4 (Intervention):** Steer pronunciation via feature patching
4. **Phase 5 (Distillation):** Transfer to smaller models

---

## Documentation

- 📘 **[Activation Capture Workflow](docs/ACTIVATION_CAPTURE_WORKFLOW.md)** — Complete guide with examples
- 🔧 **[Troubleshooting](docs/ACTIVATION_CAPTURE_TROUBLESHOOTING.md)** — Common issues & fixes
- 📊 **[Phase 1 Status](docs/PHASE1_STATUS.md)** — Full implementation status
- 🏗️ **[Model Structure](docs/QWEN3_TTS_STRUCTURE.md)** — How Qwen3-TTS is organized
- 📝 **[Project Plan](docs/PROJECT_PLAN.md)** — 5-phase research roadmap

---

**✅ Ready to capture? Run: `python scripts/test_activation_capture.py`**
