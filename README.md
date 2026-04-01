# Phonetic SAE — Mechanistic Interpretability for LLM-based TTS

This repository implements a pipeline to discover, analyze, and intervene on phonetic features in LLM-based TTS models using Sparse Autoencoders (SAEs). The project combines activation capture from TTS models, Matryoshka SAE training (MSAE), and causal intervention tooling.

## Quick Links

**Getting Started (START HERE!):**
- 🚀 [Phase 1 Quick Start](docs/PHASE1_QUICKSTART.md) — Your first activation capture in 30 minutes ⭐ **START HERE**
- ✅ [Onboarding Checklist](docs/ONBOARDING_CHECKLIST.md) — Track your Phase 1 progress with checkboxes
- 📖 [Getting Started](docs/GETTING_STARTED.md) — Installation & environment setup

**Project Documentation:**
- 🎯 [Project Overview](docs/PROJECT_OVERVIEW.md) — Goals & research context
- 📋 [Project Plan](docs/PROJECT_PLAN.md) — 6-week roadmap + next steps
- 📊 [Executive Summary](docs/EXECUTIVE_SUMMARY.md) — High-level overview
- ✅ [Implementation Status](docs/IMPLEMENTATION_STATUS.md) — What's been built
- 📤 [Delivery Report](docs/DELIVERY_REPORT.md) — Final delivery summary
- 🔧 [MSAE Integration](docs/MSAE_for_TTS.md) — MSAE setup guide

**Architecture & Technical References:**
- [Qwen3-TTS Architecture](docs/QWEN3_TTS_0.6B_ARCHITECTURE.md)
- [CosyVoice2 Architecture](docs/COSYVOICE2_0.5B_ARCHITECTURE.md)
- [Model Comparison](docs/MODEL_COMPARISON.md)
- [Dataset Guide](docs/DATASET.md)
- [Phoneme Alignment Guide](docs/PHONEME_ALIGNMENT.md) ⭐ **NEW**
- [Qwen3-ForcedAligner Inference](docs/QWEN3_FORCEDALIGNER_INFERENCE.md) ⭐ **NEW**
- [Phoneme Set Sources](docs/PHONEME_SETS_SOURCES.md) ⭐ **NEW**
- [Phoneme Alignment Strategy](docs/PHONEME_ALIGNMENT_STRATEGY.md) — Audio-first approach (better than G2P alone)
- [Better Phoneme Alignment](docs/BETTER_PHONEME_ALIGNMENT.md) — Practical implementation guide
- [Phoneme Preparation Quick Reference](docs/PHONEME_PREPARATION_QUICK_REFERENCE.md) — TL;DR for phoneme converters
- [API Discovery Guide](docs/ACTUAL_API_DISCOVERY.md) — How to verify the actual model API
- [Tools & Utilities Reference](docs/TOOLS_AND_UTILITIES.md) — Complete guide to all scripts and tools

## Prerequisites

### System Requirements
- **Python:** 3.10+ (3.11 recommended)
- **GPU:** RTX 3090/4090 or better (24GB+ VRAM) — models require substantial memory
- **CUDA:** 12.1+ (for GPU support)
- **Disk Space:** ~500GB for full 50K-sample activation capture
- **Git:** Required for cloning and submodule management

### Clone Repository with Submodules

This repository uses Git submodules for third-party dependencies (Qwen3-TTS, CosyVoice2, MSAE):

```bash
# Clone with submodules (recommended)
git clone --recursive https://github.com/your-repo/phonetic-sae.git
cd phonetic-sae

# OR if you've already cloned without --recursive:
git submodule update --init --recursive
```

**Submodules included:**
- `third_party/Qwen3-TTS/` — Qwen3-TTS model code
- `third_party/CosyVoice2/` — CosyVoice2 model code
- `third_party/MSAE/` — Matryoshka SAE reference implementation

Verify submodules are initialized:
```bash
git submodule status
# Should show all with ✓ status (no '-' prefix)
```

## Getting started

### 1. Environment Setup

Create a Python virtual environment (recommended Python 3.10/3.11):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements-cpu.txt
```

For GPU hosts, use `requirements-cuda.txt` and a compatible CUDA toolchain (cu121 recommended):
```bash
pip install -r requirements-cuda.txt
```

### 2. Verify Dependencies

```bash
# Test imports
python -c "
from src.hooks import ActivationHook
from src.sae import TopKSAE
from src.alignment import QwenForcedAligner
print('✅ All core imports successful')
"

# Verify CUDA (if using GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify submodules
git submodule status
```

### 3. Download Models

Models are downloaded automatically on first use from HuggingFace:
- Qwen3-TTS-0.6B (2.5GB)
- CosyVoice2-0.5B (2GB)
- Qwen3-ForcedAligner-0.6B (2.5GB)

First run may take time for downloads. Set cache directory if needed:
```bash
export HF_HOME=/path/to/models/cache
```

## Data Preparation

### Supported Datasets

The project supports English, Mandarin, and Cantonese datasets in the following formats:

#### LibriTTS-R (English)
Pre-configured English dataset with speaker diversity and controlled text:
```bash
python scripts/full_capture.py \
  --model qwen3tts \
  --dataset libritts \
  --output data/activations/qwen3tts \
  --num-samples 50000
```

#### Custom Multilingual Dataset
Supports both **JSONL** and **CSV** datasets with text-audio pairs across multiple languages (English, Mandarin, Cantonese).

**JSONL Format (Recommended):**
```jsonl
{"text": "你好", "lang": "zh", "speech_token": [...], "codec": [...]}
{"text": "你好", "lang": "yue", "speech_token": [...], "codec": [...]}
{"text": "Hello", "lang": "en", "speech_token": [...], "codec": [...]}
```

**CSV Format:**
```csv
text,lang,audio_path,speaker_id
你好，今天天气很好,zh,data/audio/mandarin_001.wav,speaker_001
點樣啊,yue,data/audio/cantonese_001.wav,speaker_002
How are you today?,en,data/audio/english_001.wav,speaker_003
```

**Language Codes:** `"en"`, `"zh"`, `"yue"` (or full names: English, Mandarin, Cantonese)

**Usage:**
```bash
python scripts/full_capture.py \
  --model qwen3tts \
  --dataset custom \
  --dataset-csv data/out.jsonl \
  --output data/activations/qwen3tts \
  --num-samples 50000
```

### Activation Capture

**Step 1: Pilot Run** (validate pipeline on 100 sentences)
```bash
python scripts/pilot_capture.py \
  --model qwen3tts \
  --output data/pilot_activations \
  --num-samples 100 \
  --device cuda
```

Output: Activation statistics and `.npy` files per layer in `data/pilot_activations/layer_*.npy`

**Step 2: Full Capture** (50,000+ sentences)
```bash
python scripts/full_capture.py \
  --model qwen3tts \
  --dataset libritts \
  --output data/activations/qwen3tts \
  --num-samples 50000 \
  --device cuda
```

**Storage Requirements:**
- **Qwen3-TTS** (1024-dim): ~100 GB for 50M vectors (FP16)
- **CosyVoice2** (896-dim): ~90 GB for 50M vectors (FP16)

### Model Support

The pipeline supports two LLM-based TTS architectures:

| Model | Backbone Layers | Target Layers | d_model | TTS Task |
|-------|-----------------|---------------|---------|----------|
| **Qwen3-TTS-0.6B** | Talker (28L) | 1–7 | 1024 | FastSpeech2-style |
| **CosyVoice2-0.5B** | Qwen2.5 (24L) | 1–6 | 896 | Zero-shot voice cloning |

Select model with `--model qwen3tts` or `--model cosyvoice2`.

## Training

### SAE Training Pipeline

Once activations are captured, train a Sparse Autoencoder to discover phonetic features.

**Step 1: Verify Activations**

```bash
python scripts/train_sae.py \
  --config configs/sae_qwen3tts.yaml \
  --activation-dir data/activations/qwen3tts \
  --output checkpoints/sae_qwen3tts \
  --validate-only
```

**Step 2: Start Training**

```bash
python scripts/train_sae.py \
  --config configs/sae_qwen3tts.yaml \
  --activation-dir data/activations/qwen3tts \
  --output checkpoints/sae_qwen3tts \
  --device cuda \
  --wandb-project phonetic-sae
```

### Configuration

Hyperparameters for each architecture are pre-configured:

**`configs/sae_qwen3tts.yaml`:**
```yaml
d_in: 1024              # Qwen3-TTS hidden dim
d_sae: 16384            # 16× expansion factor
k: 32                   # Top-K sparsity
learning_rate: 1e-3
batch_size: 8192
max_steps: 500000
warmup_steps: 10000
decay_schedule: cosine
dtype: float16
```

**`configs/sae_cosyvoice2.yaml`:**
```yaml
d_in: 896               # CosyVoice2 hidden dim
d_sae: 14336            # 16× expansion factor
k: 32
# ... same hyperparameters as above
```

Override any parameter via CLI:
```bash
python scripts/train_sae.py \
  --config configs/sae_qwen3tts.yaml \
  --activation-dir data/activations/qwen3tts \
  --learning-rate 5e-4 \
  --batch-size 4096
```

### Monitoring Training

Training logs metrics to Weights & Biases (W&B):
- **Reconstruction loss** — How well SAE reconstructs original activations
- **Explained variance** — % of activation variance captured
- **Dead features** — Number of never-activated SAE features (resampled automatically)
- **Sparsity** — Average # of active features per sample

View live dashboard:
```bash
wandb sweep configs/sae_qwen3tts.yaml
```

### Hardware Requirements

Estimated VRAM and time for RTX 3090/4090 (24 GB):

| Phase | Operation | VRAM | Time (50K samples) |
|-------|-----------|------|-------------------|
| 1 | Activation mining | 3–4 GB | ~2–3 hours |
| 2 | SAE training | 300–500 MB | ~12 hours |
| 3 | Feature mapping | <100 MB | ~30 min |

### Output Artifacts

After training completes:
```
checkpoints/sae_qwen3tts/
├── sae_checkpoint_100000.pt     # Model at step 100k
├── sae_checkpoint_200000.pt     # Model at step 200k
└── sae_final.pt                  # Final trained model
```

Load trained SAE:
```python
import torch
from src.sae import TopKSAE

sae = TopKSAE.load("checkpoints/sae_qwen3tts/sae_final.pt")
z_sparse, z_full = sae.encode(activations)
x_recon = sae.decode(z_sparse)
```

## Troubleshooting

### Git Submodule Issues

**Problem: `fatal: No submodule mapping found in .gitmodules for path 'third_party/...'`**

Solution: Initialize submodules properly:
```bash
git submodule update --init --recursive
```

**Problem: Submodule directory is empty**

Solution: Fetch submodule contents:
```bash
git submodule update --init --recursive --depth 1
```

**Problem: Want to update submodules to latest**

```bash
git submodule update --remote --recursive
git add .gitmodules third_party/
git commit -m "Update submodules"
```

**Problem: Clone failed due to network issues**

Try shallow clone first:
```bash
git clone --recursive --depth 1 https://github.com/your-repo/phonetic-sae.git
```

Then fetch full history:
```bash
git fetch --unshallow
```

### Model Download Issues

**Problem: HuggingFace model download times out**

Set custom cache and retry:
```bash
export HF_HOME=/path/to/fast/disk
rm -rf $HF_HOME/*  # Clear cache
python scripts/inspect_aligner.py  # Retry download
```

**Problem: CUDA/GPU issues**

Fall back to CPU temporarily:
```bash
python scripts/pilot_capture.py --device cpu --num-samples 10
```

Check CUDA status:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
nvidia-smi  # Check GPU memory
```

### Import Errors

**Problem: `ModuleNotFoundError: No module named 'src.alignment'`**

Make sure submodules are initialized and you're in repo root:
```bash
pwd  # Should be: .../phonetic-sae
ls src/alignment/  # Should show files
python -c "from src.alignment import QwenForcedAligner"
```

**Problem: Submodule imports fail**

Ensure submodules are initialized:
```bash
git submodule status
# Should show something like:
#  a1b2c3d4... third_party/Qwen3-TTS (detached HEAD)
#  e5f6g7h8... third_party/CosyVoice2 (detached HEAD)
#  i9j0k1l2... third_party/MSAE (detached HEAD)
```

## Docs
See `docs/` for integration and experiment notes. Current docs:
- `docs/MSAE_for_TTS.md` — MSAE implementation summary and step-by-step integration for LLM-TTS activations.

## Contributing
- Follow the instructions in `CLAUDE.md` for project phases and research workflow.
