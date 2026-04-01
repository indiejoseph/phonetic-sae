# Phonetic SAE — Mechanistic Interpretability for LLM-based TTS

This repository implements a pipeline to discover, analyze, and intervene on phonetic features in LLM-based TTS models using Sparse Autoencoders (SAEs). The project combines activation capture from TTS models, Matryoshka SAE training (MSAE), and causal intervention tooling.

## Quick Links

**Documentation Structure:**
- 📖 [Getting Started](docs/GETTING_STARTED.md) — Installation & quick start
- 🎯 [Project Overview](docs/PROJECT_OVERVIEW.md) — Goals & research context
- 📋 [Project Plan](docs/PROJECT_PLAN.md) — 6-week roadmap + next steps
- 📊 [Executive Summary](docs/EXECUTIVE_SUMMARY.md) — High-level overview
- ✅ [Implementation Status](docs/IMPLEMENTATION_STATUS.md) — What's been built
- 📤 [Delivery Report](docs/DELIVERY_REPORT.md) — Final delivery summary
- 🔧 [MSAE Integration](docs/MSAE_for_TTS.md) — MSAE setup guide

**Architecture References:**
- [Qwen3-TTS Architecture](docs/QWEN3_TTS_0.6B_ARCHITECTURE.md)
- [CosyVoice2 Architecture](docs/COSYVOICE2_0.5B_ARCHITECTURE.md)
- [Model Comparison](docs/MODEL_COMPARISON.md)
- [Dataset Guide](docs/DATASET.md)

## Getting started
1. Create a Python virtual environment (recommended Python 3.10/3.11).
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
```

For GPU hosts, use `requirements-cuda.txt` and a compatible CUDA toolchain (cu121 recommended for this repo).

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

## Docs
See `docs/` for integration and experiment notes. Current docs:
- `docs/MSAE_for_TTS.md` — MSAE implementation summary and step-by-step integration for LLM-TTS activations.

## Contributing
- Follow the instructions in `CLAUDE.md` for project phases and research workflow.
