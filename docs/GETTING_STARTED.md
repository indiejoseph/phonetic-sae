# Getting Started with PhoneticSAE

**Latest Update:** April 1, 2026
**Status:** Phase 0-2 Implementation Complete ✅

This guide covers installation, quick start examples, and detailed module usage.

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Install

```bash
cd /path/to/phonetic-sae
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Step 2: Run Pilot (100 sentences)

```bash
python scripts/pilot_capture.py \
  --model qwen3tts \
  --output data/pilot_activations \
  --num-samples 100 \
  --device cuda
```

**Output:** Activation `.npy` files + statistics in `data/pilot_activations/`

### Step 3: Train SAE on Pilot Data

```bash
python scripts/train_sae.py \
  --config configs/sae_qwen3tts.yaml \
  --activation-dir data/pilot_activations \
  --output checkpoints/sae_pilot \
  --max-steps 10000 \
  --device cuda
```

**Result:** Trained SAE in `checkpoints/sae_pilot/sae_final.pt`

---

## 📊 Full Pipeline (Production Workflow)

```bash
# 1. Capture 50K sentences (1-2 hours)
python scripts/full_capture.py \
  --model qwen3tts \
  --dataset libritts \
  --output data/activations/qwen3tts \
  --num-samples 50000

# 2. Train SAE (12 hours on RTX 4090)
python scripts/train_sae.py \
  --config configs/sae_qwen3tts.yaml \
  --activation-dir data/activations/qwen3tts \
  --output checkpoints/sae_qwen3tts \
  --wandb-project phonetic-sae

# 3. (Later) Feature discovery, interventions, distillation...
```

---

## 🔧 Installation Details

### System Requirements

- **Python:** 3.10+ (3.11 recommended)
- **GPU:** RTX 3090/4090 or better (24GB+ VRAM)
- **CUDA:** 12.1 or later
- **PyTorch:** 2.0+

### Virtual Environment Setup

```bash
# CPU-only (for development)
pip install -r requirements-cpu.txt

# With GPU (CUDA 12.1)
pip install -r requirements-cuda.txt

# Or install from source
pip install -e ".[dev,viz,eval]"
```

### Verify Installation

```bash
python -c "
from src.hooks import ActivationHook
from src.sae import TopKSAE
from src.data.activation_buffer import ActivationBuffer
print('✅ All imports successful')
"
```

---

## 📚 Key Modules

### ActivationHook: Capture Internal Activations

Captures model activations via PyTorch forward hooks.

```python
from src.hooks import ActivationHook
from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper

wrapper = Qwen3TTSWrapper(device="cuda")
model = wrapper.model
target_layers = wrapper.get_target_layers()  # [1, 2, 3, 4, 5, 6, 7]

# Attach hook to MLP post-activations
hook = ActivationHook(
    model,
    layer_indices=target_layers,
    device="cuda",
    dtype=torch.float16,
)
hook.attach("mlp")  # or "residual"

# Forward pass
input_ids = wrapper.tokenizer.encode("Hello", return_tensors="pt").to("cuda")
_ = model(input_ids)

# Collect activations
activations = hook.collect()  # dict[int, Tensor]
hook.detach()
```

**Parameters:**
- `layer_indices`: Which layers to hook (e.g., [1,2,3,4,5,6,7])
- `hook_point`: "mlp" or "residual" stream
- `dtype`: torch.float32, torch.float16 (default), etc.

---

### TopKSAE: Sparse Autoencoder

Decomposes activations into sparse interpretable features.

```python
from src.sae import TopKSAE
from src.sae.topk_sae import SAEConfig

# Configuration
config = SAEConfig(
    d_in=1024,        # Input dimension (Qwen3-TTS)
    d_sae=16384,      # Hidden dimension (16× expansion)
    k=32,             # Top-K sparsity (only 32 active features)
)

# Initialize SAE
sae = TopKSAE(config)

# Forward pass
x = torch.randn(32, 1024)  # Batch of activations
loss, metrics = sae(x)     # Returns (mse_loss, dict of metrics)

# Encode/decode for interventions
z_sparse, z_full = sae.encode(x)  # Sparse codes
x_hat = sae.decode(z_sparse)      # Reconstructed activations

# Monitor sparsity
print(f"Explained variance: {metrics['explained_variance']:.2%}")
print(f"Dead features: {metrics['dead_features']}")
```

**Methods:**
- `encode(x)` → (z_sparse, z_full) — Compress to sparse codes
- `decode(z)` → x_hat — Reconstruct from codes
- `forward(x)` → (loss, metrics) — Training forward pass
- `resample_dead_features()` — Replace inactive features

---

### ActivationBuffer: Streaming Capture

Memory-efficient buffer for saving captured activations to disk.

```python
from src.data.activation_buffer import ActivationBuffer

buffer = ActivationBuffer(
    output_dir="data/activations/qwen3tts",
    layer_indices=[1, 2, 3, 4, 5, 6, 7],
    batch_size=512,
    dtype="float16",
)

# In forward loop
for text in dataset:
    activations = hook.collect()
    buffer.add_batch(activations)  # Auto-flushes when batch_size reached

# Manual flush
buffer.flush()

# Saved structure:
# data/activations/qwen3tts/
# ├── layer_01_batch_000000.npy
# ├── layer_02_batch_000000.npy
# └── ...
```

**Features:**
- Automatic disk flush when batch size reached
- Per-layer file organization
- Quantization support (FP32/FP16/INT8)
- Efficient for 50M+ vectors

---

### ShuffledActivationBuffer: Training Data Loading

Loads and shuffles saved activations for SAE training.

```python
from src.data.shuffled_activation_buffer import ShuffledActivationBuffer

train_loader = ShuffledActivationBuffer(
    activation_dir="data/activations/qwen3tts",
    layer_index=1,           # Train one layer at a time
    batch_size=8192,
    shuffle=True,
    device="cuda",
)

# Iteration
for batch_activations in train_loader:
    loss, metrics = sae(batch_activations)
    loss.backward()
    optimizer.step()
```

---

### SAETrainer: Full Training Loop

Complete training with checkpointing, mixed precision, and W&B logging.

```python
from src.sae.trainer import SAETrainer

trainer = SAETrainer(
    sae=sae,
    train_loader=train_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device="cuda",
    dtype=torch.float16,
    wandb_enabled=True,
)

trainer.train(max_steps=500000, checkpoint_every=5000)
```

**Features:**
- Automatic checkpointing every N steps
- Weights & Biases integration
- Mixed precision (AMP) training
- Dead feature resampling

---

### Model Wrappers

#### Qwen3-TTS

```python
from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper

wrapper = Qwen3TTSWrapper(device="cuda")
target_layers = wrapper.get_target_layers()  # [1, 2, 3, 4, 5, 6, 7]
model = wrapper.model

# Generate speech
tokens = wrapper.tokenizer.encode("Hello", return_tensors="pt")
output = model(tokens)
```

**Model Info:**
- **Architecture:** Talker (28 layers) + Code Predictor (5 layers)
- **Hidden dim:** 1024
- **Target layers:** 1-7 (first 25%)
- **Hook point:** `model.talker.layers[i].mlp`

#### CosyVoice2

```python
from src.models.cosyvoice2_wrapper import CosyVoice2Wrapper

wrapper = CosyVoice2Wrapper(device="cuda")
target_layers = wrapper.get_target_layers()  # [1, 2, 3, 4, 5, 6]
model = wrapper.model
```

**Model Info:**
- **Architecture:** Qwen2.5 LLM (24 layers)
- **Hidden dim:** 896
- **Target layers:** 1-6 (first 25%)
- **Hook point:** `model.llm.layers[i].mlp`

---

## 📥 Dataset Preparation

### LibriTTS-R (English)

Pre-configured English dataset:

```bash
python scripts/full_capture.py \
  --model qwen3tts \
  --dataset libritts \
  --output data/activations/qwen3tts \
  --num-samples 50000
```

### Custom Multilingual Dataset

Supports both **CSV** and **JSONL** formats with language codes:

**JSONL Format (Recommended):**
```jsonl
{"text": "你好，今天天气很好", "lang": "zh", "speech_token": [...], "codec": [...], "speaker_id": "speaker_001"}
{"text": "點樣啊", "lang": "yue", "speech_token": [...], "codec": [...], "speaker_id": "speaker_002"}
{"text": "How are you?", "lang": "en", "speech_token": [...], "codec": [...], "speaker_id": "speaker_003"}
```

**CSV Format (Alternative):**
```csv
text,lang,audio_path,speaker_id
你好，今天天气很好,zh,data/audio/mandarin_001.wav,speaker_001
點樣啊,yue,data/audio/cantonese_001.wav,speaker_002
How are you?,en,data/audio/english_001.wav,speaker_003
```

**Supported Language Codes:**
- `"en"` or `"English"` → English
- `"zh"` or `"Mandarin"` → Mandarin Chinese
- `"yue"` or `"Cantonese"` → Cantonese

**Usage with JSONL:**

```bash
python scripts/full_capture.py \
  --model qwen3tts \
  --dataset custom \
  --dataset-csv data/out.jsonl \
  --output data/activations/qwen3tts \
  --num-samples 50000
```

**Usage with CSV:**

```bash
python scripts/full_capture.py \
  --model qwen3tts \
  --dataset custom \
  --dataset-csv data/custom_dataset.csv \
  --output data/activations/qwen3tts \
  --num-samples 50000
```

---

## 💾 Configuration Files

All hyperparameters are pre-configured in `configs/`:

**`configs/sae_qwen3tts.yaml`:**
```yaml
d_in: 1024              # Input dimension
d_sae: 16384            # 16× expansion
k: 32                   # Top-K sparsity
learning_rate: 1e-3
batch_size: 8192
max_steps: 500000
warmup_steps: 10000
decay_schedule: cosine
dtype: float16
```

Override via CLI:

```bash
python scripts/train_sae.py \
  --config configs/sae_qwen3tts.yaml \
  --activation-dir data/activations/qwen3tts \
  --learning-rate 5e-4 \
  --batch-size 4096
```

---

## 🖥️ Hardware Requirements

| Phase | Operation | VRAM | Time |
|-------|-----------|------|------|
| 1 | Activation mining (50K) | 3-4 GB | 1-2 hours |
| 2 | SAE training (500K steps) | 300-500 MB | 12 hours |
| 3 | Feature analysis | <100 MB | 30 min |

**Tested on:** RTX 3090, RTX 4090, A100

---

## ✅ Verify Installation

Run the minimal test suite (no GPU required):

```bash
pip install -e .
python -m pytest tests/test_activation_hook.py -v
```

For a quick functional test:

```bash
python -c "
import torch
from src.sae import TopKSAE
from src.sae.topk_sae import SAEConfig

config = SAEConfig(d_in=64, d_sae=256, k=16)
sae = TopKSAE(config)
x = torch.randn(32, 64)
loss, metrics = sae(x)
print(f'✅ SAE works: MSE = {loss.item():.6f}')
"
```

---

## 🔍 Common Issues

| Issue | Solution |
|-------|----------|
| Model files not found | Check `pretrained_models/` directory, download from HuggingFace if needed |
| CUDA out of memory | Reduce `batch_size`, use FP16 (already default) |
| Import errors | Run `pip install -e .` and verify `__init__.py` files exist |
| Hook not capturing | Verify layer indices match model structure, test with small batch first |
| Slow activation capture | Reduce `batch_size` or use FP16 quantization |

---

## 📖 Next Steps

1. **Complete Quick Start (5 min)** → Verify installation works
2. **Run Pilot Capture (30 min)** → Test on 100 sentences
3. **Full Capture (2-3 hours)** → Capture 50K sentences
4. **Train SAE (12+ hours)** → Begin feature discovery
5. **See PROJECT_PLAN.md** → Road map for Phases 3-5

---

## 📚 Additional Resources

- **README.md** — Main project overview
- **PROJECT_PLAN.md** — 6-week implementation roadmap
- **PROJECT_OVERVIEW.md** — Research goals and motivation
- **docs/** — Architecture docs, model comparisons, technical details

