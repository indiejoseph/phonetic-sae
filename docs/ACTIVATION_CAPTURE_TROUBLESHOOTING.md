# Activation Capture Troubleshooting Guide

## Quick Diagnostics

First, run the validation test:
```bash
python scripts/test_activation_capture.py
```

This tests the complete pipeline on a single sample. If it works, skip to "Full Capture Issues".

---

## Installation & Setup Issues

### Error: `ModuleNotFoundError: No module named 'torch'`

**Cause:** Virtual environment not activated or dependencies not installed

**Fix:**
```bash
# Activate venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

**Verify:**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

---

### Error: `ModuleNotFoundError: No module named 'src'`

**Cause:** Running script from wrong directory

**Fix:**
```bash
# Make sure you're in the repo root
cd phonetic-sae
python scripts/test_activation_capture.py
```

**Verify:**
```bash
ls src/  # Should show: hooks.py, models/, data/, etc.
```

---

### Error: `ImportError: cannot import name 'Qwen3TTSWrapper'`

**Cause:** Submodules not initialized

**Fix:**
```bash
git submodule update --init --recursive
```

**Verify:**
```bash
ls src/models/  # Should show qwen3_tts_wrapper.py, cosyvoice2_wrapper.py, etc.
```

---

## Model Loading Issues

### Error: `RuntimeError: CUDA out of memory`

**Cause:** GPU doesn't have enough memory to load model

**Solutions (in order):**
1. Use CPU (slower, but works):
   ```bash
   python scripts/test_activation_capture.py  # Uses CPU by default
   ```

2. Use smaller GPU (if available):
   ```bash
   # Run on CPU instead
   CUDA_VISIBLE_DEVICES="" python scripts/full_capture.py \
       --model qwen3tts \
       --dataset pilot \
       --num-samples 10 \
       --device cpu
   ```

3. Use bfloat16 (already default):
   ```bash
   python scripts/full_capture.py --dtype bfloat16
   ```

4. Free GPU memory:
   ```bash
   nvidia-smi  # Check what else is using GPU
   pkill python  # Kill other Python processes
   ```

**Recommended:** Use GPU with 24GB+ VRAM (RTX 4090, A100, etc.)

---

### Error: `Connection timeout downloading model`

**Cause:** HuggingFace model download network issue

**Fix:**
1. Set custom cache location:
   ```bash
   export HF_HOME=/path/to/fast/disk
   python scripts/test_activation_capture.py
   ```

2. Clear cache and retry:
   ```bash
   rm -rf ~/.cache/huggingface
   python scripts/test_activation_capture.py
   ```

3. Pre-download model:
   ```python
   from transformers import AutoModel
   model = AutoModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
   ```

---

### Error: `Failed to load model: 'NoneType' object has no attribute 'model'`

**Cause:** Model loading failed silently

**Debug:**
```bash
python -c "
from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper
import torch
wrapper = Qwen3TTSWrapper(device='cpu', dtype=torch.bfloat16)
print(f'Model type: {type(wrapper.model)}')
print(f'Tokenizer type: {type(wrapper.tokenizer)}')
"
```

---

## Activation Hook Issues

### Error: Activation dict is empty after `hook.collect()`

**Symptoms:**
- Script runs without error
- But `activations = {}` (empty dict)

**Cause:** Hooks didn't fire, usually because:
1. Layers weren't found
2. Forward pass didn't run
3. Hooks attached to wrong layer

**Debug checklist:**
```python
from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper
import torch

wrapper = Qwen3TTSWrapper(device='cpu', dtype=torch.bfloat16)

# 1. Check layers exist
try:
    layer = wrapper.get_layer_accessor()(wrapper.model, 1)
    print(f"✅ Layer 1 found: {type(layer)}")
except Exception as e:
    print(f"❌ Layer access failed: {e}")

# 2. Check target layers
target_layers = wrapper.get_target_layers()
print(f"Target layers: {target_layers}")

# 3. Check tokenizer
if wrapper.tokenizer:
    tokens = wrapper.tokenizer.encode("test", return_tensors="pt")
    print(f"✅ Tokenizer works: {tokens.shape}")
else:
    print("❌ No tokenizer")
```

---

### Error: "Cannot access layer X"

**Cause:** Layer index out of range or wrong accessor

**Fix:**
```python
# Check how many layers model has
from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper
wrapper = Qwen3TTSWrapper(device='cpu')

# Should return [1, 2, 3, 4, 5, 6, 7] - only 7 target layers
target_layers = wrapper.get_target_layers()
print(f"Target layers: {target_layers}")

# Each layer should be accessible
for i in target_layers:
    try:
        layer = wrapper.get_layer_accessor()(wrapper.model, i)
        print(f"✅ Layer {i}: {type(layer).__name__}")
    except Exception as e:
        print(f"❌ Layer {i}: {e}")
```

---

## Forward Pass Issues

### Error: "'Qwen3TTSModel' object is not callable"

**This is FIXED.** If you get this error, you're using old code.

**Correct approach:**
```python
# ❌ OLD (broken)
input_ids = tokenizer.encode(text, return_tensors="pt")
output = model(input_ids)  # FAILS

# ✅ NEW (works)
output = wrapper.generate(text=text)
```

---

### Error: "Model does not support base generation mode"

**Cause:** Model doesn't have `generate()` method

**Debug:**
```python
from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper
wrapper = Qwen3TTSWrapper(device='cpu')

# Check if generate exists
if hasattr(wrapper.model, 'generate'):
    print("✅ Model has generate()")
else:
    print("❌ Model missing generate()")

# List all methods
methods = [m for m in dir(wrapper.model) if not m.startswith('_')]
generate_methods = [m for m in methods if 'generate' in m]
print(f"Generate methods: {generate_methods}")
```

---

### Error: `RuntimeError: Expected all tensors to be on the same device`

**Cause:** Input tensor on different device than model

**Fix:**
```python
# Make sure input is on correct device
input_ids = input_ids.to(device)  # Match model device

# OR use wrapper which handles this
output = wrapper.generate(text=text)  # Handles device internally
```

---

## Activation Capture Issues

### No activations captured but no error

**Symptoms:**
- Script completes
- But activation files are empty or don't exist

**Cause:** Hooks attached but not collecting data

**Debug:**
```bash
# Check output directory
ls -lh data/activations/
# Should show: layer_01.npy, layer_02.npy, etc.

# Check file sizes
du -h data/activations/layer_*.npy
# Should be >0 bytes
```

**Common causes:**
1. Hook attached to wrong place
2. Forward pass doesn't go through hooked layers
3. Buffer not flushed

**Check buffer flush:**
```python
# Make sure to flush!
hook.detach()
buffer.flush()  # REQUIRED - actually writes files
```

---

### Activation shapes are wrong

**Expected:**
- Shape: `[num_frames, 1024]` for Qwen3-TTS
- Dtype: `float16` (FP16)
- Value range: small (±1 to ±10 typically)

**Check:**
```python
import numpy as np

act = np.load("data/activations/layer_01.npy")
print(f"Shape: {act.shape}")
print(f"Dtype: {act.dtype}")
print(f"Min: {act.min()}, Max: {act.max()}")
print(f"Mean: {act.mean()}, Std: {act.std()}")

# Should output something like:
# Shape: (86, 1024)
# Dtype: float16
# Min: -2.5, Max: 3.2
# Mean: 0.1, Std: 0.8
```

**If shapes wrong:**
- Wrong hook attachment point
- Wrong layer accessor
- Model architecture mismatch

---

### Out of memory during capture

**Symptoms:**
- Runs for a while, then crashes with CUDA error

**Cause:** Memory builds up during iteration

**Fix (in order of effect):**

1. **Reduce batch size** (fastest):
   ```bash
   python scripts/full_capture.py --batch-size 256
   ```

2. **Move hooks to CPU**:
   ```python
   hook = ActivationHook(
       model,
       layer_indices=target_layers,
       layer_accessor=layer_accessor,
       device="cpu",  # Save GPU memory
       dtype=torch.float16,
   )
   ```

3. **Reduce num-samples**:
   ```bash
   python scripts/full_capture.py --num-samples 1000
   ```

4. **Use quantization**:
   ```bash
   python scripts/full_capture.py --dtype float16
   ```

5. **Use CPU only** (slowest):
   ```bash
   python scripts/full_capture.py --device cpu
   ```

---

## Phoneme Alignment Issues

### Error: "Failed to load forced aligner"

**Cause:** Qwen3-ForcedAligner model not available

**Fix:**
```bash
# Script falls back to unaligned capture automatically
# To verify aligner works:
python -c "
from src.alignment import QwenForcedAligner
aligner = QwenForcedAligner(device='cpu', language='en')
print('✅ Aligner loaded')
"
```

---

### Alignment errors don't stop capture

**This is by design.** The script logs alignment failures but continues:
```
⚠️ Sample X: no aligner or no audio path, skipping alignment
```

This is expected if you're capturing from text-only dataset.

---

### Audio files not found

**Cause:** Dataset file references audio files that don't exist

**Check:**
```bash
# Look at your dataset file
head -5 data/out.jsonl
# Check if audio_path or ref_audio_path exists

# Find where audio is
find data/ -name "*.wav" | head -10
```

**Fix:** Update dataset file to have correct audio paths

---

## Disk Space Issues

### Error: No space left on device

**Cause:** 50K activation capture takes ~100 GB

**Fix:**
1. Check available space:
   ```bash
   df -h /
   du -sh data/
   ```

2. Use different disk:
   ```bash
   python scripts/full_capture.py \
       --output /mnt/large_disk/activations
   ```

3. Capture fewer samples:
   ```bash
   python scripts/full_capture.py --num-samples 10000
   ```

---

## Performance Issues

### Capture is very slow

**Expected performance:**
- ~30-50 samples/minute on RTX 4090
- ~2-3 hours for 50K samples

**If slower:**
1. Check GPU usage:
   ```bash
   nvidia-smi -l 1  # Watch GPU utilization
   ```

2. Check if CPU-bound:
   ```bash
   # Should see high GPU usage (>90%)
   # If low, probably bottlenecked elsewhere
   ```

3. Profile:
   ```bash
   python -m cProfile scripts/full_capture.py --num-samples 10
   ```

---

## Verification Checklist

Before assuming everything works:

```bash
# 1. Test basic functionality
python scripts/test_activation_capture.py

# 2. Check output format
ls -lh data/activations/
file data/activations/layer_01.npy

# 3. Verify activation data
python << 'EOF'
import numpy as np
act = np.load("data/activations/layer_01.npy")
assert act.dtype == np.float16, f"Wrong dtype: {act.dtype}"
assert len(act.shape) == 2, f"Wrong shape: {act.shape}"
assert act.shape[1] == 1024, f"Wrong d_model: {act.shape[1]}"
print(f"✅ Activation format correct: {act.shape}")
EOF

# 4. Check for NaN/Inf
python << 'EOF'
import numpy as np
act = np.load("data/activations/layer_01.npy")
assert not np.isnan(act).any(), "Contains NaN"
assert not np.isinf(act).any(), "Contains Inf"
print(f"✅ No NaN/Inf values")
EOF
```

---

## Getting Help

If you're stuck:

1. **Run diagnostics:**
   ```bash
   python scripts/test_activation_capture.py  # Check full pipeline
   ```

2. **Check logs:**
   ```bash
   grep -i error *.log
   ```

3. **Enable debug logging:**
   ```bash
   LOGLEVEL=DEBUG python scripts/full_capture.py --num-samples 10
   ```

4. **Check documentation:**
   - [Activation Capture Workflow](ACTIVATION_CAPTURE_WORKFLOW.md)
   - [Qwen3-TTS Structure](QWEN3_TTS_STRUCTURE.md)
   - [Phase 1 Status](PHASE1_STATUS.md)

5. **Report issues** with:
   - Error message (full traceback)
   - Your hardware specs (GPU model, VRAM, etc.)
   - Steps to reproduce
   - Output from `test_activation_capture.py`

---

## Common Success Patterns

### Minimal Test (5 minutes)
```bash
python scripts/test_activation_capture.py
```

### Quick Validation (30 minutes)
```bash
python scripts/full_capture.py \
    --model qwen3tts \
    --dataset pilot \
    --num-samples 100 \
    --output data/test_activations
```

### Production Run (3 hours)
```bash
python scripts/full_capture.py \
    --model qwen3tts \
    --dataset libritts \
    --num-samples 50000 \
    --output data/activations_qwen3tts
```

### With Alignment (optional, slower)
```bash
python scripts/capture_with_alignment.py \
    --model qwen3tts \
    --dataset-file data/my_data.jsonl \
    --lang en \
    --num-samples 50000 \
    --output data/activations_aligned
```

---

**Last updated:** 2026-04-01
**Status:** All tests passing ✅
