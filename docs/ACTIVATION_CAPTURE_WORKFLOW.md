# Activation Capture Workflow for Qwen3-TTS

## Overview

This document explains the complete workflow for capturing activations from Qwen3-TTS during text-to-speech synthesis. The pipeline records internal hidden states from phonetic processing layers, which are later used to train Sparse Autoencoders (SAEs).

## Architecture

```
User Text Input
    ↓
Wrapper.generate(text="...")
    ↓
Model.generate(text, language)  [Qwen3TTSModel]
    ↓
model.model.talker  [Qwen3TTSTalkerForConditionalGeneration]
    ↓
model.model.talker.model.layers[1-7]  [Transformer blocks]
    ↓
├─ layer.self_attn
├─ layer.mlp  ← HOOK POINT (post-activation)
└─ layer.norm
    ↓
Activation Hook captures: (num_frames, d_model=1024)
    ↓
ActivationBuffer stores → .npy files
```

## Components

### 1. Qwen3TTSWrapper (`src/models/qwen3_tts_wrapper.py`)

**Responsible for:**
- Loading the Qwen3TTSModel from HuggingFace
- Handling model configuration (device, dtype=bfloat16)
- Providing a unified `generate()` interface for different generation modes
- Returning layer accessor function for activation hooks

**Key methods:**
- `generate(text, language="English")` - Base TTS mode (used for activation capture)
- `get_target_layers()` - Returns [1, 2, 3, 4, 5, 6, 7] (first 25% of 28-layer talker)
- `get_layer_accessor()` - Returns accessor function for `model.model.talker.model.layers[i]`

### 2. ActivationHook (`src/hooks.py`)

**Responsible for:**
- Attaching PyTorch forward hooks to model layers
- Recording hidden state activations during forward pass
- Collecting and organizing activations by layer index

**Key methods:**
- `attach(hook_type="mlp")` - Attach hooks to MLP post-activations
- `collect()` - Return `dict[layer_idx, Tensor]` of captured activations
- `detach()` - Remove hooks from model

**Hook point:** `model.model.talker.model.layers[i].mlp` (post-activation)

### 3. ActivationBuffer (`src/data/activation_buffer.py`)

**Responsible for:**
- Buffering activations in memory
- Converting to FP16 for storage efficiency
- Flushing buffers to disk as `.npy` files
- Tracking statistics (layer counts, save paths)

**Output structure:**
```
data/activations/
├── layer_01.npy  (shape: [num_frames, 1024])
├── layer_02.npy
├── ...
└── layer_07.npy
```

### 4. AlignedActivationBuffer (phoneme-aligned version)

**Additional features:**
- Organizes activations by phoneme
- Stores frame-to-phoneme alignment
- Generates phoneme inventory statistics

**Output structure:**
```
data/activations_aligned/en/
├── layer_01/
│   ├── phoneme_ah.npy
│   ├── phoneme_eh.npy
│   └── ...
├── layer_02/
├── ...
├── phoneme_inventory.json
└── frame_labels.jsonl
```

## Workflow: Step-by-Step

### Phase 1: Setup

```python
from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper
from src.hooks import ActivationHook

# Load model
wrapper = Qwen3TTSWrapper(device="cuda", dtype=torch.bfloat16)
model = wrapper.model
target_layers = wrapper.get_target_layers()
layer_accessor = wrapper.get_layer_accessor()
```

**Why bfloat16?**
- More stable numerically than float16
- Doesn't significantly impact activation quality
- Reduces memory usage on NVIDIA GPUs

### Phase 2: Attach Hooks

```python
hook = ActivationHook(
    model,
    layer_indices=target_layers,  # [1, 2, 3, 4, 5, 6, 7]
    layer_accessor=layer_accessor,
    device="cpu",  # Move hooks to CPU to save GPU memory
    dtype=torch.float16,
)

hook.attach("mlp")
```

**Why layer_accessor?**
- Qwen3TTSModel has a complex nested structure (PEFT-wrapped)
- Direct layer access doesn't work
- Custom accessor navigates: `model.model.talker.model.layers[i]`

### Phase 3: Run Inference

```python
with torch.no_grad():
    # Generate speech from text
    # This triggers forward pass through all talker layers
    output = wrapper.generate(text="Hello world")
    # output = {"waveform": array, "sample_rate": 24000}
```

**Critical:** Use `wrapper.generate()` not `model(input_ids)`
- Qwen3TTSModel doesn't support direct calling
- `generate()` internally handles tokenization and forward pass
- Hooks fire during the internal forward pass

### Phase 4: Collect Activations

```python
# Get activations from all hooked layers
activations = hook.collect()
# Returns: {1: Tensor[frames, 1024], 2: Tensor[frames, 1024], ...}

# Optional: reset for next sample
hook.reset()
```

### Phase 5: Buffer and Save

```python
buffer = ActivationBuffer(
    output_dir="data/activations",
    layer_indices=target_layers,
    dtype="float16"
)

buffer.add_batch(activations)

# After processing all samples
buffer.flush()
```

### Phase 6: Cleanup

```python
hook.detach()  # Remove hooks from model
```

## Complete Minimal Example

```python
#!/usr/bin/env python3
import torch
from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper
from src.hooks import ActivationHook
from src.data.activation_buffer import ActivationBuffer

# Setup
wrapper = Qwen3TTSWrapper(device="cuda", dtype=torch.bfloat16)
model = wrapper.model
target_layers = wrapper.get_target_layers()
layer_accessor = wrapper.get_layer_accessor()

# Create hook
hook = ActivationHook(
    model,
    layer_indices=target_layers,
    layer_accessor=layer_accessor,
    device="cpu",
    dtype=torch.float16,
)
hook.attach("mlp")

# Create buffer
buffer = ActivationBuffer(
    output_dir="data/activations_test",
    layer_indices=target_layers,
    dtype="float16"
)

# Process a few samples
texts = [
    "Hello, this is a test.",
    "How are you today?",
    "The weather is nice.",
]

for text in texts:
    with torch.no_grad():
        # Forward pass
        wrapper.generate(text=text)

        # Collect activations
        activations = hook.collect()
        buffer.add_batch(activations)
        hook.reset()

# Cleanup
hook.detach()
buffer.flush()

print("✅ Activation capture complete!")
```

## Key Design Decisions

### 1. Target Layers (1-7)

**Why early layers?**
- Layers 1-7: Grapheme-to-phoneme mapping and basic phonetic decoding
- Layers 8-28: Prosody refinement, speaker identity, high-level phrasing

**Rationale:** Phonetic features are most explicit in early layers; later layers mix in prosodic and speaker factors that are harder to interpret.

### 2. MLP Post-Activation Hook Point

**Why MLP, not attention?**
- MLP layers process and refine representations
- Post-activation captures after nonlinearity (ReLU/GELU)
- More informative than attention weights alone

**Structure:**
```
residual_stream
    ↓
[attention block]
    ↓
[mlp block] ← Hook here (post-activation)
    ↓
[layer_norm]
    ↓
output
```

### 3. CPU-side Hook Collection

**Why move to CPU?**
```python
device="cpu"  # Move hook tensors to CPU
```

- Saves GPU VRAM during capture (3-4 GB freed)
- GPU focuses entirely on inference
- CPU has much more RAM for buffering

### 4. FP16 Storage

**Memory efficiency:**
- FP32: ~200 GB for 50M vectors (1024-dim)
- FP16: ~100 GB for same data
- Minimal quality loss for SAE training

## Troubleshooting

### Issue: Hooks don't fire

**Symptoms:** `activations` dict is empty after `collect()`

**Solutions:**
1. Verify `hook.attach("mlp")` completed without errors
2. Check that `layer_accessor` returns correct layers
3. Ensure `generate()` is called in `torch.no_grad()` context
4. Verify model is in eval mode (usually automatic)

### Issue: "'Qwen3TTSModel' object is not callable"

**Symptoms:** Error when trying `model(input_ids)`

**Solution:** Use `wrapper.generate(text="...")` instead
- Never call the model directly with input_ids
- Use the high-level wrapper interface

### Issue: Out of memory

**Solutions:**
- Reduce batch size
- Move hooks to CPU: `device="cpu"`
- Use float16 instead of float32
- Process fewer samples per run

### Issue: Wrong layers accessed

**Symptoms:** Hooks attached but layers are empty

**Solution:** Always use the custom `layer_accessor` from wrapper
```python
layer_accessor = wrapper.get_layer_accessor()  # ✅ Correct
```

Don't try to access `model.talker.layers[i]` directly - this won't work due to PEFT wrapping.

## Performance Benchmarks

| Phase | Operation | VRAM | Time (100 samples) |
|-------|-----------|------|-------------------|
| Load Model | model.from_pretrained() | 2-3 GB | ~30 sec |
| Setup Hooks | attach() | <100 MB | <1 sec |
| Capture | forward pass × 100 | 4-5 GB | ~3-4 min |
| Buffer | add_batch() + flush() | 200-300 MB | <30 sec |

**Total time for 50K samples:** ~2-3 hours on RTX 4090

## References

- [Qwen3-TTS Model Structure](QWEN3_TTS_STRUCTURE.md)
- [Activation Hook Implementation](../src/hooks.py)
- [ActivationBuffer Implementation](../src/data/activation_buffer.py)
- [Qwen3TTSWrapper Implementation](../src/models/qwen3_tts_wrapper.py)
