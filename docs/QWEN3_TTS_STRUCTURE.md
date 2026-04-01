# Qwen3-TTS Model Structure & Layer Access

## Overview

The `qwen_tts.Qwen3TTSModel` has a complex nested wrapper structure. This document explains how the model is organized and how to correctly access its internal layers for activation capture and analysis.

## Architecture Diagram

```
Qwen3TTSModel (from qwen_tts package)
│
├── .model ────────────────────────────────────────────────────┐
│   Qwen3TTSForConditionalGeneration (PEFT-wrapped)            │
│                                                               │
│   ├── .base_model (unwrapped Qwen3TTSForConditionalGeneration)
│   │   └── .talker ──────────────────────────────────────┐   │
│   │       Qwen3TTSTalkerForConditionalGeneration         │   │
│   │       (PEFT-wrapped)                                 │   │
│   │                                                       │   │
│   │       ├── .base_model (unwrapped)                   │   │
│   │       │   └── .model ──────────────┐                │   │
│   │       │       Qwen3TTSTalkerModel   │ ← TARGET       │   │
│   │       │       ├── .layers[0]       │   (Contains    │   │
│   │       │       ├── .layers[1]       │    28 layers)  │   │
│   │       │       ├── ...              │                │   │
│   │       │       └── .layers[27]      │                │   │
│   │       │                            │                │   │
│   │       └── .model ─────────────────┘                 │   │
│   │                                                       │   │
│   └── .talker ─────────────────────────────────────────┘   │
│                                                               │
├── .speaker_encoder                                          │
│   Qwen3TTSSpeakerEncoder                                    │
│                                                               │
└── .processor                                                 │
    Processor for audio/text input preparation               │
```

## Layer Access Paths

### Correct Path (Used in Activation Hooks)

```python
model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")

# Navigate to talker layers:
layer_i = model.model.talker.model.layers[i]
```

**Full decomposition:**
- `model` - Qwen3TTSModel wrapper from qwen_tts
- `model.model` - Qwen3TTSForConditionalGeneration (PEFT-wrapped)
- `model.model.talker` - Qwen3TTSTalkerForConditionalGeneration (PEFT-wrapped)
- `model.model.talker.model` - Qwen3TTSTalkerModel (the actual transformer)
- `model.model.talker.model.layers[i]` - Individual transformer layer

### PEFT Wrapper Handling

The model uses PEFT (Parameter-Efficient Fine-Tuning) wrapping, which means accessing attributes sometimes requires going through `.base_model`:

```python
# If you need the unwrapped model:
talker_wrapper = model.model.talker
talker_unwrapped = talker_wrapper.base_model  # Unwrap PEFT

# The layer accessor in our code handles this automatically
```

## Component Breakdown

### 1. Qwen3TTSModel (from qwen_tts)
**Type:** Wrapper from the qwen-tts package
**Attributes:**
- `model` - The actual TTS model
- `processor` - Input processor
- `generate_*` methods for different generation modes

### 2. Qwen3TTSForConditionalGeneration
**Type:** PEFT-wrapped conditional generation model
**Attributes:**
- `talker` - The phonetic/linguistic processing component ⭐
- `speaker_encoder` - Extracts speaker characteristics from reference audio
- Various config and metadata attributes

### 3. Qwen3TTSTalkerForConditionalGeneration (PEFT-wrapped)
**Type:** The main phonetic processing component
**Attributes:**
- `model` - Actual talker transformer ⭐
- `code_predictor` - Acoustic detail refinement (5 layers)
- `codec_head` - Linear projection for codec space
- `base_model` - PEFT unwrapped version

### 4. Qwen3TTSTalkerModel ⭐ (TARGET)
**Type:** Transformer with 28 layers
**Attributes:**
- `layers[0]` through `layers[27]` - The 28 transformer blocks
- Each layer contains:
  - Self-attention module
  - MLP (feed-forward) module
  - Layer normalization

## Target Layers for SAE

For mechanistic interpretability and sparse autoencoder training, we focus on **early layers** of the talker:

```python
# Phonetic SAE target layers (first 25% of 28-layer talker)
target_layers = list(range(1, 8))  # Layers 1-7
```

**Rationale:**
- **Layers 1-7**: Grapheme-to-phoneme mapping and early phonetic processing
- **Layers 8-28**: Prosody refinement, speaker characteristics, and high-level phrasing

Hook point: `model.talker.model.layers[i].mlp` (MLP post-activation)

## Activation Capture

### Using ActivationHook with Custom Layer Accessor

```python
from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper
from src.hooks import ActivationHook

# Load model
wrapper = Qwen3TTSWrapper(device="cuda", dtype=torch.float16)
model = wrapper.model
target_layers = wrapper.get_target_layers()
layer_accessor = wrapper.get_layer_accessor()

# Create hook with custom accessor
hook = ActivationHook(
    model,
    layer_indices=target_layers,
    layer_accessor=layer_accessor,  # CRITICAL: pass the custom accessor
    device="cpu",
    dtype=torch.float16,
)

# Attach to MLP post-activations
hook.attach("mlp")

# Run inference
with torch.no_grad():
    output = model(input_ids)

# Collect activations
activations = hook.collect()  # dict[int, Tensor]
```

## Why Complex Wrapping?

The nested structure exists because:

1. **PEFT Wrapping** - Parameter-efficient fine-tuning adds a wrapper layer around each module
2. **Multi-component Architecture** - Qwen3-TTS has separate components:
   - Talker (phonetic/linguistic)
   - Speaker Encoder (speaker characteristics)
   - Code Predictor (acoustic refinement)
   - Speech Decoder (waveform synthesis)
3. **API Abstraction** - The qwen_tts package wraps the underlying model to provide high-level methods like `generate_voice_clone()` and `generate_custom_voice()`

## Debugging Model Structure

To inspect the actual model structure:

```bash
PYTHONPATH="." python scripts/inspect_qwen3tts_structure.py
```

This script recursively prints the module tree to help understand the hierarchy.

## Common Issues & Solutions

### Issue: "Cannot find layers" Error
**Solution:** Use the custom layer accessor from `Qwen3TTSWrapper.get_layer_accessor()` rather than relying on default layer access patterns.

### Issue: Accessing wrong layers
**Solution:** Always use `model.model.talker.model.layers[i]`, not shortcuts like `model.talker.layers[i]` (the PEFT wrapper adds an extra `.model` level).

### Issue: PEFT wrapper conflicts
**Solution:** The layer accessor automatically unwraps PEFT layers. Don't manually unwrap unless you need the base model for other purposes.

## References

- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [qwen-tts PyPI](https://pypi.org/project/qwen-tts/)
