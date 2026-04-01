# Qwen3-TTS-0.6B Architecture Documentation

**Model Name:** Qwen/Qwen3-TTS-12Hz-0.6B-Base
**Framework:** PyTorch / Transformers 4.57.3
**Model Type:** Conditional Generation (Autoregressive TTS)
**Total Parameters:** ~0.6B
**Date Inspected:** 2026-03-31

---

## Overview

Qwen3-TTS is a two-stage autoregressive text-to-speech model based on the Qwen2 Transformer architecture. It uses a hierarchical codec (16 codebooks) for speech representation and features a dedicated Code Predictor for acoustic refinement.

### Generation Pipeline

```
Text Input → [Talker LLM] → 1st Codebook (Semantic)
           → [Code Predictor] → Remaining 15 Codebooks (Acoustic)
           → [Speech Decoder] → Waveform Output
```

---

## 1. Talker (Language Model)

The Talker is the main phonetic processing component and the primary target for activation mining.

### Model Specs

| Parameter | Value |
| :--- | :--- |
| **Architecture** | Qwen3-TTS Talker (Transformer-based) |
| **Hidden Size** ($d_{model}$) | 1024 |
| **Number of Layers** | 28 |
| **Attention Heads** | 16 |
| **Key-Value Heads** (GQA) | 8 |
| **Head Dimension** | 128 |
| **Intermediate Size (FFN)** | 3072 |
| **Activation Function** | SiLU (Swish) |
| **Position Embeddings** | 32,768 |
| **Max Sequence Length** | 32,768 |
| **RMS Norm ε** | 1e-6 |
| **Rope θ (base)** | 1,000,000 |
| **Rope Scaling** | Multi-dimensional RoPE (MRoPE) |
| **MRoPE Sections** | [24, 20, 20] (temporal, height, width) |
| **Attention Bias** | No |
| **Attention Dropout** | 0.0 |
| **Vocab Size** | 3,072 (codec tokens) |
| **Text Vocab Size** | 151,936 (Qwen tokenizer) |

### Layer Structure

Each Transformer layer contains:
- **Self-Attention Block**
  - Multi-head grouped-query attention (MRoPE-based)
  - 16 query heads, 8 key-value heads
  - Head dimension: 128 (1024 / 8)
  - No dropout, no bias

- **Feed-Forward Network (FFN)**
  - Linear: 1024 → 3072 (SiLU activation)
  - Linear: 3072 → 1024
  - Residual connection

- **Layer Normalization**
  - RMS Norm (ε = 1e-6) before each block

- **Residual Connections**
  - Post-addition normalization

### Position Encoding

**MRoPE (Multi-dimensional Rotary Position Embedding)**
- **Sections:** [24, 20, 20]
  - Dimension 0–23 (24 dims): Temporal dimension
  - Dimension 24–43 (20 dims): Height/vertical position
  - Dimension 44–63 (20 dims): Width/horizontal position
- **Rope Theta:** 1,000,000 (base frequency)
- **Interleaved:** True (applies to consecutive dims)

This multi-dimensional approach allows the model to encode 3D positional information, useful for modeling temporal and spatial aspects of speech synthesis.

### Embeddings

- **Input Embeddings:** Text tokens + codec tokens (separate vocabularies)
- **Speaker Encoder (Optional):** 1024-dimensional speaker embeddings
- **Token Embeddings:** Learnable lookup tables

---

## 2. Code Predictor

A lightweight module that refines the semantic tokens (1st codebook) predicted by the Talker into the remaining 15 acoustic codebooks.

### Model Specs

| Parameter | Value |
| :--- | :--- |
| **Architecture** | Qwen3-TTS Talker Code Predictor |
| **Hidden Size** | 1024 |
| **Number of Layers** | 5 |
| **Attention Heads** | 16 |
| **Key-Value Heads** (GQA) | 8 |
| **Head Dimension** | 128 |
| **Intermediate Size (FFN)** | 3072 |
| **Layer Types** | Full Attention (all 5 layers) |
| **Max Sequence Length** | 65,536 |
| **Num Code Groups** | 16 (one per codebook) |
| **Vocab Size** | 2,048 (codec token vocab) |
| **Attention Bias** | No |
| **Attention Dropout** | 0.0 |
| **Cache Support** | Yes (use_cache=True) |

### Purpose

- Takes the Talker's 1st codebook prediction as input
- Autoregressively predicts the remaining 15 codebooks
- Captures acoustic details: formants, voicing, duration, etc.
- Lighter weight (5 layers) than Talker for efficiency

---

## 3. Speech Tokenizer

A causal convolutional autoencoder that quantizes speech into discrete tokens across 16 codebooks.

### Specifications

| Parameter | Value |
| :--- | :--- |
| **Type** | Causal Convolutional Autoencoder |
| **Sample Rate** | 24 kHz |
| **Frame Rate** | 12 Hz (one token per ~80ms = 24,000/12) |
| **Codebooks** | 16 |
| **Token Vocabulary per Codebook** | ~2,048 tokens |
| **Architecture** | Causal convolutions (no lookahead) |

The tokenizer is **frozen** during generation; it's used only for offline encoding of ground-truth audio during training.

---

## 4. Speaker Encoder

Optional module for voice cloning and speaker control.

| Parameter | Value |
| :--- | :--- |
| **Embedding Dimension** | 1024 |
| **Sample Rate** | 24 kHz |
| **Type** | Learned speaker embeddings |

---

## 5. Special Token IDs

These tokens have reserved roles in the model:

| Token Name | ID | Purpose |
| :--- | :--- | :--- |
| **im_start** | 151644 | Marks beginning of instruction/prompt |
| **im_end** | 151645 | Marks end of instruction/prompt |
| **assistant** | 77091 | Identifies assistant role in multi-turn dialogue |
| **tts_bos** | 151672 | TTS Begin-of-Sequence (text input start) |
| **tts_eos** | 151673 | TTS End-of-Sequence (generation complete) |
| **tts_pad** | 151671 | Padding token |
| **codec_bos** | 2149 | Codec sequence begin |
| **codec_eos** | 2150 | Codec sequence end |
| **codec_pad** | 2148 | Codec padding |
| **codec_think** | 2154 | Reasoning/thinking token (internal) |
| **codec_nothink** | 2155 | Non-thinking state |
| **codec_think_bos** | 2156 | Thinking sequence start |
| **codec_think_eos** | 2157 | Thinking sequence end |

### Language-Specific Codec IDs

The model supports 10 languages with dedicated codec tokens:

| Language | Codec ID |
| :--- | :--- |
| English | 2050 |
| Chinese | 2055 |
| Spanish | 2054 |
| German | 2053 |
| Portuguese | 2071 |
| Italian | 2070 |
| Japanese | 2058 |
| Korean | 2064 |
| French | 2061 |
| Russian | 2069 |

---

## 6. Generation Configuration

Default sampling parameters for inference:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **do_sample** | true | Use sampling instead of greedy decoding |
| **temperature** | 0.9 | Softmax temperature (lower = more deterministic) |
| **top_p** | 1.0 | Nucleus sampling parameter (disabled) |
| **top_k** | 50 | Top-K sampling (keep top 50 tokens) |
| **repetition_penalty** | 1.05 | Penalize repeated tokens (small penalty) |
| **subtalker_temperature** | 0.9 | Code Predictor temperature |
| **subtalker_top_k** | 50 | Code Predictor top-K |
| **max_new_tokens** | 8192 | Maximum generation length |

---

## 7. Activation Mining Target

For mechanistic interpretability (Phase 1 of Phonetic SAE project):

### Target Layers

**Layers 1–7** (First 25% of 28 layers)
- These layers handle early phonetic decoding
- Grapheme-to-phoneme mapping occurs here
- Minimal influence from later prosodic modeling

### Hook Points

1. **MLP Post-Activation** (Recommended)
   - Shape: `(batch, seq_len, 1024)`
   - Captures non-linear phonetic transformations
   - Less redundant than residual stream

2. **Residual Stream** (Alternative)
   - Shape: `(batch, seq_len, 1024)`
   - Contains all information from earlier layers
   - Higher reconstruction quality but more polysemanticity

### Expected Activation Statistics

- **Mean:** ~0.0 (zero-centered by residual connections)
- **Std Dev:** ~0.5–1.0 (SiLU introduces slight sparsity)
- **Min/Max:** Roughly symmetric around 0
- **Sparsity:** Low (dense activations, no built-in sparsity)

### Storage Requirements (50K sentences)

- **FP32:** 50M vectors × 1024 floats × 4 bytes = ~200 GB
- **FP16:** 50M vectors × 1024 floats × 2 bytes = ~100 GB
- **INT8:** 50M vectors × 1024 ints × 1 byte = ~50 GB

---

## 8. Key Architectural Insights

### Why MRoPE?

Multi-dimensional RoPE allows the model to encode:
- **Temporal ordering** of phonemes within an utterance
- **2D/3D structure** if input is framed (e.g., mel-spectrograms)
- Better generalization to longer sequences

### Why GQA (Grouped-Query Attention)?

- Reduces memory and computation without much quality loss
- 8 KV heads shared across 16 query heads
- Critical for fitting a 1024-d model on consumer GPUs

### Why Two Stages?

1. **Talker (28 layers):** Phonetic clarity and semantic content
2. **Code Predictor (5 layers):** Acoustic refinement (formants, voicing, duration)

This separation allows:
- Lighter fine-tuning of acoustic details without retraining the whole Talker
- Better interpretability (phonetic vs. acoustic features are separated)

---

## 9. References

- **Model Repo:** [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- **Model Card:** [Qwen/Qwen3-TTS-12Hz-0.6B-Base (HuggingFace)](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
- **Technical Report:** [arXiv 2601.15621](https://arxiv.org/abs/2601.15621)
- **Base Architecture:** Qwen2 Transformer with MRoPE

---

## 10. Files in Checkpoint

```
Qwen3-TTS-0.6B/
├── config.json                    # Main architecture config
├── generation_config.json         # Sampling parameters
├── preprocessor_config.json       # Audio preprocessing
├── model.safetensors              # Model weights (safetensors format)
├── tokenizer_config.json          # Text tokenizer config
├── vocab.json                     # BPE vocabulary
├── merges.txt                     # BPE merge rules
└── speech_tokenizer/              # Speech codec weights
    └── [codec model files]
```

---

**End of Qwen3-TTS-0.6B Architecture Documentation**
