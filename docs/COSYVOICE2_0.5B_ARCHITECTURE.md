# CosyVoice2-0.5B Architecture Documentation

**Model Name:** FunAudioLLM/CosyVoice2-0.5B
**Framework:** PyTorch + Hydra Configuration
**Model Type:** Streaming-Capable Conditional Flow Matching TTS
**Total Parameters:** ~0.5B
**Date Inspected:** 2026-03-31

---

## Overview

CosyVoice2 is a state-of-the-art streaming text-to-speech model combining:
1. **Qwen2.5-0.5B LLM** — Semantic token generation from text
2. **Causal Flow Matching (CFM)** — Mel-spectrogram generation conditioned on semantic tokens
3. **HiFi-GAN Vocoder (HiFT)** — Waveform synthesis from mel-spectrograms
4. **Speaker Encoder (CampPlus)** — Speaker embedding extraction

This three-stage pipeline enables **unified streaming and non-streaming synthesis** with support for speaker control and voice cloning.

### Generation Pipeline

```
Text Input → [Qwen2.5 LLM] → Speech Tokens (6561 vocabulary)
           → [Causal CFM] → Mel-Spectrogram (80 channels, variable length)
           → [HiFT Vocoder] → Waveform Output (24 kHz)
```

---

## 1. Qwen2.5-0.5B LLM Backbone

The semantic token generation stage, responsible for phonetic processing and text understanding.

### Model Specs

| Parameter | Value |
| :--- | :--- |
| **Base Architecture** | Qwen2.5-0.5B (Qwen LM family) |
| **Hidden Size** ($d_{model}$) | 896 |
| **Number of Layers** | 24 |
| **Attention Heads** (Query) | 14 |
| **Key-Value Heads** (GQA) | 2 |
| **Head Dimension** | 64 (896 / 14) |
| **Intermediate Size (FFN)** | ~2,400 (approx. 2.7× hidden size) |
| **Activation Function** | SiLU (Swish) |
| **Position Embeddings** | RoPE (Rotary Position Embeddings) |
| **Max Sequence Length** | Model-dependent (typically 32K+) |
| **RMS Norm ε** | 1e-6 |
| **Attention Bias** | No |
| **Attention Dropout** | 0.0 |
| **Text Vocab Size** | Qwen2.5 standard tokenizer |
| **Speech Token Vocab Size** | 6,561 (output to speech tokens) |

### Layer Structure

Each Transformer layer contains:
- **Self-Attention Block**
  - Grouped-query attention (GQA) with 14 Q heads, 2 KV heads
  - Head dimension: 64
  - RoPE position encoding
  - No dropout, no bias

- **Feed-Forward Network (FFN)**
  - SiLU activation (Swish)
  - Intermediate expansion ~2.7×

- **Layer Normalization**
  - RMS Norm before each block

- **Residual Connections**
  - Post-addition normalization

### Input Configuration

| Component | Value |
| :--- | :--- |
| **LLM Input Size** | 896 |
| **LLM Output Size** | 896 |
| **Output Projection** | Linear projection to 6,561-dim speech token space |

---

## 2. Causal Flow Matching (CFM) Module

Generates mel-spectrograms from speech tokens using diffusion-based flow matching with causal masking for streaming synthesis.

### Architecture Overview

```
Speech Tokens → [Encoder] → Token Embeddings
                          ↓
                  [CFM Decoder] → Mel-Spectrogram
                  ↑
            Speaker Embedding (192-dim)
```

### CFM Parameters

| Parameter | Value | Purpose |
| :--- | :--- | :--- |
| **Input Size** | 512 | Token embedding dimension |
| **Output Size** | 80 | Mel-spectrogram channels |
| **Speaker Embed Dim** | 192 | Speaker conditioning |
| **Vocab Size** | 6,561 | Number of speech tokens |
| **Output Type** | 'mel' | Mel-spectrogram output |
| **Token Frame Rate** | 25 Hz | Speech token frame rate |
| **Token-Mel Ratio** | 2:1 | Mel frames per token frame |
| **Pre-Lookahead Length** | 3 frames | Context for causal decoding |
| **Only Mask Loss** | True | Use causal masking only |

### Encoder: UpsampleConformerEncoder

Embeds speech tokens and upsamples to mel-spectrogram rate.

| Parameter | Value |
| :--- | :--- |
| **Input Layer** | Linear |
| **Output Size** | 512 |
| **Attention Heads** | 8 |
| **Linear Units (FFN)** | 2,048 |
| **Number of Blocks** | 6 |
| **Positional Encoding** | ESPNet Relative Position (`rel_pos_espnet`) |
| **Attention Layer Type** | Relative Self-Attention (`rel_selfattn`) |
| **Dropout Rate** | 0.1 |
| **Attention Dropout** | 0.1 |
| **Positional Dropout** | 0.1 |
| **Normalize Before** | True (pre-norm) |
| **CNN Module** | False (no convolutional blocks) |
| **Macaron Style** | False (standard FFN) |
| **Static Chunk Size** | 25 tokens (streaming chunk size) |

**Purpose:** Converts discrete speech tokens to continuous embeddings and upsamples to mel resolution.

### Decoder: CausalConditionalCFM

Diffusion-based decoder using flow matching with causal constraints for streaming compatibility.

#### Main Decoder

| Parameter | Value |
| :--- | :--- |
| **In Channels** | 240 (80 mels × 3 history frames) |
| **Number of Speakers** | 1 (speaker-agnostic, conditioned via embedding) |
| **Speaker Embed Dim** | 80 (projected from 192) |
| **Solver** | Euler (ODE solver) |
| **Time Scheduler** | Cosine annealing |
| **Sigma Min** | 1e-6 (diffusion endpoint) |
| **Training CFG Rate** | 0.2 (classifier-free guidance) |
| **Inference CFG Rate** | 0.7 (stronger guidance at inference) |
| **Regularization Loss** | L1 norm |

#### Estimator: CausalConditionalDecoder

| Parameter | Value |
| :--- | :--- |
| **In Channels** | 320 (80 mels × 4 context frames) |
| **Out Channels** | 80 (predicted mel frame) |
| **Internal Channels** | [256] |
| **Attention Head Dimension** | 64 |
| **Number of Attention Heads** | 8 |
| **Number of Blocks** | 4 (residual blocks) |
| **Mid Blocks** | 12 (diffusion time embedding blocks) |
| **Activation** | GELU |
| **Dropout** | 0.0 |
| **Static Chunk Size** | 50 frames (25 tokens × 2 mel-ratio) |
| **Decoding Left Chunks** | -1 (use all left context) |

**Purpose:** Iteratively denoise noisy mel-spectrograms conditioned on speech tokens and speaker.

### Flow Matching Configuration

- **Causal Masking:** Prevents attending to future frames (streaming compatibility)
- **Length Normalization:** True (normalize loss by sequence length)
- **Mix Ratio:** [5, 15] (mixture of teacher forcing and autoregressive decoding)
- **Sampling:** RAS (Residual Autoregressive Sampling)
  - Top-P: 0.8
  - Top-K: 25
  - Window Size: 10
  - Tau (temperature scaling): 0.1

---

## 3. HiFT Vocoder (Waveform Synthesis)

Neural vocoder that converts mel-spectrograms to waveforms with speaker-dependent NSF (Neural Source Filtering).

### Generator: HiFTGenerator

| Parameter | Value |
| :--- | :--- |
| **In Channels** | 80 (mel-spectrogram) |
| **Base Channels** | 512 |
| **Number of Harmonics (NSF)** | 8 |
| **Sampling Rate** | 24 kHz |
| **NSF Alpha** | 0.1 (source noise ratio) |
| **NSF Sigma** | 0.003 (noise scale) |
| **NSF Voiced Threshold** | 10 dB |
| **Upsample Rates** | [8, 5, 3] (total 120× upsampling) |
| **Upsample Kernel Sizes** | [16, 11, 7] |
| **ResBlock Kernel Sizes** | [3, 7, 11] |
| **ResBlock Dilation Sizes** | [[1, 3, 5], [1, 3, 5], [1, 3, 5]] |
| **ISTFT (Inverse STFT)** | n_fft=16, hop_len=4 |
| **Source ResBlock Kernels** | [7, 7, 11] (for NSF module) |
| **LReLU Slope** | 0.1 |
| **Audio Limit** | 0.99 (clipping threshold) |

#### F0 Predictor (Pitch Extraction)

Used by NSF module:

| Parameter | Value |
| :--- | :--- |
| **Type** | ConvRNNF0Predictor |
| **In Channels** | 80 (mel input) |
| **Condition Channels** | 512 (from generator) |
| **Output Classes** | 1 (continuous F0) |

**Purpose:** Predicts fundamental frequency for neural source filtering, enabling more natural voicing and pitch control.

### Discriminator: MultipleDiscriminator

Adversarial training architecture:

| Parameter | Value |
| :--- | :--- |
| **Multi-Period Discriminator** | Standard HiFi-GAN MPD |
| **Multi-Resolution Spec Discriminator** | Custom MRSD |
| **Mel Transform** | 80-channel mel-spectrogram (n_fft=1920, hop=480) |

---

## 4. Speaker Encoder (CampPlus)

Extracts speaker embeddings from voice prompts, enabling voice cloning and speaker control.

### Specifications

| Parameter | Value |
| :--- | :--- |
| **Type** | CampPlus (speaker verification encoder) |
| **Output Embedding Dim** | 192 |
| **Input** | Raw audio or mel-spectrogram |
| **Architecture** | ResNet-style with attention pooling |

---

## 5. Activation Mining Target

For mechanistic interpretability (Phase 1 of Phonetic SAE project):

### Target Layers

**Layers 1–6** (First 25% of 24 layers)
- Early phonetic and semantic processing
- Grapheme-to-phoneme and script understanding
- Less influenced by prosodic modeling

### Hook Points

1. **MLP Post-Activation** (Recommended)
   - Shape: `(batch, seq_len, 896)`
   - Captures non-linear phonetic transformations
   - Less redundant than residual stream

2. **Residual Stream** (Alternative)
   - Shape: `(batch, seq_len, 896)`
   - Contains all upstream information
   - Higher reconstruction quality

### Expected Activation Statistics

- **Mean:** ~0.0 (zero-centered by residual connections)
- **Std Dev:** ~0.4–0.7 (SiLU with narrower hidden size)
- **Min/Max:** Roughly symmetric
- **Sparsity:** Low (dense activations)

### Storage Requirements (50K sentences)

- **FP32:** 50M vectors × 896 floats × 4 bytes = ~180 GB
- **FP16:** 50M vectors × 896 floats × 2 bytes = ~90 GB
- **INT8:** 50M vectors × 896 ints × 1 byte = ~45 GB

---

## 6. Feature Processing Pipeline

### Audio Processing

| Step | Configuration |
| :--- | :--- |
| **Sample Rate** | 24 kHz |
| **Mel-Spectrogram** | n_fft=1920, hop=480, num_mels=80, fmax=8000 Hz |
| **Frame Rate** | 50 Hz (hop_size=480 at 24 kHz) |
| **F0 Extraction** | Continuous F0 tracking, 10 Hz threshold |

### Sequence Processing

| Step | Value |
| :--- | :--- |
| **Max Sequence Length** | 40,960 tokens |
| **Min Sequence Length** | 100 tokens |
| **Token Max Length** | 200 (text tokens) |
| **Token Min Length** | 1 |
| **Max Audio Length** | 24,480 frames (~1 second) |
| **Batch Type** | Dynamic (max 2000 frames per batch) |

---

## 7. Streaming Configuration

For real-time synthesis without waiting for full utterance:

| Parameter | Value | Purpose |
| :--- | :--- | :--- |
| **Chunk Size** | 25 tokens | Process tokens in 25-token chunks |
| **Token Frame Rate** | 25 Hz | One token per 40ms |
| **Decoding Left Chunks** | -1 (all) | Use all past context for causal flow |
| **Vocoder Chunk** | 50 mel frames | Process 50 mel frames at a time |

This configuration enables low-latency streaming while maintaining quality by allowing causal access to all past tokens.

---

## 8. Training Configuration

### LLM + Flow Training

| Parameter | Value |
| :--- | :--- |
| **Optimizer** | Adam |
| **Learning Rate** | 1e-5 |
| **Scheduler** | Constant LR (during supervised fine-tuning) |
| **Warmup Steps** | 2500 |
| **Max Epochs** | 200 |
| **Gradient Clipping** | 5.0 |
| **Accumulation Steps** | 2 |
| **Log Interval** | Every 100 steps |

### Adversarial (GAN) Training

| Parameter | Value |
| :--- | :--- |
| **Generator Optimizer** | Adam, lr=0.0002 |
| **Discriminator Optimizer** | Adam, lr=0.0002 |
| **Accumulation Steps** | 1 (must be 1 for GAN stability) |
| **Max Epochs** | 200 |
| **Log Interval** | Every 100 steps |

---

## 9. Key Architectural Insights

### Why Three Stages?

1. **LLM (Qwen2.5):** Semantic understanding and phonetic extraction
   - Leverages large-scale language understanding
   - Separates language processing from acoustic synthesis

2. **Causal CFM:** Mel-spectrogram generation with streaming support
   - Flow matching (diffusion-based) is efficient and high-quality
   - Causal masking enables streaming synthesis
   - Conditional on speaker embedding for voice control

3. **HiFT Vocoder:** Speaker-dependent waveform synthesis
   - Neural source filtering captures speaker-specific voicing
   - Harmonic modeling improves pitch naturalness
   - Trained with adversarial losses for human-like quality

### Why Causal Flow Matching?

- **Streaming-compatible:** Tokens processed in causal chunks without future lookahead
- **Diffusion-based:** More stable than autoregressive decoding
- **Classifier-free guidance:** Stronger at inference (0.7) than training (0.2) for better quality
- **Flow-matching:** Faster than standard DDPM diffusion

### Why GQA (Grouped-Query Attention)?

- 14 query heads share 2 key-value heads
- Reduces memory requirements without significant quality loss
- Critical for fitting 0.5B model on consumer GPUs

### Why RAS Sampling?

Residual Autoregressive Sampling with window-based selection:
- More stable than naive top-p/top-k
- Window size=10 prevents degenerative sequences
- Tau scaling adapts temperature per token

---

## 10. Comparison with Qwen3-TTS-0.6B

| Aspect | Qwen3-TTS-0.6B | CosyVoice2-0.5B |
| :--- | :--- | :--- |
| **LLM** | Custom Qwen3-TTS (28 layers) | Qwen2.5 (24 layers) |
| **Hidden Size** | 1024 | 896 |
| **Intermediate Size** | 3072 (3×) | ~2400 (2.7×) |
| **Attention Mechanism** | MRoPE (multidimensional) | Standard RoPE |
| **Acoustic Stage** | Code Predictor (5 layers) | Flow Matching (diffusion) |
| **Vocoder** | Causal ConvNet | HiFT (GAN-trained) |
| **Speaker Control** | Speaker embeddings | CampPlus speaker encoder |
| **Streaming** | Not designed for streaming | Streaming-first (causal) |
| **Key Advantage** | Strong phonetic modeling | High-quality voice control + streaming |

---

## 11. Files in Checkpoint

```
CosyVoice2-0.5B/
├── config.json                  # Minimal config (framework/task)
├── configuration.json           # Extended configuration
├── cosyvoice2.yaml              # Full Hydra config (architecture details)
├── llm.pt                        # Qwen2.5-0.5B weights
├── flow.pt                       # CFM module weights
├── hift.pt                       # HiFT vocoder weights
├── speech_tokenizer_v2.onnx      # Speech tokenizer (ONNX format)
├── speech_tokenizer_v2.batch.onnx # Batch-optimized tokenizer
├── campplus.onnx                 # Speaker encoder (CampPlus)
├── flow.decoder.estimator.fp32.onnx # CFM estimator (ONNX)
├── asset/                        # Pre-computed speaker embeddings
│   ├── spk_*.pt                  # Speaker embedding files
│   └── [other assets]
├── CosyVoice-BlankEN/            # Blank/silence audio files
│   └── [audio templates]
└── README.md
```

---

## 12. References

- **Model Repo:** [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- **Project Page:** [CosyVoice2](https://funaudiollm.github.io/cosyvoice2/)
- **Paper:** [arXiv 2412.10117](https://arxiv.org/html/2412.10117v2)
- **Base LLM:** Qwen2.5-0.5B
- **Encoder:** Conformer (ESPNet-style relative attention)
- **Decoder:** Diffusion-based flow matching

---

**End of CosyVoice2-0.5B Architecture Documentation**
