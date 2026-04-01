# Model Architecture Comparison: Qwen3-TTS vs CosyVoice2

A side-by-side comparison of the two target models for mechanistic interpretability research.

---

## 1. Overall Architecture Comparison

### Generation Pipelines

**Qwen3-TTS-0.6B (Two-Stage Codec-Based)**
```
Text → [Talker LLM: 28 layers, 1024-d]
    → [1st Codec Token: semantic]
    → [Code Predictor: 5 layers]
    → [Remaining 15 Codec Tokens: acoustic]
    → [Speech Decoder: ConvNet]
    → Waveform (24 kHz)
```

**CosyVoice2-0.5B (Three-Stage Diffusion-Based)**
```
Text → [Qwen2.5 LLM: 24 layers, 896-d]
    → [Speech Tokens: 6561 vocab]
    → [Causal Flow Matching: diffusion]
    → [Mel-Spectrogram: 80 channels]
    → [HiFT Vocoder: NSF-based GAN]
    → Waveform (24 kHz)
```

---

## 2. LLM Backbone Comparison

| Aspect | Qwen3-TTS-0.6B | CosyVoice2-0.5B |
| :--- | :--- | :--- |
| **Model Name** | Qwen3-TTS Talker | Qwen2.5-0.5B |
| **Total Params** | ~0.6B | ~0.5B |
| **Hidden Size ($d_{model}$)** | 1024 | 896 |
| **Number of Layers** | 28 | 24 |
| **Attention Heads (Q)** | 16 | 14 |
| **Key-Value Heads** | 8 (GQA) | 2 (GQA) |
| **Head Dimension** | 128 (1024/8) | 64 (896/14) |
| **Intermediate (FFN)** | 3072 (3× hidden) | ~2400 (2.7× hidden) |
| **Activation** | SiLU | SiLU |
| **Position Encoding** | MRoPE [24,20,20] | Standard RoPE |
| **Max Seq Length** | 32,768 | 32K+ |
| **Attention Dropout** | 0.0 | 0.0 |
| **Bias** | No | No |

**Key Difference:** Qwen3-TTS uses **multi-dimensional RoPE** (MRoPE) for temporal/spatial encoding; CosyVoice2 uses standard rotary embeddings.

---

## 3. Acoustic Processing Comparison

| Stage | Qwen3-TTS-0.6B | CosyVoice2-0.5B |
| :--- | :--- | :--- |
| **Semantic Representation** | 1st codec token (discrete) | 6561-token continuous vocab |
| **Acoustic Refinement** | Code Predictor (5-layer Transformer) | Causal Flow Matching (diffusion) |
| **Output** | 16 codec tokens (quantized) | 80-mel spectrogram (continuous) |
| **Streaming Support** | Limited (not designed) | Native (causal masking) |
| **Vocoder** | Causal ConvNet | HiFT (GAN with NSF) |
| **Speaker Control** | Speaker embeddings | CampPlus encoder |

**Key Difference:** Qwen3-TTS uses **discrete hierarchical codecs**; CosyVoice2 uses **continuous diffusion** for mel-spectrograms.

---

## 4. Phonetic Processing Target Layers

### Qwen3-TTS-0.6B

| Layer Range | Purpose | Hook Point |
| :--- | :--- | :--- |
| **Layers 1–7** (of 28) | Grapheme→Phoneme, early phonetic processing | MLP post-activation or residual stream |
| **Layers 8–14** | Coarticulation, allophonic variation | Secondary targets |
| **Layers 15–28** | Prosody, duration, speaker identity | Not targeted for phonetic SAE |

### CosyVoice2-0.5B

| Layer Range | Purpose | Hook Point |
| :--- | :--- | :--- |
| **Layers 1–6** (of 24) | Grapheme→Phoneme, early phonetic processing | MLP post-activation or residual stream |
| **Layers 7–12** | Semantic refinement, phonetic context | Secondary targets |
| **Layers 13–24** | Task-specific modeling, conditioning | Not targeted |

---

## 5. Activation Shapes & Sizes

### Qwen3-TTS-0.6B

| Property | Value |
| :--- | :--- |
| **Layer Activation Shape** | (batch, seq_len, 1024) |
| **Per-Vector Size** | 1024 floats |
| **FP32 per Vector** | 4 KB |
| **FP16 per Vector** | 2 KB |
| **INT8 per Vector** | 1 KB |
| **50M Vector Activation Set (FP32)** | ~200 GB |
| **50M Vector Activation Set (FP16)** | ~100 GB |
| **50M Vector Activation Set (INT8)** | ~50 GB |

### CosyVoice2-0.5B

| Property | Value |
| :--- | :--- |
| **Layer Activation Shape** | (batch, seq_len, 896) |
| **Per-Vector Size** | 896 floats |
| **FP32 per Vector** | 3.6 KB |
| **FP16 per Vector** | 1.8 KB |
| **INT8 per Vector** | 0.9 KB |
| **50M Vector Activation Set (FP32)** | ~180 GB |
| **50M Vector Activation Set (FP16)** | ~90 GB |
| **50M Vector Activation Set (INT8)** | ~45 GB |

---

## 6. SAE Configuration Recommendations

### Qwen3-TTS-0.6B

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Expansion Factor** | 16× (16,384-d) or 32× (32,768-d) | Larger model = more expressivity |
| **Sparsity (K)** | 32 | Standard for phonetic features |
| **SAE Params** | 33.6M (16×) or 67.2M (32×) | Fits on RTX 3090/4090 |
| **Training Batch Size** | 4096–8192 vectors | ~100–200 MB per batch in FP16 |
| **Estimated Training Tokens** | 5B vectors | ~100 passes over 50M activation set |

### CosyVoice2-0.5B

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Expansion Factor** | 16× (14,336-d) or 32× (28,672-d) | Slightly smaller model |
| **Sparsity (K)** | 32 | Standard for phonetic features |
| **SAE Params** | 25.7M (16×) or 51.4M (32×) | Fits comfortably on consumer GPU |
| **Training Batch Size** | 4096–8192 vectors | ~90–180 MB per batch in FP16 |
| **Estimated Training Tokens** | 5B vectors | ~100 passes over 50M activation set |

---

## 7. Hardware Requirements (RTX 3090 / 4090, 24 GB VRAM)

### Phase 1: Activation Mining

| Operation | Qwen3-TTS | CosyVoice2 |
| :--- | :--- | :--- |
| **Model Inference** | ~2 GB | ~1.5 GB |
| **Activation Buffer** | ~1 GB | ~1 GB |
| **Total** | ~3–4 GB | ~2.5–3.5 GB |
| **Time for 50K sentences** | ~4–6 hours | ~3–5 hours |

### Phase 2: SAE Training

| Operation | Value |
| :--- | :--- |
| **SAE Model (16×)** | ~67 MB (Qwen3) or ~51 MB (CosyVoice2) |
| **Activation Batch** | ~16 MB (8192 vectors) |
| **Optimizer States** | ~200 MB |
| **Total** | ~300 MB (very comfortable) |
| **Disk I/O Bottleneck** | NVMe SSD recommended |

### Phase 4: Causal Intervention

| Operation | Value |
| :--- | :--- |
| **Model Inference** | ~4 GB (full model) |
| **SAE + Patching** | ~100 MB |
| **Total** | ~4.5 GB |
| **Room for Batching** | Yes, 2–4 utterances at a time |

---

## 8. Phonetic Feature Discovery Potential

### Qwen3-TTS-0.6B Expectations

**Strengths:**
- 28 layers allow fine-grained layer-wise analysis
- 1024-d hidden states → larger SAE (up to 32,768-d)
- MRoPE may capture positional phonetic patterns
- Two-stage architecture cleanly separates semantic (Talker) from acoustic (Code Predictor)

**Challenges:**
- Discrete codec tokens may require special handling
- More layers = more potential polysemanticity

### CosyVoice2-0.5B Expectations

**Strengths:**
- Qwen2.5 backbone → stable, well-understood architecture
- Continuous speech token representation (6561 vocab) cleaner than hierarchical codecs
- Causal flow matching architecture may have specialized acoustic patterns
- Smaller model → fewer dead features in SAE

**Challenges:**
- 24 layers < 28 layers (less granularity)
- CFM decoder is separate from LLM (less direct phonetic→acoustic coupling)

---

## 9. Cross-Model Analysis Strategy

### Comparing Feature Spaces

| Approach | Purpose |
| :--- | :--- |
| **Centered Kernel Alignment (CKA)** | Measure layer-wise similarity between models |
| **Feature-Phoneme Correlation** | Identify monosemantic phonetic features in each |
| **Overlap Analysis** | Which phonemes are encoded similarly? |

### Expected Insights

1. **Early layers (1–6):** Likely to show high CKA alignment (grapheme→phoneme is universal)
2. **Acoustic models differ:** Qwen3's codec vs. CosyVoice2's mel differ fundamentally
3. **Phonetic universals:** Plosive, fricative, nasal features should emerge in both

---

## 10. Implementation Checklist

### For Qwen3-TTS-0.6B

- [ ] Load model from `pretrained_models/Qwen3-TTS-0.6B/model.safetensors`
- [ ] Hook `model.talker.layers[i].mlp` for i ∈ [1,7]
- [ ] Extract activations at shape (seq_len, 1024)
- [ ] Quantize to FP16 for storage
- [ ] Train SAE with 16× or 32× expansion

### For CosyVoice2-0.5B

- [ ] Load model from `pretrained_models/CosyVoice2-0.5B/llm.pt`
- [ ] Hook `model.llm.layers[i].mlp` for i ∈ [1,6]
- [ ] Extract activations at shape (seq_len, 896)
- [ ] Quantize to FP16 for storage
- [ ] Train SAE with 16× or 32× expansion

### Shared Infrastructure

- [ ] Generic `ActivationHook` class (works for both)
- [ ] `ShuffledActivationBuffer` (unified data loading)
- [ ] Top-K SAE training loop (unified SAE trainer)
- [ ] Feature-phoneme correlation analysis (shared utility)

---

## 11. Key Takeaways

| Aspect | Winner | Why |
| :--- | :--- | :--- |
| **Phonetic Processing Power** | Qwen3-TTS (28 layers) | More layers, larger hidden size |
| **Ease of Interpretation** | CosyVoice2 (cleaner outputs) | Continuous mel tokens vs. discrete codecs |
| **Hardware Efficiency** | CosyVoice2 (smaller) | Uses slightly fewer VRAM |
| **Streaming Compatibility** | CosyVoice2 (built-in) | Causal masking native |
| **Speaker Control** | CosyVoice2 (explicit encoder) | Dedicated speaker embeddings |
| **Research Potential** | Both comparable | Different architectural paradigms |

---

**Recommendation:** Train SAEs on **both models** to compare phonetic feature discovery across architectural styles. Qwen3-TTS for phonetic depth; CosyVoice2 for acoustic clarity.

---

End of Model Comparison Document
