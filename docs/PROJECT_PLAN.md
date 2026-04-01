# Phonetic SAE — Research Roadmap

**Project:** Mechanistic Interpretability for Phonetic Layers in LLM-Based TTS
**Target Models:** Qwen3-TTS (0.6B / 1.7B) and CosyVoice2 (0.5B)
**Hardware:** Consumer GPU — NVIDIA RTX 3090 / 4090 (24 GB VRAM)
**Timeline:** 6 weeks (adjusted from 4 weeks for consumer GPU constraints)
**Last Updated:** 2026-03-30

---

## Table of Contents

1. [Architecture Reference](#1-architecture-reference)
2. [Phase 1 — Activation Mining (Weeks 1–2)](#2-phase-1--activation-mining-weeks-12)
3. [Phase 2 — SAE Training (Weeks 2–3)](#3-phase-2--sae-training-weeks-23)
4. [Phase 3 — Phonetic Feature Mapping (Week 4)](#4-phase-3--phonetic-feature-mapping-week-4)
5. [Phase 4 — Causal Intervention & Steering (Week 5)](#5-phase-4--causal-intervention--steering-week-5)
6. [Phase 5 — Cross-Architecture Distillation (Week 6)](#6-phase-5--cross-architecture-distillation-week-6)
7. [Hardware Budget & Memory Planning](#7-hardware-budget--memory-planning)
8. [Risk Register & Decision Log](#8-risk-register--decision-log)
9. [Weekly Milestones](#9-weekly-milestones)
10. [References](#10-references)

---

## 1. Architecture Reference

Understanding the exact internal structure of both target models is critical for choosing hook points and sizing the SAE.

### 1.1 Qwen3-TTS

Qwen3-TTS is an autoregressive TTS model built on the Qwen2 Transformer architecture with a two-stage generation pipeline.

| Component | Qwen3-TTS-0.6B | Qwen3-TTS-1.7B |
| :--- | :--- | :--- |
| **Talker (LLM)** | 28 Transformer layers | 28 Transformer layers |
| Talker hidden_size ($d_{model}$) | 1024 | 2048 |
| Talker attention | MRoPE, GQA | MRoPE, GQA |
| MRoPE section | [24, 20, 20] (temporal, h, w) | [24, 20, 20] |
| **Code Predictor** | 5 layers, $d=1024$, 16 heads | 5 layers, $d=1024$, 16 heads |
| Speech Tokenizer | 12 Hz, 16-codebook (causal ConvNet) | 12 Hz, 16-codebook |
| Embeddings | Dual (text + codec) | Dual (text + codec) |
| **Total Params** | ~0.6B | ~1.7B |

**Generation Flow:**
1. **Talker** autoregressively generates the 1st codebook token (semantic) from text
2. **Code Predictor** generates the remaining 15 acoustic codebook tokens per frame
3. **Decoder** (causal ConvNet) reconstructs the waveform from all 16 codebooks

**Hook Strategy for 0.6B:** Layers 1–7 of the Talker (first 25% of 28 layers). The Talker is where grapheme-to-phoneme mapping happens; the Code Predictor is downstream acoustic refinement.

### 1.2 CosyVoice2

CosyVoice2 replaces the original custom TransformerLM with a pre-trained Qwen2.5-0.5B backbone.

| Component | CosyVoice2-0.5B |
| :--- | :--- |
| **LLM Backbone** | Qwen2.5-0.5B |
| LLM hidden_size ($d_{model}$) | 896 |
| LLM layers | 24 |
| LLM attention heads | 14 Q, 2 KV (GQA) |
| **Speech Tokenizer** | 6 Transformer blocks + RoPE (Encoder1) |
| Quantization | Finite-Scalar Quantization (FSQ) |
| **Flow Matching** | Chunk-aware causal CFM → Mel spectrogram |
| Streaming | Yes (unified streaming + non-streaming) |
| **Total Params** | ~500M |

**Generation Flow:**
1. **Qwen2.5 LLM** predicts semantic speech tokens from text + optional prompt
2. **CFM Model** converts speech tokens → Mel spectrogram (conditioned on speaker embedding)
3. **Vocoder** (HiFi-GAN variant) synthesizes waveform from Mel

**Hook Strategy:** Layers 1–6 of the Qwen2.5 backbone (first 25% of 24 layers). The LLM is the phonetic processing stage; the CFM handles acoustic rendering.

### 1.3 Comparison — What This Means for SAE Design

| Dimension | Qwen3-TTS-0.6B | CosyVoice2-0.5B |
| :--- | :--- | :--- |
| $d_{model}$ | 1024 | 896 |
| Target layers | 1–7 (of 28) | 1–6 (of 24) |
| Hook point | Talker MLP post-activation or residual stream | Qwen2.5 MLP post-activation or residual stream |
| SAE $d_{sae}$ at 16× | 16,384 | 14,336 |
| SAE $d_{sae}$ at 32× | 32,768 | 28,672 |

**Key Insight:** Both models use Qwen-family Transformers as their LLM backbone. This means the hook implementation and SAE architecture can share a common codebase, with only the layer indices and dimensions differing.

---

## 1.4 Multilingual Dataset Support (Optional)

The project supports training on multilingual datasets (e.g., Mandarin, Cantonese, English) in addition to or instead of LibriTTS-R. This enables:

**Research Benefits:**
- **Language-universal phonetic features:** Discover features shared across languages (e.g., plosives, fricatives)
- **Language-specific patterns:** Identify features that specialize by language (e.g., tones in Mandarin/Cantonese)
- **Cross-lingual generalization:** Use CKA to measure feature similarity across languages
- **Richer interventions:** Test pronunciation steering on diverse phoneme sets and language-specific errors

**Implementation Strategy:**

1. **Activation Mining (Phase 1):**
   - Label activations by language during capture
   - Optionally balance languages (e.g., ~16K sentences per language for 48K total)
   - Store metadata: language, text, phoneme alignments

2. **Feature Mapping (Phase 3):**
   - Compute feature-phoneme correlations **per language**
   - Create separate heatmaps: P(feature | phoneme) for Mandarin, Cantonese, English
   - Identify **monosemantic-within-language** vs. **polysemantic-across-languages** features
   - Analyze layer-wise language sensitivity (do early layers show language-specific patterns?)

3. **Distillation (Phase 5):**
   - Train multilingual Student to match Teacher's features across all languages
   - Optional: language-conditioned SAE (embed language as metadata during training)

**Dataset Format:**
- Text-speech pairs with language labels
- Phoneme alignments (can use MFA for English, language-specific tools for Mandarin/Cantonese)
- Minimum recommended: 10K+ sentences per language; ideal: 50K+ total balanced across languages

**TBD:** Dataset details to be provided by user (size, format, language distribution, existing alignments).

---

## 2. Phase 1 — Activation Mining (Weeks 1–2)

### 2.1 Objective

Capture the internal activations of the Talker / LLM backbone during inference on a phonetically diverse dataset. These activations become the training data for the SAE.

### 2.2 Milestones

**Week 1:**

- **M1.1 — Environment Setup**
  - Install Qwen3-TTS and CosyVoice2 inference pipelines
  - Verify inference runs on 3090/4090 (both models fit in 24 GB for inference)
  - Download LibriTTS-R (clean-360 subset, ~360 hours, high phonetic diversity)
  - Prepare a smaller dev set: 1,000 sentences from TIMIT for rapid iteration

- **M1.2 — Hook Implementation**
  - Write a generic `ActivationHook` class that attaches to any `nn.Module`
  - For Qwen3-TTS: hook `model.talker.layers[i].mlp` (post-activation) for $i \in [1..7]$
  - For CosyVoice2: hook `model.llm.layers[i].mlp` (post-activation) for $i \in [1..6]$
  - Validate: check captured tensor shapes match expected $(\text{batch}, \text{seq\_len}, d_{model})$

- **M1.3 — Pilot Capture**
  - Run 100 sentences through each model
  - Inspect activation statistics: mean, std, min, max, sparsity
  - Decide on hook point: MLP post-activation vs. residual stream (compare explained variance)
  - **Decision Point:** Which hook point gives richer phonetic signal? Residual stream preserves more information; MLP post-activation isolates the non-linear transformation.

**Week 2:**

- **M1.4 — Full-Scale Capture**
  - Run 50,000+ sentences through each model
  - **Data source:** LibriTTS-R (clean-360, English, ~360 hours) OR custom multilingual dataset (Mandarin/Cantonese/English, TBD)
  - Store activations as FP16 tensors (half the disk cost of FP32)
  - Implement a streaming buffer: process in batches of 512, flush to disk periodically
  - Target: ~50M activation vectors per model (depends on average sequence length)
  - If multilingual: capture separately per language, label activations with language_id
  - **Storage Estimate:**
    - Qwen3-TTS: 50M × 1024 × 2 bytes = ~100 GB
    - CosyVoice2: 50M × 896 × 2 bytes = ~85 GB
    - Consider quantizing to INT8 to halve storage (validate reconstruction quality first)

- **M1.5 — Activation Quality Audit**
  - Plot activation distributions per layer (histograms, PCA projections)
  - Check for degenerate patterns (all-zero rows, extreme outliers)
  - Compare Qwen3-TTS vs CosyVoice2 activation characteristics
  - **Deliverable:** A Jupyter notebook with activation statistics and PCA visualizations

### 2.3 Experiment Design Questions

1. **Single layer or multi-layer?** Start with a single layer (e.g., Layer 4 for both models — roughly the middle of the target range). If results are promising, extend to all target layers and train per-layer SAEs.
2. **Should we capture during text-only or text+prompt?** For CosyVoice2, the LLM sees both text and optional voice prompt tokens. Capturing during text-only inference isolates phonetic processing. Capturing with prompt adds speaker conditioning signal — decide based on research goals.
3. **Sequence position filtering:** Should we capture all positions, or only positions aligned with phoneme boundaries? Start with all positions (simpler), then filter in Phase 3 if needed.

---

## 3. Phase 2 — SAE Training (Weeks 2–3)

### 3.1 Objective

Train a Top-K Sparse Autoencoder that decomposes the LLM's hidden states into a sparse combination of interpretable features.

### 3.2 Milestones

**Week 2 (overlapping with Phase 1):**

- **M2.1 — SAE Architecture**
  - Implement Top-K SAE in PyTorch (or adapt from SAELens / DictionaryLearning)
  - Architecture:
    ```
    Encoder:  z = TopK(W_enc @ (x - b_dec) + b_enc, k=K)
    Decoder:  x̂ = W_dec @ z + b_dec
    Loss:     L = ||x - x̂||²
    ```
  - No $L_1$ penalty needed — sparsity is enforced by TopK selection
  - **Decision Point:** Start with expansion factor $R=16\times$ on 3090/4090 for memory safety

- **M2.2 — Memory-Constrained Configuration**

  | Parameter | Qwen3-TTS (0.6B) | CosyVoice2 (0.5B) |
  | :--- | :--- | :--- |
  | $d_{model}$ | 1024 | 896 |
  | Expansion $R$ | 16× | 16× |
  | $d_{sae}$ | 16,384 | 14,336 |
  | $K$ (sparsity) | 32 | 32 |
  | SAE params | ~33.6M | ~25.7M |
  | SAE VRAM (FP16) | ~67 MB | ~51 MB |
  | Training batch | 4096–8192 vectors | 4096–8192 vectors |

  The SAE itself is small (~34M params). The bottleneck is loading activation batches into VRAM alongside the SAE. With 24 GB, this is comfortable.

- **M2.3 — Training Pipeline**
  - Implement a `ShuffledActivationBuffer` that:
    - Loads activation files from disk in random order
    - Shuffles within and across files to prevent sentence-level overfitting
    - Yields mini-batches of activation vectors
  - Use AdamW optimizer with cosine learning rate schedule
  - Mixed precision training (AMP) for speed

**Week 3:**

- **M2.4 — Training Run**
  - Train for ~5B activation vectors (multiple passes over the dataset)
  - Log to Weights & Biases: reconstruction loss, dead feature count, feature activation histograms
  - **Early Stopping Criteria:**
    - Reconstruction loss plateaus for 500+ steps
    - Dead feature percentage < 10%
    - Explained variance > 90%
  - **Checkpoint:** Save model every 500M vectors

- **M2.5 — Dead Feature Mitigation**
  - Monitor dead features (features that never activate across 1M+ vectors)
  - If dead features > 15%, apply resampling: reinitialize dead feature weights from high-loss activation examples
  - Re-run training for another 1B vectors after resampling
  - **Decision Point:** If dead features remain high, consider reducing $R$ to 8× or increasing $K$ to 64

- **M2.6 — Hierarchical SAE Exploration (Optional)**
  - Inspired by [MSAE (Matryoshka SAE)](https://github.com/WolodjaZ/MSAE), train a hierarchical SAE that learns features at multiple granularities simultaneously
  - This could capture both low-level phonetic features (plosive bursts) and higher-level patterns (syllable structure) in a single model
  - **Decision Point:** Is the added complexity worth it? Compare standard Top-K vs. hierarchical on the same activation set

### 3.3 Experiment Design Questions

1. **Per-layer or shared SAE?** Start with per-layer SAEs (one per target layer). If cross-layer patterns emerge, consider a shared SAE with layer conditioning.
2. **Expansion factor sensitivity:** Train $R \in \{8, 16, 32\}$ on a 10% activation subset and compare reconstruction quality vs. feature interpretability.
3. **K sensitivity:** Test $K \in \{16, 32, 64\}$ — lower K means sparser (more interpretable) but potentially worse reconstruction.

---

## 4. Phase 3 — Phonetic Feature Mapping (Week 4)

### 4.1 Objective

Map the learned SAE features to known phonetic concepts. This is the interpretability payoff — turning opaque features into "The /s/ Feature" or "The Nasal Feature."

### 4.2 Milestones

- **M3.1 — Ground Truth Alignment**
  - Obtain time-aligned phoneme labels:
    - **LibriTTS:** Use Montreal Forced Aligner (MFA) to generate phone-level alignments
    - **TIMIT:** Already includes hand-corrected phoneme boundaries
  - Map each activation vector to its corresponding phoneme(s) using the time alignment
  - Handle boundary cases: vectors at phoneme transitions get assigned to both

- **M3.2 — Feature-Phoneme Correlation**
  - For each SAE feature $f_i$ and each phoneme $\phi$:
    $$P(f_i | \phi) = \frac{\text{count}(f_i \text{ active during } \phi)}{\text{count}(f_i \text{ active})}$$
    $$P(\phi | f_i) = \frac{\text{count}(f_i \text{ active during } \phi)}{\text{count}(\phi)}$$
  - Compute mutual information $I(f_i; \phi)$ for a more robust measure
  - **Deliverable:** A $|\text{features}| \times |\text{phonemes}|$ heatmap showing feature selectivity

- **M3.3 — Feature Taxonomy**
  - Categorize discovered features:
    - **Monosemantic phonetic:** High correlation with a single phoneme (e.g., "The /s/ feature")
    - **Phonetic class:** Correlates with a natural class (e.g., all plosives, all nasals, all fricatives)
    - **Positional:** Fires at word boundaries, syllable onsets, or codas
    - **Prosodic:** Correlates with stress, pitch patterns, or duration
    - **Polysemantic:** Activates for unrelated phonemes — these are failures or genuinely polysemantic
  - **Target:** At least 50 clearly monosemantic features per model

- **M3.4 — Visualization Dashboard**
  - Build an interactive tool (Streamlit or Plotly Dash) showing:
    - Feature activation overlaid on the spectrogram/waveform
    - Feature-phoneme correlation matrix (filterable)
    - Top-10 activating examples for any selected feature
    - PCA/t-SNE of feature space colored by phoneme class
  - **Decision Point:** Which features are "trustworthy" enough for causal intervention?

### 4.3 Experiment Design Questions

1. **Cross-model comparison:** Do Qwen3-TTS and CosyVoice2 learn similar phonetic features? Measure feature similarity using CKA (Centered Kernel Alignment) between the two SAE feature spaces.
2. **Layer-wise progression:** How does phonetic specificity change across layers? Early layers may encode raw phoneme identity; later target layers may encode coarticulation or allophonic variation.
3. **Language specificity:** If running on multilingual data, do features specialize by language or generalize across languages?

---

## 5. Phase 4 — Causal Intervention & Steering (Week 5)

### 5.1 Objective

Prove that SAE features causally control pronunciation by patching activations during the forward pass.

### 5.2 Milestones

- **M4.1 — Mispronunciation Catalog**
  - Run both models on 500+ challenging sentences (heteronyms, foreign words, rare phoneme sequences)
  - Use automatic speech recognition (e.g., Whisper) to detect mispronunciations
  - Build a catalog of 20+ reproducible mispronunciation examples per model
  - Categorize errors: phoneme substitution, deletion, insertion, stress misplacement

- **M4.2 — Patching Framework**
  - Implement `ActivationPatcher` that:
    1. Hooks into the target layer during forward pass
    2. Encodes the activation: $z = \text{SAE.encode}(x)$
    3. Modifies specific features: zero out, amplify, or replace
    4. Decodes back: $\hat{x} = \text{SAE.decode}(z_{modified})$
    5. Replaces the original activation with $\hat{x}$
  - Support batch interventions: multiple features modified simultaneously

- **M4.3 — Single-Feature Interventions**
  - For each mispronunciation, identify the candidate SAE features (from Phase 3 mapping)
  - Test three intervention types:
    - **Ablation:** Zero out the incorrectly-active feature → does the error disappear?
    - **Amplification:** Boost the correct feature by 2×, 5×, 10× → does pronunciation improve?
    - **Swap:** Zero incorrect + boost correct simultaneously
  - Evaluate using:
    - Whisper transcription accuracy (automated)
    - Mel spectrogram difference (quantitative)
    - Human listening test (10 evaluators, A/B comparison)
  - **Success Metric:** ≥70% of targeted mispronunciations corrected by single-feature intervention

- **M4.4 — Collateral Damage Assessment**
  - When patching Feature X, measure:
    - Does the rest of the utterance change? (Mel spectrogram distance for non-target regions)
    - Does speaker identity shift? (Speaker embedding cosine similarity before/after)
    - Does prosody break? (F0 contour correlation before/after)
  - **Threshold:** Collateral change < 5% on all non-target metrics

- **M4.5 — Multi-Feature Steering**
  - Combine multiple feature interventions to achieve larger phonetic shifts:
    - Accent transfer (e.g., American → British "schedule" /ˈskɛdʒuːl/ → /ˈʃɛdjuːl/)
    - Emphasis control (boost stress-related features on target syllables)
    - Speaking rate (if temporal features are discovered)
  - **Decision Point:** Is multi-feature steering stable, or do interactions cause artifacts?

### 5.3 Experiment Design Questions

1. **Layer choice for intervention:** Which layer gives the best patching results? Earlier layers may be more effective for phoneme-level changes; later layers for prosodic shifts.
2. **Intervention strength:** How much amplification is too much? Sweep from 1.5× to 20× and measure quality degradation.
3. **Generalization:** Does a feature intervention learned on one sentence generalize to other sentences with the same phoneme? (It should, if the feature is truly monosemantic.)

---

## 6. Phase 5 — Cross-Architecture Distillation (Week 6)

### 6.1 Objective

Use the discovered phonetic features to transfer pronunciation quality from a Teacher (Qwen3-TTS-1.7B) to a Student (CosyVoice2-0.5B or a custom lightweight model).

### 6.2 Milestones

- **M5.1 — Feature Bridge Design**
  - The Teacher and Student have different $d_{model}$ (1024 vs 896 or 2048 vs 896)
  - Train a linear projection $W_{bridge}: \mathbb{R}^{d_{student}} \to \mathbb{R}^{d_{teacher}}$
  - Alternatively: project both into the Teacher's SAE space directly
    - $z_{teacher} = \text{SAE}_{teacher}.\text{encode}(h_{teacher})$
    - $z_{student} = \text{SAE}_{teacher}.\text{encode}(W_{bridge} \cdot h_{student})$

- **M5.2 — Consistency Loss**
  - Add a feature-consistency term to the Student's training objective:
    $$L_{total} = L_{TTS} + \lambda \cdot L_{consistency}$$
    $$L_{consistency} = ||z_{teacher} - z_{student}||^2$$
  - $\lambda$ controls the strength of phonetic imitation — sweep $\{0.01, 0.1, 1.0\}$
  - Only apply on the monosemantic phonetic features (not all 16K+ features)

- **M5.3 — Distillation Training**
  - Freeze the Teacher; train only the Student + bridge
  - Dataset: Same LibriTTS-R subset used for activation mining
  - Training on 3090/4090: The Student (0.5B) + bridge fits easily; the Teacher (1.7B) runs inference-only in FP16
  - **Constraint:** 24 GB VRAM limits simultaneous loading — use gradient checkpointing or sequential forward passes

- **M5.4 — Evaluation**
  - Compare Student (before distillation) vs Student (after distillation) vs Teacher:
    - MOS (Mean Opinion Score) — human evaluation
    - Character Error Rate (CER) via Whisper transcription
    - Phoneme accuracy on the mispronunciation catalog
    - Speaker similarity (if applicable)
  - **Target:** Distilled Student achieves ≥95% of Teacher's phoneme accuracy at 50% of parameters

### 6.3 Experiment Design Questions

1. **Which features to distill?** Only monosemantic phonetic features, or all features? Starting with phonetic-only reduces noise.
2. **Bridge architecture:** Linear projection may be too weak. Consider a 2-layer MLP with ReLU if linear underperforms.
3. **Cross-model feasibility:** Can a CosyVoice2 Student learn from a Qwen3-TTS Teacher despite different tokenizers and generation flows? The SAE provides a shared representation space, but the downstream decoders are very different.

---

## 7. Hardware Budget & Memory Planning

All estimates assume a single NVIDIA RTX 3090 or 4090 (24 GB VRAM).

### 7.1 Phase 1 — Activation Mining

| Operation | VRAM Usage | Notes |
| :--- | :--- | :--- |
| Qwen3-TTS-0.6B inference | ~2 GB (FP16) | Talker only, no vocoder |
| CosyVoice2-0.5B inference | ~1.5 GB (FP16) | LLM backbone only |
| Activation buffer (batch of 512) | ~1 GB | 512 × seq_len × 1024 × 2 bytes |
| **Total** | **~4 GB** | Comfortable headroom |

**Disk:** ~200 GB for both models' activations (FP16). Consider INT8 quantization (~100 GB) after validating reconstruction.

### 7.2 Phase 2 — SAE Training

| Operation | VRAM Usage | Notes |
| :--- | :--- | :--- |
| SAE model (16× expansion) | ~67 MB | Tiny relative to GPU |
| Activation batch (8192 vectors) | ~16 MB | 8192 × 1024 × 2 bytes |
| Optimizer states (AdamW) | ~200 MB | 2× model size for momentum |
| **Total** | **~300 MB** | Very comfortable — can go to 32× if desired |

**Note:** If expanding to $R=32\times$, VRAM for SAE doubles to ~134 MB. Still fine on 24 GB. The real constraint is disk I/O for loading activations — use NVMe SSD for training speed.

### 7.3 Phase 4 — Causal Intervention

| Operation | VRAM Usage | Notes |
| :--- | :--- | :--- |
| TTS model (full, for generation) | ~4 GB (FP16) | Need full model for audio output |
| SAE model | ~67 MB | Loaded alongside TTS |
| Patching overhead | ~50 MB | Intermediate activations |
| **Total** | **~4.5 GB** | Comfortable |

### 7.4 Phase 5 — Distillation

| Operation | VRAM Usage | Notes |
| :--- | :--- | :--- |
| Teacher (Qwen3-TTS-1.7B, inference) | ~4 GB (FP16) | Frozen, no gradients |
| Student (CosyVoice2-0.5B, training) | ~3 GB (FP16 + grads) | With gradient checkpointing |
| SAE + bridge | ~200 MB | Small overhead |
| Optimizer states | ~1.5 GB | AdamW for Student + bridge |
| **Total** | **~9 GB** | Fits in 24 GB with room for batch size |

**Bottleneck:** Batch size during distillation. With ~15 GB free, you can afford batch sizes of 8–16 utterances depending on sequence length. Use gradient accumulation to simulate larger effective batches.

---

## 8. Risk Register & Decision Log

### 8.1 Risks

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| Dead features > 20% | Medium | High — wasted capacity | Resampling, reduce R, increase K |
| No monosemantic phonetic features | Low | Critical — project fails | Try residual stream instead of MLP; try different layers |
| Activation storage exceeds disk | Medium | Blocks Phase 2 | Use INT8 quantization; reduce dataset size to 20K sentences |
| Patching causes audio artifacts | High | Blocks Phase 4 | Use smaller intervention strengths; patch later layers |
| Teacher-Student domain mismatch | High | Blocks Phase 5 | Train SAE on combined activations; use domain adaptation |

### 8.2 Decision Log

| Date | Decision | Rationale | Status |
| :--- | :--- | :--- | :--- |
| 2026-03-30 | Target Qwen3-TTS + CosyVoice2 | Both use Qwen-family backbones; shared codebase possible | Confirmed |
| 2026-03-30 | Start with 16× expansion | 32× is safe on VRAM but 16× trains faster for prototyping | Pending validation |
| 2026-03-30 | Start with single layer (Layer 4) | Faster iteration; extend to multi-layer after proof of concept | Pending |
| 2026-03-30 | Use LibriTTS-R (clean-360) | High phonetic diversity, well-studied, MFA alignments available | Confirmed |

---

## 9. Project Directory Structure

```
phonetic-sae/
├── configs/                           # Training & experiment YAML configs
│   ├── sae_qwen3tts.yaml             # Qwen3-TTS SAE hyperparameters
│   └── sae_cosyvoice2.yaml           # CosyVoice2 SAE hyperparameters
├── src/
│   ├── __init__.py
│   ├── hooks/                         # Activation capture
│   │   ├── __init__.py
│   │   └── activation_hook.py         # Generic ActivationHook (register_forward_hook)
│   ├── models/                        # TTS model wrappers
│   │   ├── __init__.py
│   │   ├── qwen3_tts_wrapper.py       # Qwen3-TTS-0.6B: hooks on talker.layers[1..7].mlp
│   │   └── cosyvoice2_wrapper.py      # CosyVoice2-0.5B: hooks on llm.layers[1..6].mlp
│   ├── data/                          # Data loading & activation buffering
│   │   ├── __init__.py
│   │   ├── activation_buffer.py       # Streaming capture buffer → FP16 .npy on disk
│   │   ├── shuffled_activation_buffer.py  # Shuffled loader for SAE training
│   │   └── dataset_prep.py            # LibriTTS-R / TIMIT downloader & sentence iterator
│   ├── sae/                           # SAE architecture & training
│   │   ├── __init__.py
│   │   ├── topk_sae.py               # Top-K SAE (encoder/decoder, TopK activation)
│   │   └── trainer.py                 # Training loop: AdamW, cosine LR, AMP, W&B logging
│   ├── analysis/                      # Feature mapping & taxonomy (Phase 3)
│   │   ├── __init__.py
│   │   ├── phoneme_alignment.py       # MFA alignment runner
│   │   ├── feature_phoneme_correlation.py  # P(feature|phoneme), mutual information
│   │   └── feature_taxonomy.py        # Classify features: monosemantic, class, positional
│   ├── intervention/                  # Causal patching (Phase 4)
│   │   ├── __init__.py
│   │   └── activation_patcher.py      # Encode → modify features → decode → replace
│   ├── evaluation/                    # Metrics & evaluation (Phase 4)
│   │   ├── __init__.py
│   │   ├── mispronunciation_catalog.py  # Whisper-based mispronunciation detection
│   │   └── collateral_assessment.py   # Mel distance, speaker similarity, F0 correlation
│   ├── distillation/                  # Cross-model distillation (Phase 5)
│   │   ├── __init__.py
│   │   ├── feature_bridge.py          # Linear projection W_bridge: d_student → d_teacher
│   │   ├── consistency_loss.py        # ||z_teacher - z_student||² on monosemantic features
│   │   └── trainer.py                 # Distillation training loop
│   └── visualization/                 # Dashboards (Phase 3)
│       ├── __init__.py
│       └── dashboard.py               # Streamlit: heatmaps, spectrograms, PCA
├── scripts/                           # CLI entry points
│   ├── pilot_capture.py               # 100-sentence pilot through each model
│   ├── full_capture.py                # 50K-sentence full activation mining
│   ├── train_sae.py                   # SAE training entry point
│   ├── run_alignment.py               # MFA phoneme alignment
│   ├── compute_correlations.py        # Feature-phoneme correlation analysis
│   ├── run_interventions.py           # Causal patching experiments
│   └── run_distillation.py            # Cross-model distillation
├── tools/                             # MSAE adapters & utilities
│   └── tts_precompute_activations.py  # Convert captured .npy → MSAE SAEDataset format
├── notebooks/                         # Analysis notebooks
│   └── activation_audit.ipynb         # Activation statistics, PCA, per-layer distributions
├── tests/                             # Unit & integration tests
│   ├── test_activation_hook.py
│   ├── test_topk_sae.py
│   └── test_activation_buffer.py
├── data/                              # Runtime data (gitignored)
│   ├── activations/                   # Captured activation .npy files
│   ├── alignments/                    # MFA phoneme alignment outputs
│   └── datasets/                      # Downloaded LibriTTS-R / TIMIT
├── checkpoints/                       # SAE model checkpoints (gitignored)
├── third_party/                       # Git submodules (existing)
│   ├── CosyVoice2/
│   ├── Qwen3-TTS/
│   └── MSAE/
├── pretrained_models/                 # TTS model weights (existing, gitignored)
│   ├── Qwen3-TTS-0.6B/
│   └── CosyVoice2-0.5B/
├── docs/                              # Architecture docs (existing)
├── CLAUDE.md
├── PLAN.md
├── requirements.txt
├── requirements-cpu.txt
└── pyproject.toml                     # Package definition
```

---

## 10. Weekly Milestones

### Week 0: Scaffolding (Pre-requisite)
- [ ] Create `src/` package with all subpackages and `__init__.py` files
- [ ] Create `configs/`, `scripts/`, `tools/`, `notebooks/`, `tests/` directories
- [ ] Add `pyproject.toml` with package metadata and dependencies
- [ ] Update `.gitignore` for `data/`, `checkpoints/`, `outputs/`, `wandb/`
- [ ] Implement `src/hooks/activation_hook.py` — generic hook class
- [ ] Implement `src/sae/topk_sae.py` — Top-K SAE architecture
- [ ] Implement `src/data/activation_buffer.py` — streaming capture buffer
- [ ] Add basic smoke tests in `tests/`
- **Gate:** `import src` works; SAE forward pass runs on random tensors

### Week 1: Foundation
- [ ] Environment: both models running inference on GPU
- [ ] Dataset: LibriTTS-R downloaded, TIMIT dev set prepared
- [ ] Hooks: `ActivationHook` class implemented and validated
- [ ] Pilot: 100-sentence activation capture with statistics notebook
- **Gate:** Activations look reasonable (no NaN, sensible distributions)

### Week 2: Scale Up + SAE Prototype
- [ ] Full capture: 50K sentences through both models
- [ ] Activation quality audit notebook complete
- [ ] SAE architecture implemented (Top-K, 16× expansion)
- [ ] Training pipeline: ShuffledActivationBuffer + training loop
- [ ] Pilot SAE training on 1M vectors — verify loss decreases
- **Gate:** SAE reconstructs activations with < 0.1 MSE on held-out set

### Week 3: SAE Training at Scale
- [ ] Full training run: ~5B activation vectors
- [ ] Dead feature monitoring + resampling if needed
- [ ] W&B dashboard with training curves
- [ ] Model checkpointed and evaluated
- **Gate:** Dead features < 10%, explained variance > 90%

### Week 4: Feature Discovery
- [ ] MFA alignments generated for LibriTTS-R
- [ ] Feature-phoneme correlation matrix computed
- [ ] Feature taxonomy: 50+ monosemantic features identified
- [ ] Visualization dashboard (Streamlit) deployed locally
- **Gate:** Clear phoneme-feature correspondences visible in heatmap

### Week 5: Causal Proof
- [ ] Mispronunciation catalog: 20+ examples per model
- [ ] Patching framework implemented
- [ ] Single-feature interventions tested on 10+ examples
- [ ] Collateral damage assessment complete
- **Gate:** ≥70% of targeted mispronunciations corrected

### Week 6: Distillation & Wrap-Up
- [ ] Feature bridge + consistency loss implemented
- [ ] Distillation training run complete
- [ ] Before/after evaluation (CER, MOS, phoneme accuracy)
- [ ] Final report with all results, visualizations, and code
- **Gate:** Distilled Student achieves ≥95% of Teacher phoneme accuracy

---

## 11. References

### Core Papers & Repositories

1. **MSAE (Matryoshka SAE)** — Hierarchical sparse autoencoders, multi-scale feature learning (ICML 2025)
   - [WolodjaZ/MSAE](https://github.com/WolodjaZ/MSAE)
   - Relevant for hierarchical phonetic feature discovery (Phase 2 optional extension)

2. **Qwen3-TTS** — Open-source multilingual TTS with voice design and cloning
   - [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
   - [Technical Report (arXiv 2601.15621)](https://arxiv.org/abs/2601.15621)
   - [Model Card (HuggingFace)](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)

3. **CosyVoice2** — Scalable streaming speech synthesis with LLMs
   - [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
   - [Paper (arXiv 2412.10117)](https://arxiv.org/html/2412.10117v2)
   - [Project Page](https://funaudiollm.github.io/cosyvoice2/)

4. **TransformerLens** — Mechanistic interpretability toolkit
   - [TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)

5. **SAELens** — Library for training and analyzing sparse autoencoders
   - Used as reference implementation for Top-K SAE training

### Datasets

6. **LibriTTS-R** — Improved multi-speaker English TTS corpus
   - ~585 hours, 2,456 speakers, 24 kHz
   - Use clean-360 subset (~360 hours) for activation mining

7. **TIMIT** — Phonetically-rich continuous speech corpus
   - 6,300 utterances, 630 speakers
   - Hand-corrected phoneme boundaries — gold standard for Phase 3

### Tools

8. **Montreal Forced Aligner (MFA)** — Phoneme alignment tool for LibriTTS
9. **Whisper** — ASR model for automated pronunciation evaluation
10. **Weights & Biases** — Experiment tracking for SAE training

### Background Reading

11. **Awesome Mechanistic Interpretability** — Curated resource collection
    - [apartresearch/mechanisticinterpretability](https://github.com/apartresearch/mechanisticinterpretability)


---

# Next Steps & Implementation Priorities


- Language distribution? (e.g., 16K English, 16K Mandarin, 18K Cantonese)
- Format? (CSV, parquet, HuggingFace dataset, directory structure?)
- Path/location where it's stored?
- Do you have pre-computed phoneme alignments, or should we use Montreal Forced Aligner?
- Are there speaker IDs or speaker embeddings?
```

### 2. Model Paths Verification

Confirm that these directories exist:

```bash
ls -la pretrained_models/Qwen3-TTS-0.6B/
ls -la pretrained_models/CosyVoice2-0.5B/
ls -la third_party/CosyVoice2/
ls -la third_party/MSAE/
```

If any are missing:
- [ ] CosyVoice2: `git submodule update --init third_party/CosyVoice2`
- [ ] MSAE: `git submodule update --init third_party/MSAE`
- [ ] Models: Download from HuggingFace or provide paths

### 3. Hardware / Environment

- [ ] GPU available? (RTX 3090/4090 or similar)
- [ ] CUDA version? (run `nvidia-smi`)
- [ ] Python version? (3.10+)
- [ ] Can run test: `python -c "import torch; print(torch.cuda.is_available())"`

---

## What's Ready to Test Right Now

### Test 1: Load Models

```bash
cd /path/to/phonetic-sae
python -c "from src.models.qwen3_tts_wrapper import Qwen3TTSWrapper; w = Qwen3TTSWrapper(device='cpu'); print('✅ Qwen3-TTS loaded')"
python -c "from src.models.cosyvoice2_wrapper import CosyVoice2Wrapper; w = CosyVoice2Wrapper(device='cpu'); print('✅ CosyVoice2 loaded')"
```

### Test 2: Run Pilot Capture

```bash
python scripts/pilot_capture.py --model qwen3tts --output data/pilot_activations --num-samples 10
```

This will:
- Load Qwen3-TTS
- Attach hooks to phonetic layers
- Run 10 sentences through
- Save `.npy` files with activation statistics

Expected output:
```
data/pilot_activations/
├── layer_01_batch_000000.npy
├── layer_02_batch_000000.npy
├── ...
├── layer_07_batch_000000.npy
└── statistics.txt
```

---

## Implementation Priorities (Choose One)

### Option A: Complete Phase 1 (Recommended)
**Next:** Full capture + SAE training baseline

**Time:** 2-3 days
**Deliverables:**
- [ ] Full capture script (`scripts/full_capture.py`)
- [ ] Shuffled activation buffer (`src/data/shuffled_activation_buffer.py`)
- [ ] Activation audit notebook (`notebooks/activation_audit.ipynb`)
- [ ] Test SAE training on 1M vectors

**Success Criteria:**
- Activations capture smoothly on 50K sentences
- SAE reconstruction MSE < 0.1
- Notebook shows reasonable activation distributions

### Option B: Start Phase 2 Immediately (If Activations Exist)
**Next:** SAE training infrastructure

**Time:** 2-3 days
**Requires:** Pre-captured activation `.npy` or `.pt` files
**Deliverables:**
- [ ] SAE trainer (`src/sae/trainer.py`)
- [ ] Training loop with W&B logging
- [ ] Checkpoint management

### Option C: Focus on Multilingual Features (If Dataset Ready)
**Next:** Dataset adapter + multilingual analysis

**Time:** 1-2 days
**Requires:** Multilingual dataset details
**Deliverables:**
- [ ] Update `src/data/dataset_prep.py` for your dataset format
- [ ] Language-aware activation capture
- [ ] Per-language statistics in audit notebook

---

## Recommended Workflow

### Week 1: Activation Mining
1. **Day 1-2:** Provide dataset details
2. **Day 2-3:** Run pilot capture (100 sentences)
3. **Day 3-4:** Run full capture (50K sentences) — automated
4. **Day 5:** Activation audit notebook — verify quality

### Week 2: SAE Training
1. **Day 1-2:** Implement SAE trainer
2. **Day 2-3:** Train on 1M vectors (prototype)
3. **Day 3-5:** Full training on 5B vectors
4. **Day 5:** Save checkpoints + diagnostics

### Week 3: Feature Discovery
1. **Day 1-2:** Phoneme alignment (MFA)
2. **Day 2-3:** Feature-phoneme correlation analysis
3. **Day 3-5:** Visualization dashboard + feature taxonomy

### Weeks 4-6: Intervention & Distillation
- Phase 4: Causal intervention experiments
- Phase 5: Cross-model distillation training

---

## Minimal Test (5 minutes)

If you just want to verify the codebase works:

```bash
cd /path/to/phonetic-sae

# 1. Install package
pip install -e .

# 2. Run tests
python -m pytest tests/test_activation_hook.py -v

# 3. Test imports
python -c "
from src.hooks import ActivationHook
from src.sae import TopKSAE
from src.data.activation_buffer import ActivationBuffer
print('✅ All imports successful')
"

# 4. Test core functionality (no GPU needed)
python -c "
import torch
from src.sae import TopKSAE
from src.sae.topk_sae import SAEConfig

config = SAEConfig(d_in=64, d_sae=256, k=16)
sae = TopKSAE(config)
x = torch.randn(32, 64)
loss, metrics = sae(x)
print(f'✅ SAE works: MSE loss = {loss.item():.6f}')
"
```

---

## Blockers & Solutions

| Issue | Solution |
|-------|----------|
| Models not found | Download from HuggingFace or provide local path |
| CUDA out of memory | Reduce batch_size, use FP16 (already default) |
| CosyVoice2 import fails | Ensure `git submodule update --init third_party/CosyVoice2` |
| Dataset format mismatch | Share CSV sample or directory structure, I'll adapt loader |
| Activation statistics look bad | Check hook point (MLP vs residual), verify layer indices |

---

## Questions for You

1. **Dataset:** What's the format and size of your Mandarin/Cantonese/English dataset?
2. **Priority:** Want to complete Phase 1 first, or jump to SAE training if activations exist?
3. **Models:** Are both Qwen3-TTS and CosyVoice2 available on your system?
4. **Hardware:** RTX 3090/4090 or different GPU? (affects batch sizes)
5. **Timeline:** Any deadline for proof-of-concept results?

---

## What Happens Next (After Your Input)

1. **You provide dataset details** → I update `src/data/dataset_prep.py`
2. **You confirm model paths** → I test `pilot_capture.py`
3. **Pilot runs successfully** → I implement full capture + SAE trainer
4. **Full capture completes** → We move to feature discovery and intervention

---

## Files to Review

- **IMPLEMENTATION_STATUS.md** — What was built and why
- **README_IMPLEMENTATION.md** — How to use each module
- **PLAN.md** — Full 6-week roadmap
- **CLAUDE.md** — Project goals and research context

---

## Ready to Go? 🚀

Once you provide the above information, we can:
1. ✅ Run pilot capture immediately
2. ✅ Scale to full 50K-sentence capture
3. ✅ Train SAE baseline within a week
4. ✅ Begin feature discovery and intervention experiments

**Estimated time to first results:** 1-2 weeks (with your dataset ready)
