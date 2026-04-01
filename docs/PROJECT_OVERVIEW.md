# Phonetic SAE: Mechanistic Interpretability for LLM-Based TTS

**Project Goal:** Untangle polysemantic neurons in the phonetic layer of LLM-based TTS systems using Sparse Autoencoders (SAEs), enabling interpretability and controllable pronunciation steering.

**Status:** Planning & Implementation Phase
**Last Updated:** 2026-03-30

---

## Project Overview

This project applies mechanistic interpretability techniques to understand and control how large language models process phonetic information during text-to-speech synthesis. By training a Sparse Autoencoder (SAE) on the hidden activations of the LLM's early layers, we can discover monosemantic phonetic features that correspond to specific phonemes, prosodic patterns, and articulation features.

### Key Innovation
Unlike monolithic neural network layers, the SAE decomposes the LLM's phonetic representations into a sparse dictionary of interpretable features, enabling:
- **Feature Discovery:** Identify which SAE features encode specific phonemes or prosodic patterns
- **Causal Intervention:** Steer pronunciation by amplifying/dampening specific SAE features
- **Cross-Model Distillation:** Transfer the Teacher's superior articulation to lightweight Student models

---

## 5-Phase Implementation Plan

### Phase 1: Activation Mining (Week 1)

**Objective:** Record the LLM's internal phonetic processing across a large diverse dataset.

**1.1 Layer Selection**
- **Target:** First 25% of Transformer blocks (e.g., Layers 1–6 in a 24-layer model)
- **Hook Point:** MLP post-activation or Residual Stream ($x + \text{attn} + \text{mlp}$)
- **Rationale:** Grapheme-to-Phoneme mapping occurs early; prosody and speaker identity stabilize later

**1.2 Data Generation**
- **Dataset:** LibriTTS or TIMIT (high phonetic diversity)
- **Scale:** 50,000+ sentences during inference
- **Storage:** Quantized activation tensors (~50M to 100M vectors at $d_{model} \approx 1024$)

**Implementation Checklist:**
- [ ] Identify exact layer indices for your TTS architecture
- [ ] Implement `forward_hooks` for activation capture
- [ ] Set up efficient activation buffer with shuffling
- [ ] Validate captured activations (shape, dtype, value ranges)

---

### Phase 2: SAE Architecture & Training (Week 2)

**Objective:** Build a sparse "dictionary" representing the LLM's hidden states as sparse combinations of phonetic features.

**2.1 Model Specifications**
- **Type:** Top-K SAE (preferred over $L_1$ SAEs for better feature separation)
- **Expansion Factor:** $32 \times$ (If $d_{model} = 1024$, then $d_{sae} = 32,768$)
- **Sparsity ($K$):** $K=32$ (Only 32 features active simultaneously)
- **Loss Function:**
  $$L = ||x - \hat{x}||^2 \text{ (Reconstruction Error)}$$
  *(No $L_1$ term needed; sparsity enforced by architecture)*

**2.2 Training Configuration**
- **Hardware:** 1× A100 or H100 (80GB)
- **Framework:** SAELens or DictionaryLearning (PyTorch)
- **Duration:** ~10 billion tokens to convergence

**Monitoring Metrics:**
- [ ] Dead Feature Count (features that never activate)
- [ ] Explained Variance (what fraction of the LLM state the SAE captures)
- [ ] Reconstruction Error over training iterations

---

### Phase 3: Phonetic Feature Mapping (Week 3)

**Objective:** Identify which SAE features encode specific phonetic concepts.

**3.1 Automated Labeling**
1. **Alignment:** Use ground truth time-aligned phonemes from the dataset
2. **Correlation Analysis:**
   $$P(f_i | \text{Phoneme } \phi) = \frac{\text{Activations during } \phi}{\text{Total activations}}$$
3. **Discovery:** High-probability correlations identify monosemantic phonetic features

**3.2 Phonetic Envelope Visualization**
- Create a dashboard showing feature "firing patterns" over waveforms
- **Expected Result:** "Plosive Feature" fires at burst locations; "Fricative Feature" fires during sustained noise

**Deliverables:**
- [ ] Feature-to-phoneme correlation matrix
- [ ] Interactive visualization dashboard
- [ ] List of "interpretable" features (top 50-100)

---

### Phase 4: Causal Intervention & Steering (Week 4)

**Objective:** Prove that SAE features causally control pronunciation.

**4.1 The "Pronunciation Patch"**
1. Identify a mispronounced word (e.g., "Schedule" as /sk/ vs /sh/)
2. During forward pass:
   - Map LLM activation $x$ into SAE space: $z = \text{encode}(x)$
   - **Zero out** the "Hard-K" feature
   - **Boost** the "Fricative /sh/" feature
   - Reconstruct: $\hat{x} = \text{decode}(z)$
   - Pass $\hat{x}$ back into the next LLM layer
3. **Success:** Audio output changes to correct pronunciation

**4.2 Intervention Suite**
- [ ] Implement patching framework (activation replacement)
- [ ] Test on 10+ mispronounced examples
- [ ] Measure success rate and collateral effects
- [ ] Document feature interactions

---

### Phase 5: Cross-Architecture Distillation (Final Goal)

**Objective:** Unify Teacher (Large LLM) and Student (Small Model) via phonetic feature consistency.

| Step | Action |
| :--- | :--- |
| **Feature Bridge** | Map Student's hidden states into Teacher's SAE space |
| **Consistency Loss** | Force Student to activate same SAE features as Teacher for same text |
| **Training** | Finetune Student with combined reconstruction + consistency loss |
| **Benefit** | Student inherits Teacher's articulation without parameter overhead |

**Implementation Checklist:**
- [ ] Define Student architecture (e.g., 6-layer Transformer)
- [ ] Implement feature bridging layer
- [ ] Add consistency loss to training loop
- [ ] Benchmark Student vs Teacher pronunciation quality

---

## Research References

- **MSAE (Matryoshka SAEs):** Hierarchical sparse autoencoders for multi-scale feature discovery (ICML 2025)
  - Achieves 0.99 cosine similarity with ~80% sparsity
  - Relevant for multi-granularity phonetic features
  - [WolodjaZ/MSAE](https://github.com/WolodjaZ/MSAE)

- **TransformerLens:** De facto standard for mechanistic interpretability research
  - Provides activation hooks and feature visualization tools
  - [TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)

- **Awesome Mechanistic Interpretability:** Curated collection of resources
  - [apartresearch/mechanisticinterpretability](https://github.com/apartresearch/mechanisticinterpretability)

---

## FAQ

**Q: Why start with the first 25% of layers?**
A: Phonetic decoding (grapheme → phoneme mapping) happens early. Later layers refine prosody, speaker characteristics, and high-level phrasing.

**Q: Why Top-K over $L_1$ SAEs?**
A: Top-K provides sharper feature boundaries and avoids the hyperparameter tuning required by $L_1$ regularization.

**Q: What if my TTS model is different?**
A: The methodology is architecture-agnostic. The key is identifying where phonetic processing occurs—usually before prosody or speaker embedding.

**Q: Can I use smaller datasets?**
A: You can start with 10,000 sentences for prototyping, but 50,000+ is recommended for robust feature discovery.

---

## Next Steps

1. **Identify TTS Architecture:** Which LLM-based TTS are you using? (e.g., Vall-E, Fish-Speech, GPT-SoVITS)
2. **Layer Indexing:** Map the specific layer indices for activation hooks
3. **Data Pipeline:** Download and prepare LibriTTS / TIMIT, implement activation capture
4. **SAE Baseline:** Train a simple Top-K SAE on 1M activations to validate the pipeline

---

## Docs

- `docs/MSAE_for_TTS.md`: MSAE integration guide for LLM-based TTS — implementation notes, data prep, training commands, and intervention workflow.

