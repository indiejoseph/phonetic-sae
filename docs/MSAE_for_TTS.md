# Using Matryoshka SAE (MSAE) with LLM-based TTS activations

This document summarizes the MSAE implementation (third_party/MSAE) and gives concrete guidance to apply it to LLM-based TTS models (e.g., Qwen3-TTS, CosyVoice2). It covers core implementation details, required data preprocessing, training/evaluation commands, and integration points for causal interventions.

## 1. What MSAE provides (high level)
- Implements Sparse Autoencoders (SAE) and Matryoshka SAE (hierarchical Top-K variants).
- Key features: TopK / BatchTopK activations, JumpReLU activation, SoftCapping of latent magnitudes.
- MatryoshkaAutoencoder exposes multiple nested sparsity levels (different k values) so the same model can be used at multiple budgets.
- Training tools: `train.py` (training loop), `precompute_activations.py` (dataset prep), evaluation scripts (`extract_sae_embeddings.py`, `sae_naming.py`, `score_topk_sae_embeddings.py`).

## 2. Core implementation details (files)
- `sae.py` — Autoencoder and MatryoshkaAutoencoder implementations. Important points:
  - `Autoencoder` has `encoder`, `decoder`, `pre_bias`, `latent_bias`, and `activation` modules.
  - `encode(x)` returns sparse (TopK enforced) and full activations for analysis; `decode(z)` reconstructs inputs.
  - Weight normalization: `scale_to_unit_norm()` keeps decoder rows unit-norm and rescales encoder/biases accordingly (helps interpretability).
  - Gradient projection: `project_grads_decode()` projects decoder gradients to preserve norms during update.
  - `MatryoshkaAutoencoder` constructs multiple TopK activations (e.g., k in [16,32,64]) and returns reconstructions for each nesting level.

- `loss.py` — `SAELoss` combines reconstruction loss (`mse`, `nmse`, `cosine`) and sparsity regularizer (`l1`, `l0`).

- `train.py` — training pipeline. Notes:
  - Inputs are precomputed embedding files loaded through `SAEDataset` (see `utils.py`).
  - Expansion factor is specified as `--expansion_factor` (latent dim = expansion * input_dim).
  - Matryoshka usage: provide `--model` set to `MSAE_UW` or `MSAE_RW` and set `--activation` (e.g., `TopKReLU_64`) and `--expansion_factor`.
  - Tracks metrics: explained variance (FVU), CKNNA, cosine similarity, dead neuron counts, orthogonality.

- `precompute_activations.py` — used in original repo to compute CLIP activations from datasets. For TTS you will replace or adapt the data source to LLM activations captured from the Talker/LLM layers.

## 3. Why MSAE fits LLM-based TTS activations
- MSAE is model-agnostic: it operates on fixed-size vector embeddings. LLM internal activations (residual stream or MLP post-activation) are suitable inputs.
- Advantages for TTS:
  - Hierarchical sparsity: Matryoshka lets you inspect features at multiple sparsity budgets (useful for deciding how many SAE features to use for intervention).
  - TopK activations produce sharp, monosemantic features (good for phoneme specificity).
  - Unit-norm decoder features + gradient projection increase interpretability and stability.

## 4. Practical integration steps (overview)
1. Capture activations from LLM Talker/LLM layers during inference (pilot 100 sentences):
   - Hook points: MLP post-activation or residual stream. Save per-token activations as float32/float16 NumPy arrays with shape `(N_examples, d_model)` or `(N_frames_total, d_model)` depending on tokenization.
   - Use a pipeline similar to `precompute_activations.py` but reading your saved activations instead of computing CLIP features.

2. Prepare datasets for MSAE training:
   - Format: NumPy `.npy` arrays where rows are vectors. The MSAE `SAEDataset` expects `.npy` files of embeddings (see `precompute_activations.py` for example usage).
   - Consider mean-centering or not — MSAE has `mean_center` config and `bias_init_median` options.

3. Configure model hyperparameters:
   - Expansion factor (R): 16× is a conservative start; 32× used in paper for sharper features. Latent dim = R * d_model.
   - Activation: use `TopKReLU_k` (e.g., `TopKReLU_32`) to enforce exactly K active features per sample.
   - Matryoshka nesting_list: choose levels like `[16,32,64]` or `[32,64]` depending on resources.

4. Train MSAE:
   - Use `train.py -dt <train.npy> -ds <val.npy> -m MSAE_UW -a TopKReLU_32 --expansion_factor 16 --epochs 30`
   - Monitor metrics logged by the script: reconstruction loss, FVU (explained variance), sparsity, dead neurons.

5. Evaluate and map features to phonetic concepts:
   - Run `extract_sae_embeddings.py` to save SAE activations for held-out data.
   - Use `sae_naming.py` or compute `P(feature | phoneme)` by aligning activation timestamps with phoneme boundaries (MFA alignments).

6. Use SAE for causal interventions:
   - During LLM forward pass, encode the LLM activation `x` with `sae.encode(x)` → `z`.
   - Modify `z` (zero/boost relevant feature indices), decode `x_hat = sae.decode(z)` and write `x_hat` back into the model's residual/MLP input.
   - The `encode`/`decode` API in `sae.py` is exactly suited for this workflow.

## 5. Implementation tips & gotchas
- Activation selection: capturing either residual stream or MLP post-activation will yield different signal; test both in a small pilot and compare explained variance when reconstructing.
- Normalization: MSAE supports `normalize=True` and mean-centering. For LLM activations try both with a small subset — keep training config consistent across runs.
- Bias init: `bias_init_median` can be used to initialize `pre_bias` from dataset median (robust). `train.py` supports a geometric median computation to set bias.
- Dead neurons: Monitor with training callbacks. MSAE includes utilities to reinitialize dead features (see code comments) or reduce expansion.
- Data scale: MSAE expects many vectors (millions) to learn stable features; start with a 1M-vector pilot before scaling to 50M.
- Compute: training is efficient, but precomputing activations and I/O are the bottleneck — use NVMe and streaming buffers.

## 6. Example commands (TTS-adapted)

# Precompute activations (example placeholder — adapt to your capture script)
```bash
# Suppose you saved TTS activations to `data/tts_train_activations.npy` and `data/tts_val_activations.npy`
python train.py -dt data/tts_train_activations.npy -ds data/tts_val_activations.npy -m MSAE_UW -a TopKReLU_32 --expansion_factor 16 --epochs 30
```

# Extract SAE embeddings for analysis
```bash
python extract_sae_embeddings.py --model path/to/msae.pt --data data/tts_val_activations.npy --out data/tts_val_sae.npy
```

# Score and name features (requires vocab files; you can build phoneme vocab or reuse provided vocabs)
```bash
python sae_naming.py --sae-emb data/tts_val_sae.npy --vocab vocab/your_phoneme_vocab.txt --out results/phoneme_feature_similarity.npy
```

## 7. Recommended hyperparameters to start
- Expansion `R`: 16
- K (TopK): 32
- Batch size: tune to GPU memory; MSAE training is lightweight compared to full models (the heavy part is data I/O)
- Learning rate: use `train.py` defaults from `config.py` and adjust if training is unstable

## 8. Next steps for integration into this repo
1. Add a small adaptor script `tools/tts_precompute_activations.py` that reads saved activation tensors from the TTS model and writes `.npy` training/val files compatible with `SAEDataset`.
2. Create an example `configs/msae_tts.yaml` (or pass CLI args) with recommended TTS settings (R=16, TopKReLU_32).
3. Implement a runtime `ActivationPatcher` (in `src/intervention.py`) that uses `sae.encode()` / `sae.decode()` to patch activations in the TTS forward hook (see Phase 4 in PLAN.md).

## 9. References in submodule
- See `README.md` in `third_party/MSAE` for original paper references, dataset usage, and example notebooks (demo.ipynb).

---
If you want, I can:
- create `tools/tts_precompute_activations.py` scaffold that adapts saved activations to MSAE `SAEDataset` format, and
- add an example `docs/MSAE_TTS_quickstart.md` with concrete commands tailored to Qwen3-TTS and CosyVoice2 hook points.
Which of those would you like me to implement next?
