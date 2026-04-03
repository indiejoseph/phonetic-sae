# START HERE: 5-Minute Quick Start

You have PhoneticSAE and want to see it work. Here's the minimum viable path.

---

## The 5-Minute Path

### Minute 1: Validate
```bash
cd phonetic-sae
python scripts/validate_environment.py
```
If this says `✅ All checks passed!`, you can continue. Otherwise, fix the errors shown.

### Minute 2-3: Run Setup
```bash
bash scripts/setup.sh
```
This will:
- ✅ Verify your environment
- ✅ Check the Qwen3-ForcedAligner API
- ✅ Generate synthetic test data

**This downloads ~5GB of models. Should complete in 3-5 minutes.**

### Minute 4-5: First Capture Run
```bash
python scripts/capture_with_alignment.py \
  --model qwen3tts \
  --dataset custom \
  --dataset-file data/test_dataset/dataset.jsonl \
  --lang en \
  --output data/pilot_activations \
  --num-samples 5 \
  --device cuda
```

This will capture activations from 5 test samples. Should complete in 1-2 minutes.

### Check It Worked
```bash
# Should see files like this:
ls data/pilot_activations/layer_01/

# Should show: phoneme_h.npy, phoneme_ah.npy, etc.
cat data/pilot_activations/phoneme_inventory.json
# Should show: {"h": 30, "ah": 25, ...}
```

✅ **Done! You have your first activation captures.**

---

## Next: Read the Docs

Now that you know it works:

1. **Full guide:** `docs/PHASE1_QUICKSTART.md` (30 min walkthrough with details)
2. **All tools:** `docs/TOOLS_AND_UTILITIES.md` (reference for all scripts)
3. **Scale up:** Follow Step 6 in quickstart to capture 50K samples

---

## If Something Fails

### Most Common Issues

**PyTorch version error?**
```bash
pip install torch --upgrade
python scripts/validate_environment.py  # Check again
```

**CUDA out of memory?**
```bash
python scripts/capture_with_alignment.py ... --batch-size 1 --device cpu
# Use CPU instead of GPU (slower but works)
```

**Can't download models?**
```bash
export HF_HOME=/path/to/larger/disk
bash scripts/setup.sh  # Try again
```

**Other issues?**
```bash
# Get detailed diagnostics
python scripts/inspect_aligner_api.py --device cpu
python scripts/inspect_aligner.py --lang en
# This will show exactly what's wrong
```

---

## What You Just Did

You:
1. ✅ Validated your Python/PyTorch/CUDA setup
2. ✅ Verified Qwen3-ForcedAligner works
3. ✅ Generated synthetic test data
4. ✅ Ran the main activation capture pipeline
5. ✅ Got per-phoneme activations as output

This is **Phase 1 of the PhoneticSAE project:**
- Record the LLM's internal states during speech synthesis
- Align these states to individual phonemes
- Store them organized by (layer, phoneme) for later analysis

---

## What's Next?

### To get more captures (1-3 hours):
See `docs/PHASE1_QUICKSTART.md` Step 6 — run on your full dataset.

### To understand what happened:
- `docs/PHASE1_QUICKSTART.md` — full step-by-step guide
- `docs/QWEN3_FORCEDALIGNER_INFERENCE.md` — how the model works
- `docs/PHONEME_ALIGNMENT.md` — how alignment works

### To move to Phase 2:
Train a Sparse Autoencoder (SAE) on the captured activations. See `README.md#training`.

---

## Command Reference

```bash
# Setup (run once)
bash scripts/setup.sh

# Validate (if having issues)
python scripts/validate_environment.py

# Generate more test data
python scripts/generate_test_dataset.py --num-samples 100

# Run main pipeline
python scripts/capture_with_alignment.py \
  --model qwen3tts \
  --dataset custom \
  --dataset-file data/your_dataset.jsonl \
  --lang en \
  --output data/activations \
  --num-samples 50000 \
  --device cuda

# Debug tools
python scripts/inspect_aligner_api.py --device cuda      # Check API
python scripts/inspect_aligner.py --lang en              # Check phonemes
python scripts/analyze_aligner_inference.py --profile    # Profile speed
```

---

## Full Documentation

Now that you have basic understanding, see full docs:

- 📖 [Phase 1 Quick Start](docs/PHASE1_QUICKSTART.md)
- 🔧 [Tools & Utilities Reference](docs/TOOLS_AND_UTILITIES.md)
- ⚙️ [Qwen3-ForcedAligner Internals](docs/QWEN3_FORCEDALIGNER_INFERENCE.md)
- 📚 [Phoneme Alignment Guide](docs/PHONEME_ALIGNMENT.md)
- 🎯 [Project Overview](docs/PROJECT_OVERVIEW.md)

---

Happy phoneme aligning! 🎯
