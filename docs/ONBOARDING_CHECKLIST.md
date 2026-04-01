# PhoneticSAE Onboarding Checklist

Complete these steps to get started with Phase 1 (Activation Mining).

## ✅ Before You Start

- [ ] You have the PhoneticSAE repository cloned
- [ ] You have Python 3.10+ installed
- [ ] You have a GPU with 8GB+ VRAM (or can use CPU, slower)
- [ ] You have ~500GB free disk space (or 50GB for testing)
- [ ] You have internet access (for downloading models)

---

## 🚀 Quick Start (5 minutes)

1. **Validate Environment** (1 min)
   ```bash
   python scripts/validate_environment.py
   ```
   - [ ] Shows ✅ All checks passed

2. **Run Setup** (2-3 min)
   ```bash
   bash scripts/setup.sh
   ```
   - [ ] Downloads models successfully
   - [ ] Generates synthetic test data
   - [ ] Shows "✅ Environment is ready for Phase 1"

3. **First Capture** (1-2 min)
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
   - [ ] Shows progress for 5 samples
   - [ ] Completes without errors

4. **Verify Output**
   ```bash
   ls data/pilot_activations/layer_01/
   cat data/pilot_activations/phoneme_inventory.json
   ```
   - [ ] See `phoneme_*.npy` files
   - [ ] See phoneme counts in JSON

✅ **Quick Start Complete! You have working activation captures.**

---

## 📖 Understanding (30 minutes)

- [ ] Read: `START_HERE.md` (5 min summary)
- [ ] Read: `docs/PHASE1_QUICKSTART.md` (detailed 7-step guide)
- [ ] Skim: `docs/TOOLS_AND_UTILITIES.md` (reference guide)

**Questions?** Check:
- `docs/PHONEME_ALIGNMENT.md` — How alignment works
- `docs/QWEN3_FORCEDALIGNER_INFERENCE.md` — Model internals
- `README.md#troubleshooting` — Common issues

---

## 📊 Scaling Up (2-3 hours)

1. **Prepare Your Dataset**
   - [ ] Have JSONL file with: `text`, `lang`, `audio_path`
   - [ ] Audio files are 16kHz WAV format
   - [ ] Dataset has 1000+ samples

2. **Run Full Capture**
   ```bash
   python scripts/capture_with_alignment.py \
     --model qwen3tts \
     --dataset custom \
     --dataset-file data/your_dataset.jsonl \
     --lang en \
     --output data/activations/qwen3tts \
     --num-samples 50000 \
     --device cuda \
     --batch-size 4
   ```
   - [ ] Monitor with: `du -sh data/activations/qwen3tts`
   - [ ] Should grow from 0 → 100GB over 2-3 hours
   - [ ] Completes without errors

3. **Validate Results**
   ```python
   import json, numpy as np
   from pathlib import Path
   
   activation_dir = Path("data/activations/qwen3tts")
   
   # Check phoneme coverage
   with open(activation_dir / "phoneme_inventory.json") as f:
       inv = json.load(f)
       print(f"Unique phonemes: {len(inv)}")
       print(f"Total samples: {sum(inv.values())}")
   
   # Check activation shapes
   for npy_file in activation_dir.glob("layer_*/phoneme_*.npy"):
       arr = np.load(npy_file)
       print(f"{npy_file.name}: {arr.shape}")
   ```
   - [ ] Have activations for 30+ phonemes
   - [ ] Each layer has 1000+ activation vectors
   - [ ] Activation shapes are (N, 1024) or similar

✅ **Phase 1 Complete! Ready for Phase 2 (SAE Training)**

---

## 🔧 If You Get Stuck

### Issue: "CUDA out of memory"
- [ ] Reduce `--batch-size 4` to `--batch-size 2` or `--batch-size 1`
- [ ] Use `--device cpu` (slower but works)
- [ ] See `docs/TOOLS_AND_UTILITIES.md#out-of-memory`

### Issue: "Module not found: src.alignment"
- [ ] Ensure git submodules are initialized: `git submodule update --init --recursive`
- [ ] Ensure you're in repo root: `cd /path/to/phonetic-sae`
- [ ] See `README.md#troubleshooting`

### Issue: "Model download timeout"
- [ ] Set HuggingFace cache: `export HF_HOME=/path/to/large/disk`
- [ ] Retry the setup: `bash scripts/setup.sh`
- [ ] See `docs/TOOLS_AND_UTILITIES.md#troubleshooting`

### Issue: "Phoneme alignment seems wrong"
- [ ] Verify phoneme inventory: `python scripts/inspect_aligner.py --lang en`
- [ ] Check audio quality (16kHz, mono, not corrupted)
- [ ] See `docs/PHONEME_ALIGNMENT.md#troubleshooting`

### Issue: Something else?
- [ ] Run diagnostics: `python scripts/validate_environment.py`
- [ ] Check API: `python scripts/inspect_aligner_api.py --device cpu`
- [ ] Read: `docs/ACTUAL_API_DISCOVERY.md`
- [ ] See: `docs/TOOLS_AND_UTILITIES.md#troubleshooting`

---

## 📚 Next Steps

After Phase 1 (Activation Mining) is complete:

### Phase 2: SAE Training
- Train a Sparse Autoencoder on captured activations
- See: `README.md#training`
- Time: ~12 hours on RTX 3090

### Phase 3: Feature Mapping
- Discover which SAE features represent phonemes
- See: `docs/PROJECT_PLAN.md#phase-3`

### Phase 4: Causal Intervention
- Test if you can steer pronunciation by manipulating SAE features
- See: `docs/PROJECT_PLAN.md#phase-4`

### Phase 5: Cross-Model Distillation
- Transfer Teacher model's phonetic knowledge to Student model
- See: `docs/PROJECT_PLAN.md#phase-5`

---

## 📖 Documentation Map

**Start Here:**
- `START_HERE.md` — 5-minute overview
- `docs/PHASE1_QUICKSTART.md` — 30-minute detailed guide

**References:**
- `docs/TOOLS_AND_UTILITIES.md` — All scripts documented
- `README.md` — Project overview

**Deep Dives:**
- `docs/PHONEME_ALIGNMENT.md` — How alignment works
- `docs/QWEN3_FORCEDALIGNER_INFERENCE.md` — Model internals
- `docs/ACTUAL_API_DISCOVERY.md` — API verification

**Project Planning:**
- `docs/PROJECT_PLAN.md` — 6-week roadmap
- `docs/PROJECT_OVERVIEW.md` — Research goals

---

## 🎉 Success Criteria

You've successfully completed Phase 1 when you have:

- ✅ Environment validated with `validate_environment.py`
- ✅ Synthetic test capture successful (5-10 samples)
- ✅ Full dataset capture complete (50K+ samples)
- ✅ Output directory structure:
  ```
  data/activations/qwen3tts/
  ├── layer_01/phoneme_*.npy    (40+ phoneme files)
  ├── layer_02/phoneme_*.npy
  ├── ... (7 layers total)
  ├── phoneme_inventory.json    (30+ unique phonemes)
  └── frame_labels.jsonl        (50K+ samples)
  ```
- ✅ Activation statistics validated (see scaling up section above)

---

## 💡 Tips

1. **Start small:** Test with 5 samples before committing to 50K
2. **Monitor progress:** Use `du -sh data/activations/` in another terminal
3. **Keep logs:** Redirect output: `bash scripts/setup.sh > setup.log 2>&1`
4. **Validate often:** Check intermediate outputs at each step
5. **Read errors:** Most issues have clear error messages with solutions

---

## ❓ FAQ

**Q: How long does Phase 1 take?**
- Setup: 5-10 min
- Pilot capture (10 samples): 5-10 min
- Full capture (50K samples): 2-3 hours on RTX 3090

**Q: Can I use CPU instead of GPU?**
- Yes, but much slower (~10x slower)
- Use: `--device cpu`

**Q: Do I need real speech data?**
- For testing: No, use synthetic data from `generate_test_dataset.py`
- For Phase 2+: Yes, need real data for meaningful SAE features

**Q: What if my language isn't en/zh/yue?**
- Currently only these languages supported
- Can add new languages (see `docs/PHONEME_SETS_SOURCES.md`)

**Q: How much disk space do I need?**
- Testing: ~5GB
- Full Phase 1: ~100GB (for 50K samples)
- Total (all phases): ~500GB recommended

---

## 🚀 Ready?

1. Ensure prerequisites above are checked
2. Run: `bash scripts/setup.sh`
3. Follow: `docs/PHASE1_QUICKSTART.md`
4. Check this list as you go

**Questions?** See documentation links above or check GitHub issues.

Happy activating! 🎯
