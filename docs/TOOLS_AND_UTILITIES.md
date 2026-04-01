# PhoneticSAE Tools and Utilities

Complete reference for all scripts and tools available in the PhoneticSAE project for Phase 1 (Activation Mining) and beyond.

---

## Quick Reference

| Tool | Purpose | Use Case |
|------|---------|----------|
| `setup.sh` | One-command environment setup | Getting started for the first time |
| `validate_environment.py` | Check all dependencies | Before running any Phase 1 scripts |
| `generate_test_dataset.py` | Create synthetic test data | Quick validation without real audio |
| `inspect_aligner_api.py` | Verify Qwen3-ForcedAligner works | Debug API issues |
| `inspect_aligner.py` | Extract phoneme inventories | Understand what phonemes the model knows |
| `analyze_aligner_inference.py` | Profile model performance | Estimate inference latency and memory |
| `capture_with_alignment.py` | Main activation capture pipeline | Collect per-phoneme activations |

---

## Setup & Validation

### `bash scripts/setup.sh`

**Purpose:** One-command setup and validation for the entire project.

**What it does:**
1. Validates environment (Python, PyTorch, CUDA, disk space)
2. Checks git submodules are initialized
3. Verifies Qwen3-ForcedAligner API
4. Extracts phoneme inventories for all languages
5. Generates synthetic test dataset

**When to use:**
- First time setup
- After major environment changes
- To verify everything is working

**Output:**
- Console output with status of each check
- `aligner_api_verification.txt` - API inspection results
- `phoneme_inventory_*.json` - Phoneme sets per language
- `data/test_dataset/` - Synthetic test data

**Time:** ~5-10 minutes (depending on model downloads)

---

### `python scripts/validate_environment.py`

**Purpose:** Comprehensive environment validation.

**What it checks:**
- ✅ Python version (3.10+)
- ✅ Repository structure (src/, scripts/, docs/)
- ✅ Git submodules initialized
- ✅ PyTorch installed (2.0+)
- ✅ CUDA availability
- ✅ Transformers library
- ✅ TorchAudio library
- ✅ NumPy library
- ✅ Core module imports (src.hooks, src.sae, src.data)
- ✅ Disk space (recommended 500GB)
- ✅ HuggingFace connectivity

**Options:**
```bash
python scripts/validate_environment.py        # Standard checks
python scripts/validate_environment.py --verbose  # Detailed output
```

**Output:**
```
✅ Python version: Python 3.11 (need 3.10+)
✅ Repository structure: All required directories found
✅ Git submodules: All 3 submodules initialized
✅ PyTorch installation: PyTorch 2.1.0
✅ CUDA availability: RTX 4090 (24GB)
... (more checks)

RESULTS: 10 passed, 0 failed
✅ All checks passed! Ready for Phase 1.
```

**Exit codes:**
- `0` = All checks passed
- `1` = Some critical checks failed

---

## API Inspection & Phoneme Discovery

### `python scripts/inspect_aligner_api.py`

**Purpose:** Discover and verify the actual Qwen3-ForcedAligner API.

**What it does:**
1. Loads Qwen3-ForcedAligner model
2. Lists all callable methods and attributes
3. Inspects processor capabilities
4. Inspects model configuration
5. Tests inference with different input patterns
6. Saves results to JSON

**Options:**
```bash
python scripts/inspect_aligner_api.py --device cuda   # Use GPU (default)
python scripts/inspect_aligner_api.py --device cpu    # Use CPU (slower)
```

**Output:**
- Console output listing methods and test results
- `aligner_api_inspection.json` with structured results

**Expected output (excerpt):**
```
MODEL METHODS & FUNCTIONS
Looking for key methods:
  ✅ align: (text, audio, language, sample_rate)
  ✅ forward: (...)
  ...

TESTING ACTUAL MODEL INPUT/OUTPUT
Testing inference...
  Testing: text + audio...
    ✅ Success!
    Output type: ModelOutput
    Output keys: ['align_boundaries', 'frame_to_phoneme', ...]
```

**Use this to:**
- Verify the model's actual API matches our implementation
- Debug API mismatches if capture fails
- Check processor capabilities

---

### `python scripts/inspect_aligner.py`

**Purpose:** Extract and verify phoneme inventories from the model.

**What it does:**
1. Loads Qwen3-ForcedAligner for specified language
2. Extracts actual phoneme set the model knows
3. Compares with default/expected phoneme sets
4. Validates phonemes against dataset
5. Saves inventory to JSON

**Options:**
```bash
python scripts/inspect_aligner.py --lang en    # English (ARPAbet)
python scripts/inspect_aligner.py --lang zh    # Mandarin (Pinyin)
python scripts/inspect_aligner.py --lang yue   # Cantonese (Jyutping)
python scripts/inspect_aligner.py --lang en zh yue  # All languages
```

**Output:**
```
Extracted phoneme inventory for language: en
Found 43 phonemes:
['aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', ...]

Comparing with default ARPAbet set...
✅ Model inventory matches default ARPAbet set

Saved to: phoneme_inventory_en.json
```

**Generated files:**
- `phoneme_inventory_{lang}.json` - Phoneme inventory for language
- Console output with validation results

**Use this to:**
- Understand what phonemes the model recognizes
- Verify dataset phonemes match model vocabulary
- Debug alignment issues related to missing phonemes

---

### `python scripts/analyze_aligner_inference.py`

**Purpose:** Analyze model architecture and profile inference performance.

**What it does:**
1. Analyzes model architecture (layers, parameters)
2. Profiles inference latency and throughput
3. Measures GPU memory usage
4. Lists major modules and components
5. Saves analysis to JSON

**Options:**
```bash
python scripts/analyze_aligner_inference.py --lang en              # Language
python scripts/analyze_aligner_inference.py --device cuda          # Device
python scripts/analyze_aligner_inference.py --profile              # Enable profiling (slower)
python scripts/analyze_aligner_inference.py --lang en --profile    # Both
```

**Output:**
```
========= QWEN3-FORCEDALIGNER INFERENCE ANALYSIS =========

Model Summary:
  Type: QwenForCausalLM
  Total params: 600,000,000
  Trainable params: 600,000,000
  Size on disk: 2286.0 MB

INFERENCE PROFILING (en)
Profiling 10 inference runs...
  Completed 3/10 runs
  ...
Inference Performance:
  Mean latency: 125.34 ms
  Median latency: 123.45 ms
  Std dev: 5.23 ms
  Throughput: 7.98 samples/sec
  Peak memory: 2100.5 MB
```

**Generated files:**
- `aligner_analysis_{lang}.json` - Detailed analysis results

**Use this to:**
- Estimate how long full dataset capture will take
- Verify GPU has enough memory
- Understand model performance characteristics

---

## Data Generation

### `python scripts/generate_test_dataset.py`

**Purpose:** Create synthetic test datasets for quick pipeline validation.

**What it does:**
1. Generates synthetic audio (white noise)
2. Creates sample texts in multiple languages
3. Writes dataset to JSONL format
4. No internet/model download needed

**Options:**
```bash
python scripts/generate_test_dataset.py \
  --output data/test_dataset \         # Output directory
  --num-samples 10 \                   # Samples per language (default: 10)
  --lang en zh yue \                   # Languages (default: en)
  --duration 1.0                       # Audio duration in seconds (default: 1.0)
```

**Output:**
```
======= GENERATING SYNTHETIC TEST DATASET =======

Output directory: data/test_dataset
Languages: ['en', 'zh', 'yue']
Samples per language: 10
Audio duration: 1.0s

en: Generated 5/10 samples
en: Generated 10/10 samples
zh: Generated 10/10 samples
yue: Generated 10/10 samples

✅ DATASET CREATED SUCCESSFULLY

Dataset file: data/test_dataset/dataset.jsonl
Total samples: 30

  en: 10 samples
  zh: 10 samples
  yue: 10 samples
```

**Generated files:**
- `dataset.jsonl` - JSONL file with text/audio pairs
- `audio/` - Synthetic audio files (.wav)

**Dataset format:**
```jsonl
{"text": "hello world", "lang": "en", "audio_path": "data/test_dataset/audio/en_0000.wav"}
{"text": "你好世界", "lang": "zh", "audio_path": "data/test_dataset/audio/zh_0001.wav"}
...
```

**Use this to:**
- Validate the full pipeline without real audio
- Test on limited data before committing to full dataset
- Debug issues in isolation

---

## Main Phase 1 Pipeline

### `python scripts/capture_with_alignment.py`

**Purpose:** Main activation capture pipeline with phoneme alignment.

**What it does:**
1. Loads TTS model (Qwen3-TTS or CosyVoice2)
2. Loads dataset (JSONL or CSV format)
3. Records hidden activations during inference
4. Uses Qwen3-ForcedAligner to align phonemes
5. Organizes activations by (layer, phoneme)
6. Saves to disk with metadata

**Options:**
```bash
python scripts/capture_with_alignment.py \
  --model qwen3tts \                  # Model (qwen3tts, cosyvoice2)
  --dataset custom \                  # Dataset type (custom, libritts)
  --dataset-csv data/dataset.jsonl \  # Dataset path
  --lang en \                         # Language (en, zh, yue)
  --output data/activations \         # Output directory
  --num-samples 50000 \               # Number of samples to process
  --device cuda \                     # Device (cuda, cpu)
  --batch-size 4 \                    # Batch size (adjust for GPU memory)
  --verbose                           # Verbose logging
```

**Output structure:**
```
data/activations/
├── layer_01/
│   ├── phoneme_h.npy          # (N, 1024) array
│   ├── phoneme_eh.npy
│   ├── phoneme_l.npy
│   └── ...
├── layer_02/
│   ├── phoneme_h.npy
│   └── ...
├── layer_03/
│   └── ...
├── phoneme_inventory.json     # {"h": 1500, "eh": 1200, ...}
└── frame_labels.jsonl         # Per-sample metadata
```

**Console output (excerpt):**
```
Loading Qwen3-TTS (0.6B)...
Initializing Qwen3-ForcedAligner (language: en)...
Processing 50000 samples...
  [1/50000] hello (156 frames) | Phonemes: 4
  [2/50000] world (142 frames) | Phonemes: 4
  ...
  [100/50000] (ETA: 1h 23m)
Organizing activations by phoneme...
✅ Capture complete! Output: data/activations/
```

**Key parameters:**
- `--num-samples`: Control how many samples to process (smaller for testing)
- `--batch-size`: Reduce if out of GPU memory
- `--verbose`: Enable detailed logging
- `--device`: Use cpu if cuda has issues

**Use this to:**
- Capture activations for full Phase 1 pipeline
- Process custom datasets
- Extract per-phoneme representations for SAE training

---

## Typical Workflows

### 1. First-Time Setup

```bash
# Complete one-command setup
bash scripts/setup.sh

# Or step-by-step
python scripts/validate_environment.py
python scripts/inspect_aligner_api.py --device cuda
python scripts/inspect_aligner.py --lang en
```

### 2. Pilot Capture (10 samples, 5 minutes)

```bash
# Generate test data
python scripts/generate_test_dataset.py --num-samples 10 --output data/test_dataset

# Run pilot capture
python scripts/capture_with_alignment.py \
  --model qwen3tts \
  --dataset custom \
  --dataset-csv data/test_dataset/dataset.jsonl \
  --lang en \
  --output data/pilot_activations \
  --num-samples 10 \
  --device cuda

# Inspect results
ls -la data/pilot_activations/
python -c "import json; print(json.load(open('data/pilot_activations/phoneme_inventory.json')))"
```

### 3. Full Capture (50K samples, 2-3 hours)

```bash
# Assuming you have prepared your dataset at data/your_dataset.jsonl
python scripts/capture_with_alignment.py \
  --model qwen3tts \
  --dataset custom \
  --dataset-csv data/your_dataset.jsonl \
  --lang en \
  --output data/activations/qwen3tts \
  --num-samples 50000 \
  --device cuda \
  --batch-size 4
```

### 4. Debug API Issues

```bash
# Step 1: Verify API works
python scripts/inspect_aligner_api.py --device cpu

# Step 2: Extract phonemes
python scripts/inspect_aligner.py --lang en

# Step 3: Profile performance
python scripts/analyze_aligner_inference.py --lang en --profile

# Step 4: Try small capture
python scripts/generate_test_dataset.py --num-samples 3
python scripts/capture_with_alignment.py \
  --model qwen3tts \
  --dataset custom \
  --dataset-csv data/test_dataset/dataset.jsonl \
  --lang en \
  --output data/debug_activations \
  --num-samples 3 \
  --device cpu \
  --verbose
```

---

## Environment Variables

### HuggingFace Model Cache

```bash
export HF_HOME=/path/to/large/disk/huggingface
python scripts/inspect_aligner_api.py --device cuda
```

### CUDA Settings

```bash
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
python scripts/capture_with_alignment.py ...
```

### PyTorch Settings

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python scripts/capture_with_alignment.py --batch-size 4
```

---

## Troubleshooting

### Script Execution Errors

**Problem:** `python: can't open file 'scripts/...'`
```bash
# Make sure you're in the repo root
cd /path/to/phonetic-sae
python scripts/validate_environment.py
```

**Problem:** `ModuleNotFoundError: No module named 'src'`
```bash
# Make sure Python path is set correctly
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/validate_environment.py
```

### Permission Errors

```bash
# Make scripts executable
chmod +x scripts/*.sh scripts/*.py

# Or run with python explicitly
python scripts/setup.sh  # This won't work
bash scripts/setup.sh    # Correct way
```

### Out of Memory

```bash
# Reduce batch size
python scripts/capture_with_alignment.py --batch-size 2

# Or use CPU (much slower)
python scripts/capture_with_alignment.py --device cpu --batch-size 1
```

---

## Next Steps

1. **Run setup:** `bash scripts/setup.sh`
2. **Read quickstart:** `docs/PHASE1_QUICKSTART.md`
3. **Try pilot capture:** See "Typical Workflows" above
4. **Scale to full dataset:** See `docs/PHASE1_QUICKSTART.md` Step 6
5. **Phase 2:** Train SAE on captured activations

For detailed guides, see:
- 📖 `docs/PHASE1_QUICKSTART.md` - Step-by-step Phase 1
- 🔧 `docs/PHONEME_ALIGNMENT.md` - Detailed alignment guide
- ⚙️ `docs/QWEN3_FORCEDALIGNER_INFERENCE.md` - Model internals
- 📚 `docs/ACTUAL_API_DISCOVERY.md` - API verification guide
