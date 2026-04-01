# Documentation Index

**Quick Navigation Guide for All Phase 1 Documentation**

---

## 🚀 Getting Started (Pick One)

### I just want to run it
→ **[QUICK_START.md](QUICK_START.md)** (This page: 2 minutes to first run)

- Copy/paste commands
- Common options table
- Troubleshooting quick reference

### I want to understand what's happening
→ **[docs/ACTIVATION_CAPTURE_WORKFLOW.md](docs/ACTIVATION_CAPTURE_WORKFLOW.md)** (Complete workflow: 20 minutes)

- Architecture diagram
- Step-by-step workflow
- Code examples with explanations
- Design decisions
- Performance benchmarks

### I want to know the current status
→ **[docs/PHASE1_STATUS.md](docs/PHASE1_STATUS.md)** (Status report: 15 minutes)

- What works ✅
- What's been fixed
- Hardware requirements
- Test scripts available
- Next steps

### Something's broken
→ **[docs/ACTIVATION_CAPTURE_TROUBLESHOOTING.md](docs/ACTIVATION_CAPTURE_TROUBLESHOOTING.md)** (Reference: As needed)

- 30+ common errors
- Specific fixes for each
- Diagnostic checklist
- Performance tuning

---

## 📚 Detailed References

### Model Architecture
→ **[docs/QWEN3_TTS_STRUCTURE.md](docs/QWEN3_TTS_STRUCTURE.md)**

- Qwen3-TTS wrapper hierarchy
- Layer access paths
- PEFT wrapper explanation
- Why nested structure exists
- Debugging tips

### Project Context
→ **[docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)**

- Research goals
- Mechanistic interpretability
- Why SAEs for TTS
- Related work

### Implementation Details
→ **[docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)**

- 5-phase research roadmap
- Phase 1-5 objectives
- Research references
- FAQ

---

## 🔍 Session Work

### This Session's Fixes
→ **[PHASE1_FIX_SUMMARY.md](PHASE1_FIX_SUMMARY.md)** (What changed: 10 minutes)

- Root cause of 'not callable' error
- How it was fixed
- Why the fix works
- Critical design decisions
- What now works

---

## 🧪 Scripts & Validation

### Test Installation
```bash
python scripts/test_activation_capture.py
```

→ **[scripts/test_activation_capture.py](scripts/test_activation_capture.py)**

- Validates complete pipeline
- Tests model loading
- Tests hook attachment
- Tests activation capture
- Confirms shapes and dtypes

### Run Full Capture
```bash
python scripts/full_capture.py --help
```

→ **[scripts/full_capture.py](scripts/full_capture.py)**

- Main activation capture script
- Supports pilot/libritts/custom datasets
- Configurable batch size, samples, device
- Progress logging

### Run With Alignment
```bash
python scripts/capture_with_alignment.py --help
```

→ **[scripts/capture_with_alignment.py](scripts/capture_with_alignment.py)**

- Activation capture with phoneme alignment
- Organizes by phoneme
- Generates phoneme inventory
- Supports en/zh/yue languages

---

## 🗂️ Full Documentation Structure

```
phonetic-sae/
├── QUICK_START.md                          ← Commands to run
├── PHASE1_FIX_SUMMARY.md                   ← What was fixed
├── DOCUMENTATION_INDEX.md                  ← This file
├── README.md                               ← Project overview
├── CLAUDE.md                               ← Project instructions
│
├── docs/
│   ├── PHASE1_STATUS.md                    ← Current status
│   ├── ACTIVATION_CAPTURE_WORKFLOW.md      ← Complete workflow
│   ├── ACTIVATION_CAPTURE_TROUBLESHOOTING.md ← Error fixes
│   ├── QWEN3_TTS_STRUCTURE.md              ← Model architecture
│   ├── PROJECT_OVERVIEW.md                 ← Research context
│   ├── PROJECT_PLAN.md                     ← 5-phase roadmap
│   ├── GETTING_STARTED.md                  ← Installation
│   ├── DATASET.md                          ← Data format
│   ├── PHONEME_ALIGNMENT.md                ← Alignment guide
│   └── (other docs)
│
├── scripts/
│   ├── test_activation_capture.py          ← Validation (NEW)
│   ├── full_capture.py                     ← Main script
│   ├── capture_with_alignment.py           ← With alignment
│   ├── train_sae.py                        ← SAE training
│   └── (other scripts)
│
└── src/
    ├── models/
    │   ├── qwen3_tts_wrapper.py            ← Model wrapper
    │   └── cosyvoice2_wrapper.py           ← Alternative model
    ├── hooks.py                            ← Activation hooks
    ├── data/
    │   └── activation_buffer.py            ← Storage
    └── (other modules)
```

---

## 🎯 Decision Tree: Which Doc Do I Need?

```
┌─ "I want to run the code now"
│  └─→ QUICK_START.md
│
├─ "I'm getting an error"
│  └─→ ACTIVATION_CAPTURE_TROUBLESHOOTING.md
│
├─ "I want to understand the workflow"
│  └─→ ACTIVATION_CAPTURE_WORKFLOW.md
│
├─ "I want to know what was fixed"
│  └─→ PHASE1_FIX_SUMMARY.md
│
├─ "I want to know the current status"
│  └─→ PHASE1_STATUS.md
│
├─ "I want to understand the model architecture"
│  └─→ QWEN3_TTS_STRUCTURE.md
│
├─ "I want the research context"
│  └─→ PROJECT_OVERVIEW.md
│
├─ "I want the full research plan"
│  └─→ PROJECT_PLAN.md
│
└─ "I want to know how to format my data"
   └─→ DATASET.md
```

---

## 📋 Reading Paths by Role

### 👨‍💻 Developer/Experimenter
1. **QUICK_START.md** — Get it running (5 min)
2. **ACTIVATION_CAPTURE_WORKFLOW.md** — Understand the code (20 min)
3. **QWEN3_TTS_STRUCTURE.md** — Understand the model (10 min)
4. **ACTIVATION_CAPTURE_TROUBLESHOOTING.md** — Reference when needed

### 🧪 Researcher
1. **PHASE1_STATUS.md** — What's ready (10 min)
2. **PROJECT_OVERVIEW.md** — Research context (10 min)
3. **PROJECT_PLAN.md** — Full roadmap (15 min)
4. **ACTIVATION_CAPTURE_WORKFLOW.md** — How it works (20 min)

### 🔧 DevOps/System Admin
1. **QUICK_START.md** — Commands (5 min)
2. **ACTIVATION_CAPTURE_TROUBLESHOOTING.md** — Common issues (10 min)
3. **PHASE1_STATUS.md** — Hardware requirements (5 min)

### 📚 Onboarding New Team Member
1. **README.md** — Project overview (10 min)
2. **GETTING_STARTED.md** — Setup (15 min)
3. **QUICK_START.md** — First run (5 min)
4. **ACTIVATION_CAPTURE_WORKFLOW.md** — Deep dive (20 min)
5. **PROJECT_PLAN.md** — Research context (15 min)

---

## 🔗 Quick Links

### Running Code
- **[QUICK_START.md](QUICK_START.md)** — Copy/paste commands
- **[scripts/test_activation_capture.py](scripts/test_activation_capture.py)** — Validation script
- **[scripts/full_capture.py](scripts/full_capture.py)** — Main capture script

### Understanding
- **[ACTIVATION_CAPTURE_WORKFLOW.md](docs/ACTIVATION_CAPTURE_WORKFLOW.md)** — How it works
- **[QWEN3_TTS_STRUCTURE.md](docs/QWEN3_TTS_STRUCTURE.md)** — Model architecture
- **[PROJECT_PLAN.md](docs/PROJECT_PLAN.md)** — Research roadmap

### Troubleshooting
- **[ACTIVATION_CAPTURE_TROUBLESHOOTING.md](docs/ACTIVATION_CAPTURE_TROUBLESHOOTING.md)** — Error fixes
- **[PHASE1_STATUS.md](docs/PHASE1_STATUS.md)** — Status & what works

### Context
- **[PHASE1_FIX_SUMMARY.md](PHASE1_FIX_SUMMARY.md)** — This session's work
- **[README.md](README.md)** — Project overview
- **[CLAUDE.md](CLAUDE.md)** — Project instructions

---

## 🆕 New Documentation in This Session

All created 2026-04-01:

1. **QUICK_START.md** — Quick reference for running commands
2. **PHASE1_FIX_SUMMARY.md** — Summary of what was fixed
3. **DOCUMENTATION_INDEX.md** — This file
4. **docs/PHASE1_STATUS.md** — Complete status report
5. **docs/ACTIVATION_CAPTURE_WORKFLOW.md** — Workflow guide
6. **docs/ACTIVATION_CAPTURE_TROUBLESHOOTING.md** — Troubleshooting guide
7. **scripts/test_activation_capture.py** — Validation script
8. **MEMORY.md updates** — Persistence for future sessions

---

## ✅ Status

| Component | Status | Documentation |
|-----------|--------|-----------------|
| Model loading | ✅ Works | QWEN3_TTS_STRUCTURE.md |
| Hook attachment | ✅ Works | ACTIVATION_CAPTURE_WORKFLOW.md |
| Forward pass | ✅ FIXED | PHASE1_FIX_SUMMARY.md |
| Activation capture | ✅ Works | ACTIVATION_CAPTURE_WORKFLOW.md |
| Storage/buffering | ✅ Works | ACTIVATION_CAPTURE_WORKFLOW.md |
| Phoneme alignment | ✅ Works | PHONEME_ALIGNMENT.md |
| Validation | ✅ Available | test_activation_capture.py |

---

## 🚀 Next Steps

1. **Run validation:** `python scripts/test_activation_capture.py`
2. **Read QUICK_START.md** for copy/paste commands
3. **Start with 100 samples** to verify setup
4. **Run full 50K capture** when ready
5. **Proceed to Phase 2** once captures complete

---

**Last updated:** 2026-04-01
**Status:** All documentation complete ✅
**Next action:** `python scripts/test_activation_capture.py`
