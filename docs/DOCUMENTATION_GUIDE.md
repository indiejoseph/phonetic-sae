# Documentation Guide

This document explains the new organized documentation structure for PhoneticSAE.

---

## 📁 File Organization

All documentation has been reorganized into the `docs/` folder with clear categorization:

### Root Level (Only Essential Files)
```
phonetic-sae/
├── README.md              # Main entry point (unchanged)
├── AGENTS.md              # Agent instructions (unchanged)
└── docs/                  # All other docs organized here
```

### docs/ Structure by Category

#### 🚀 Getting Started
- **GETTING_STARTED.md** — Comprehensive guide covering:
  - 5-minute quick start
  - Full production pipeline
  - Installation & setup
  - Core module usage (ActivationHook, TopKSAE, Trainer, etc.)
  - Configuration & hardware requirements
  - Troubleshooting

#### 📚 Project Information
- **PROJECT_OVERVIEW.md** (was CLAUDE.md) — Project goals, innovation, 5-phase plan
- **PROJECT_PLAN.md** (merged PLAN.md + NEXT_STEPS.md) — 6-week roadmap + implementation priorities
- **EXECUTIVE_SUMMARY.md** — High-level summary for stakeholders
- **IMPLEMENTATION_STATUS.md** — What has been built and current status
- **DELIVERY_REPORT.md** (was PROJECT_COMPLETION_REPORT.md) — Final delivery summary

#### 🛠️ Architecture & Technical
- **QWEN3_TTS_0.6B_ARCHITECTURE.md** — Qwen3-TTS internals & layer details
- **COSYVOICE2_0.5B_ARCHITECTURE.md** — CosyVoice2 internals & layer details
- **MODEL_COMPARISON.md** — Side-by-side comparison of TTS models
- **INFERENCE_AND_INPUT_IDS.md** — Input formatting and inference details
- **MSAE_for_TTS.md** — MSAE integration guide
- **DATASET.md** — Dataset format and preparation
- **PHONEME_ALIGNMENT.md** — Forced aligner setup & per-phoneme feature analysis ⭐ NEW

---

## 📖 Reading Guide by Role

### 👨‍💼 For Decision Makers / Stakeholders
1. Start: **README.md** (30 seconds)
2. Read: **EXECUTIVE_SUMMARY.md** (5 minutes)
3. Reference: **IMPLEMENTATION_STATUS.md** (2 minutes)

### 👨‍💻 For Developers / ML Engineers
1. Start: **README.md** (30 seconds)
2. Follow: **GETTING_STARTED.md** (Installation through module examples)
3. Deep dive: Architecture docs based on your model choice
4. Reference: **PROJECT_PLAN.md** (understand scope & timeline)

### 🔬 For Researchers
1. Start: **PROJECT_OVERVIEW.md** (understand goals)
2. Read: **PROJECT_PLAN.md** (research phases)
3. Dive into: Architecture docs + **MSAE_for_TTS.md**
4. Reference: **DELIVERY_REPORT.md** (current implementation status)

---

## 🔗 Link Map

**README.md contains:**
- Quick links to all major docs
- 5-minute installation guide
- Data preparation overview
- Training pipeline overview

**docs/ contains:**
- **GETTING_STARTED.md** → Full 25+ minute technical guide
- **PROJECT_PLAN.md** → Detailed 6-week roadmap (can replace PLAN.md references)
- **PROJECT_OVERVIEW.md** → Detailed project context (can replace CLAUDE.md references)
- Architecture docs → Model-specific implementation details

---

## 📊 File Categories

### Progress Tracking Docs
Documents that track implementation progress and what's next:
- `PROJECT_PLAN.md` — Roadmap, milestones, next steps
- `IMPLEMENTATION_STATUS.md` — What's been built
- `DELIVERY_REPORT.md` — Final delivery summary

### Informational Docs
Documents that explain the project and how to use it:
- `GETTING_STARTED.md` — How to install and use
- `PROJECT_OVERVIEW.md` — What & why
- `EXECUTIVE_SUMMARY.md` — High-level summary
- `MSAE_for_TTS.md` — MSAE specific guidance
- `DATASET.md` — Dataset details
- `QWEN3_TTS_*.md`, `COSYVOICE2_*.md` — Architecture references

---

## ✅ Reorganization Summary

| What Changed | Before | After |
|--------------|--------|-------|
| **Root clutter** | 10+ .md files | 2 .md files (README.md, AGENTS.md) |
| **Documentation** | Scattered | Organized in `docs/` |
| **Quick start** | Split across multiple files | **GETTING_STARTED.md** (single source of truth) |
| **Project plan** | 2 separate files (PLAN.md + NEXT_STEPS.md) | **PROJECT_PLAN.md** (merged, 1 file) |
| **Project context** | CLAUDE.md (confusing name) | **PROJECT_OVERVIEW.md** (clearer) |
| **Delivery info** | PROJECT_COMPLETION_REPORT.md (long) | **DELIVERY_REPORT.md** (clearer) |

---

## 🎯 Benefits of This Organization

1. **Clear Hierarchy**: Root level only has README.md + AGENTS.md
2. **Easy Navigation**: All docs are in one place with clear categorization
3. **Merged Related Files**: PLAN + NEXT_STEPS → PROJECT_PLAN (easier to find)
4. **Better Names**: CLAUDE.md → PROJECT_OVERVIEW.md (self-explanatory)
5. **Role-Based**: Different docs for stakeholders vs developers vs researchers
6. **Quick Reference**: README.md has categorized links to all docs

---

## 📋 Migration Checklist

If updating external references (GitHub, wikis, etc.):
- [ ] Update README.md links (already done ✅)
- [ ] Search for references to CLAUDE.md → use PROJECT_OVERVIEW.md
- [ ] Search for references to PLAN.md → use PROJECT_PLAN.md
- [ ] Search for references to README_IMPLEMENTATION.md → use GETTING_STARTED.md
- [ ] Search for references to QUICKSTART.md → use GETTING_STARTED.md

---

## 📝 Future Additions

As the project grows, consider adding to `docs/`:
- `notebooks/` — Jupyter notebooks for analysis and visualization
- `TROUBLESHOOTING.md` — Common issues and solutions
- `FAQ.md` — Frequently asked questions
- `API_REFERENCE.md` — Detailed API documentation
- `EXPERIMENTS.md` — Experiment results and findings

---

## 🚀 Quick Access

| Need | File |
|------|------|
| Install & run | **GETTING_STARTED.md** |
| Understand goals | **PROJECT_OVERVIEW.md** |
| See timeline | **PROJECT_PLAN.md** |
| Understand status | **IMPLEMENTATION_STATUS.md** |
| Model internals | **QWEN3_TTS_0.6B_ARCHITECTURE.md** or **COSYVOICE2_0.5B_ARCHITECTURE.md** |
| Setup dataset | **DATASET.md** |
| High-level overview | **EXECUTIVE_SUMMARY.md** |
| MSAE setup | **MSAE_for_TTS.md** |

