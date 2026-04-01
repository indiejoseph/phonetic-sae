# Phoneme Set Sources and Verification

This document explains where the phoneme sets came from and how to verify them against the actual Qwen3-ForcedAligner model.

---

## Where Did They Come From?

The phoneme sets in `src/alignment/forced_aligner.py` are based on **standard, industry-accepted phoneme inventories** for each language:

### English ("en") — ARPAbet
**Source:** TIMIT Phoneme Set (standard in English speech recognition)
- **43 phonemes** total
- **Standard for:** LibriSpeech, CommonVoice, most English TTS/ASR models
- **Reference:** [TIMIT Acoustic-Phonetic Continuous Speech Corpus](https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir4930.pdf)

**Example phonemes:**
```
Consonants: b, ch, d, dh, f, g, hh, jh, k, l, m, n, ng, p, r, s, sh, t, th, v, w, y, z, zh
Vowels: aa, ae, ah, ao, aw, ay, eh, er, ey, ih, iy, oh, ow, oy, uh, uw
Silence: pau, sil
```

### Mandarin ("zh") — Pinyin
**Source:** Standard Mandarin phonology (Hanyu Pinyin)
- **~40 phonemes** (initials + finals)
- **Standard for:** Most Chinese TTS models (Microsoft, Baidu, Alibaba, Tencent)
- **Reference:** [Hanyu Pinyin System](https://en.wikipedia.org/wiki/Pinyin)

**Example phonemes:**
```
Initials: b, p, m, f, d, t, n, l, g, k, h, j, q, x, zh, ch, sh, r, z, c, s
Finals: a, o, e, i, u, ü, ai, ei, ao, ou, an, en, ang, eng, ong, ...
```

### Cantonese ("yue") — Jyutping
**Source:** Jyutping romanization (standard academic system)
- **~40 phonemes** (initials + finals)
- **Standard for:** Academic Cantonese research
- **Reference:** [Jyutping](https://en.wikipedia.org/wiki/Jyutping)

**Example phonemes:**
```
Initials: p, ph, m, f, t, th, n, l, k, kh, ng, h, gw, kw, z, c, s, j
Finals: a, e, i, o, u, oe, ai, ei, oi, ou, au, an, en, on, ...
```

---

## ⚠️ Critical Point: Verification Required

**The phoneme sets above are DEFAULTS based on standard inventories, but the actual Qwen3-ForcedAligner model may use DIFFERENT phoneme sets.**

When the `QwenForcedAligner` is initialized, it attempts to:
1. Extract the actual phoneme inventory from the loaded model
2. Fall back to defaults if extraction fails

**You MUST verify the actual phoneme sets used by your model.**

---

## How to Verify the Actual Phoneme Sets

### Method 1: Run the Inspection Script

```bash
python scripts/inspect_aligner.py
# or for a specific language
python scripts/inspect_aligner.py --lang en
```

**Output:**
- Prints actual phoneme sets from the model
- Saves to `phoneme_inventory_{lang}.json`
- Validates sample phonemes

**Example output:**
```
======================================================================
LANGUAGE: EN
======================================================================

✅ Loaded aligner for 'en'
   Phonemes (43): ['aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', 'd', ...]
   Saved to: phoneme_inventory_en.json

   Sample validation: ['aa', 'ae', 'ah'] → Valid: True
```

### Method 2: Query Directly in Python

```python
from src.alignment import QwenForcedAligner

# Load aligner
aligner = QwenForcedAligner(device="cuda", language="en")

# Get actual phoneme set
phonemes = aligner.get_phoneme_inventory()
print(f"English phonemes ({len(phonemes)}): {phonemes}")

# Validate specific phonemes
test_phonemes = ["m", "ə", "l"]
valid = aligner.validate_phonemes(test_phonemes)
print(f"Are {test_phonemes} valid? {valid}")
```

### Method 3: Check Model Attributes

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Qwen/Qwen3-ForcedAligner-0.6B",
    trust_remote_code=True
)

# Inspect model for phoneme-related attributes
for attr in dir(model):
    if "phoneme" in attr.lower() or "vocab" in attr.lower():
        print(f"Found: model.{attr}")
```

---

## What If the Actual Phoneme Sets Are Different?

If the model uses different phoneme sets than our defaults:

### Option A: Update the Defaults (Quick Fix)
```python
# In src/alignment/forced_aligner.py
PHONEME_SETS = {
    "en": [actual_en_phonemes],  # Replace with actual
    "zh": [actual_zh_phonemes],
    "yue": [actual_yue_phonemes],
}
```

### Option B: Use Model's Phoneme Set Dynamically (Recommended)
The code already does this via `_extract_phoneme_set()`. The model's actual inventory is used if extraction succeeds.

### Option C: Create Per-Language Configs
```python
# configs/phoneme_sets.json
{
    "en": {"phonemes": [actual list], "source": "model_extraction"},
    "zh": {"phonemes": [actual list], "source": "model_extraction"},
    "yue": {"phonemes": [actual list], "source": "model_extraction"}
}
```

---

## Action Items

### Before Running Aligned Capture:

1. **Run inspection script:**
   ```bash
   python scripts/inspect_aligner.py
   ```

2. **Verify outputs:**
   - Check `phoneme_inventory_*.json` files
   - Confirm phoneme counts match expectations

3. **Update if needed:**
   - If actual phonemes differ significantly from defaults, update the code
   - Document the source in the PHONEME_SETS dictionary

4. **Test with sample:**
   ```python
   from src.alignment import QwenForcedAligner
   aligner = QwenForcedAligner(language="en")
   alignment = aligner.align(
       text="hello world",
       audio_path="sample_audio.wav"
   )
   print(alignment.phonemes)  # Verify output format
   ```

---

## References

- **ARPAbet:** https://en.wikipedia.org/wiki/ARPABET
- **Pinyin:** https://en.wikipedia.org/wiki/Pinyin
- **Jyutping:** https://en.wikipedia.org/wiki/Jyutping
- **Qwen3-ForcedAligner:** https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B
- **TIMIT Corpus:** https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir4930.pdf

---

## Summary

| Aspect | Status | Action |
|--------|--------|--------|
| **Default Sets** | ✅ Based on standard inventories | Reference only |
| **Model Extraction** | ✅ Automatic | No action needed |
| **Verification** | ⚠️ User must verify | Run `inspect_aligner.py` |
| **Fallback** | ✅ Implemented | Handles if extraction fails |

**Next Step:** Run the inspection script to verify the actual phoneme sets your model uses!
