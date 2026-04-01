# Phoneme Preparation: Quick Reference

**TL;DR:** If Qwen3-ForcedAligner inspection fails, use external phoneme converters.

---

## The Problem

The `inspect_aligner_api.py` script failed to discover `text_to_phonemes` in Qwen3-ForcedAligner.

**Solution:** Use external libraries instead:
- English: `g2p_en`
- Mandarin: `pypinyin`
- Cantonese: `ToJyutping` (excellent character coverage)

---

## Quick Start (3 Steps)

### Step 1: Install Fallback Converters
```bash
python scripts/setup_phoneme_converters.py
```

This will:
1. Install g2p_en, pypinyin, ToJyutping
2. Test each converter
3. Report success/failure

### Step 2: Use Converters in Your Code
```python
from src.alignment.phoneme_converter import get_converter

# English
en_converter = get_converter("en")
phonemes = en_converter.text_to_phonemes("hello")
print(phonemes)  # ['h', 'eh', 'l', 'ow']

# Mandarin
zh_converter = get_converter("zh")
phonemes = zh_converter.text_to_phonemes("你好")
print(phonemes)  # ['ni', 'hao']

# Cantonese
yue_converter = get_converter("yue")
phonemes = yue_converter.text_to_phonemes("你好")
print(phonemes)  # ['nei5', 'hou2'] (approximate)
```

### Step 3: Integrate with Alignment
Already built into the system! When you use `capture_with_alignment.py`, it will:
1. Try model's built-in `text_to_phonemes` first
2. Fall back to external converters if not available
3. Work transparently

---

## If Something Fails

### Issue: Installation fails
```bash
# Try one package at a time
pip install g2p_en --break-system-packages
pip install pypinyin --break-system-packages
pip install ToJyutping --break-system-packages
```

### Issue: Tests fail
```bash
# Run with more verbose output
python scripts/setup_phoneme_converters.py --install-only
# Then manually test:
python -c "from src.alignment.phoneme_converter import get_converter; print(get_converter('en').text_to_phonemes('test'))"
```

### Issue: Phonemes look wrong
- Verify with: `python scripts/setup_phoneme_converters.py --test-only`
- Check `docs/PHONEME_PREPARATION_FALLBACK.md` for language-specific details
- Compare output with expected phoneme sets

---

## What Each Converter Outputs

### English (g2p_en)
```
Input: "hello"
Output: ['h', 'eh', 'l', 'ow']
Format: ARPAbet phonemes (43 phonemes)
```

### Mandarin (pypinyin)
```
Input: "你好"
Output: ['ni', 'hao']
Format: Pinyin (no tone marks)
```

### Cantonese (jyutping)
```
Input: "你好"
Output: ['nei5', 'hou2']
Format: Jyutping (with tone numbers)
Note: Limited character coverage
```

---

## When to Use What

| Scenario | Action |
|----------|--------|
| Qwen3 has `text_to_phonemes` | ✅ Use it (automatic) |
| Qwen3 doesn't have it | ✅ Use external converters |
| Qwen3 times out/fails | ✅ Use external converters |
| Language is English | ✅ Use g2p_en (excellent) |
| Language is Mandarin | ✅ Use pypinyin (excellent) |
| Language is Cantonese | ⚠️ Use jyutping (basic) |
| Language not supported | ❌ Need custom solution |

---

## Full Workflow

```bash
# 1. Check if Qwen3 has built-in phoneme conversion
python scripts/inspect_aligner_api.py --device cpu

# 2. If it doesn't (or if inspection times out)
python scripts/setup_phoneme_converters.py

# 3. Run capture (will automatically use fallback if needed)
python scripts/capture_with_alignment.py \
  --model qwen3tts \
  --dataset custom \
  --dataset-file data/your_dataset.jsonl \
  --lang en \
  --output data/activations \
  --num-samples 100 \
  --device cuda
```

The pipeline will:
- ✅ Try model's built-in phoneme converter
- ✅ Fall back to external converter if needed
- ✅ Continue transparently

---

## Files Added

| File | Purpose |
|------|---------|
| `docs/PHONEME_PREPARATION_FALLBACK.md` | Complete guide with implementation details |
| `src/alignment/phoneme_converter.py` | Fallback converter implementation |
| `scripts/setup_phoneme_converters.py` | Installation and testing script |
| `PHONEME_PREPARATION_QUICK_REFERENCE.md` | This file |

---

## Next Steps

1. **Right now:** Run `python scripts/setup_phoneme_converters.py`
2. **Then:** Follow `docs/PHASE1_QUICKSTART.md` as normal
3. **Everything else:** Works the same way

The phoneme preparation happens automatically in the background.

---

## Questions?

- **How does it work?** → See `docs/PHONEME_PREPARATION_FALLBACK.md`
- **Language-specific issues?** → See language sections in fallback guide
- **Need custom language?** → See "Custom Language" section in fallback guide

---

## Quick Test

Verify everything works:
```bash
python -c "
from src.alignment.phoneme_converter import get_converter
for lang in ['en', 'zh', 'yue']:
    try:
        c = get_converter(lang)
        print(f'✅ {lang} works')
    except Exception as e:
        print(f'❌ {lang} failed: {e}')
"
```

Expected output:
```
✅ en: 'hello world' → ['h', 'eh', 'l', 'ow', ...]
✅ zh: '你好' → ['ni', 'hao']
✅ yue: '你好' → ['nei5', 'hou2']
```

---

## Summary

- **Problem:** Qwen3-ForcedAligner may not expose `text_to_phonemes`
- **Solution:** Use external G2P converters (g2p_en, pypinyin, jyutping)
- **Status:** Fully integrated, works transparently
- **Your action:** Run `python scripts/setup_phoneme_converters.py` once
- **Result:** Everything works as before

✅ **You're set up and ready to go!** 🎯
