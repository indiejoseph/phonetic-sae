# Fix: Qwen3-TTS "Base Generation" Error in Activation Capture

**Error:**
```
Generation failed: Model does not support base generation mode
```

**Location:** `scripts/capture_with_alignment.py`, line 271

**Problem:** The wrapper is calling `model_wrapper.generate(text=pair.text)` with **no reference audio**, but Qwen3-TTS requires `ref_audio` + `ref_text` to extract speaker embedding.

---

## Root Cause

Qwen3-TTS architecture requires:

```
Text → Talker → Code Predictor → Decoder → Waveform
         ↑
      Speaker Embedding (from ref_audio)
      + Optional: Reference Codec Tokens (ICL mode, from ref_audio)
```

**Without ref_audio, there is no speaker embedding → synthesis fails.**

---

## The Fix

### Step 1: Prepare a Default Reference Audio

Create or download a reference audio file:

```bash
# Option A: Download a sample (example only)
mkdir -p assets
wget -O assets/default_reference_en.wav "https://example.com/reference.wav"

# Option B: Use any existing audio file
cp /path/to/existing/audio.wav assets/default_reference_en.wav
```

**Tip:** The reference audio should be:
- A single English speaker (at least 1-2 seconds)
- Clean audio (no background noise)
- Natural speech (not whispered or shouted)

### Step 2: Modify capture_with_alignment.py

**File: `scripts/capture_with_alignment.py`**

**Find:** Lines 269-274

**Replace:**
```python
# ❌ OLD (BROKEN)
with torch.no_grad():
    if args.model == "qwen3tts":
        _ = model_wrapper.generate(text=pair.text)
    else:  # cosyvoice2
        _ = model_wrapper.generate(tts_text=pair.text, prompt_text="Reference")
```

**With:**
```python
# ✅ NEW (FIXED)
with torch.no_grad():
    if args.model == "qwen3tts":
        # Provide reference audio for speaker embedding extraction
        default_ref_audio = "assets/default_reference_en.wav"
        default_ref_text = "The quick brown fox jumps over the lazy dog."
        
        _ = model_wrapper.generate(
            text=pair.text,
            language="English",  # Can be extracted from pair.language if available
            ref_audio=default_ref_audio,
            ref_text=default_ref_text,
            x_vector_only_mode=False,  # Use ICL mode for better phonetic grounding
        )
    else:  # cosyvoice2
        _ = model_wrapper.generate(
            tts_text=pair.text,
            prompt_text="Reference",
            prompt_wav="assets/default_reference_prompt.wav",
        )
```

### Step 3: (Optional) Update Wrapper's Error Handling

**File: `src/models/qwen3_tts_wrapper.py`**

**Find:** Lines 311-338 (the "else" block in `generate()`)

**Replace with:**
```python
# Base mode: voice cloning with default or provided reference
else:
    if ref_audio is None:
        raise ValueError(
            "Qwen3-TTS requires ref_audio (reference audio path or URL) "
            "for voice cloning. Please provide a reference audio file."
        )

    try:
        lang_code = self._get_language_code(language)
        
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=lang_code,
            ref_audio=ref_audio,  # Now required
            ref_text=ref_text or "Reference audio for voice cloning",
            x_vector_only_mode=x_vector_only_mode if x_vector_only_mode else False,
            **kwargs,
        )
        return {"waveform": wavs[0] if isinstance(wavs, list) else wavs, "sample_rate": sr}

    except Exception as e:
        logger.error(f"Voice cloning generation failed: {e}")
        raise
```

---

## Understanding the Language Codec ID

When you call with `language="English"`:

```python
1. generate_voice_clone(text="Hello", language="English", ref_audio=...)
                                              ↓
2. Internal mapping: "English" → language codec ID = 2050
                                              ↓
3. Input to Talker: [2050] + tokenize("Hello") 
                     ↑
                Language codec ID prepended to input sequence
```

**Language to codec ID mapping:**
- English: 2050
- Chinese: 2055
- Spanish: 2054
- German: 2053
- Portuguese: 2071
- Italian: 2070
- Japanese: 2058
- Korean: 2064
- French: 2061
- Russian: 2069

**This is critical for activation capture:** The language codec ID appears at **position 0** of every input token sequence.

---

## How to Extract Language from Your Dataset

If your dataset has language information (like "en", "zh", "yue"):

```python
# In capture_with_alignment.py, modify line 217
lang_map = {
    "en": "English",
    "zh": "Chinese",
    "yue": "Cantonese",
    # Add more as needed
}

# When getting language parameter:
language_for_model = lang_map.get(args.lang, "English")

_ = model_wrapper.generate(
    text=pair.text,
    language=language_for_model,  # "English" instead of "en"
    ref_audio=default_ref_audio,
    ref_text=default_ref_text,
)
```

---

## Verification: Test the Fix

After applying the fix, run:

```bash
python scripts/capture_with_alignment.py \
    --model qwen3tts \
    --dataset-file data/out.jsonl \
    --lang en \
    --output data/pilot_activations \
    --num-samples 2 \
    --device cuda
```

**Expected output (instead of error):**
```
[1/2] Processing: Your sample text here...
Generating base TTS for: Your sample text...
✓ Activations captured from 7 layers
... alignment processing ...
[2/2] Processing: Another sample...
Generating base TTS for: Another sample...
✓ Activations captured from 7 layers
... alignment processing ...

Capture Complete:
  Total samples: 2
  Successfully processed: 2
  ...
```

---

## What Gets Captured

When the fix is applied, activations are captured from:

**Talker's MLP post-activations (Layers 1-7):**

```
Input:  [language_codec_id=2050] + text_tokens + optional_icl_tokens
         ↓
Layer 1 MLP: activations shape (seq_len, 1024) ← CAPTURED
Layer 2 MLP: activations shape (seq_len, 1024) ← CAPTURED
...
Layer 7 MLP: activations shape (seq_len, 1024) ← CAPTURED
         ↓
Code Predictor: generates remaining 15 codebooks
         ↓
Decoder: generates waveform
```

**Activation metadata saved:**
- Language (from language_codec_id)
- Text (for phoneme alignment)
- Layer index
- Sample ID
- Sequence length

---

## FAQ

**Q: Do I need a reference audio for every sample?**
A: No. One reference audio per language is sufficient. All samples with `language="English"` can use the same reference.

**Q: What if I don't have reference audio?**
A: You can generate one using Qwen3-TTS itself (recursively), but this requires a bootstrap approach. For now, use a pre-recorded reference.

**Q: Why does Qwen3-TTS require ref_audio?**
A: The speaker embedding (x-vector) extracted from ref_audio conditions the Talker LLM to synthesize speech that sounds like the reference speaker. This is by design.

**Q: Can I use different reference audio per sample?**
A: Yes, but it will change the speaker identity for each sample. For consistent phonetic feature capture, use the same reference for all samples in a language.

**Q: What about language codec IDs?**
A: They are automatically prepended by the model. You just need to pass the language name (e.g., "English"), and the model converts it to codec_id (2050) internally.

---

## Summary Checklist

- [ ] Download/prepare default reference audio
- [ ] Place it in `assets/default_reference_en.wav`
- [ ] Update `scripts/capture_with_alignment.py` line 271
- [ ] Update `src/models/qwen3_tts_wrapper.py` (optional but recommended)
- [ ] Test with 2-3 samples
- [ ] Verify activations are captured
- [ ] Scale up to full dataset

