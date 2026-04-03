# Qwen3-TTS Generation API: Input Construction & Language Codec

**Issue:** "Model does not support base generation mode" — calling `generate()` without `ref_audio`.

**Root Cause:** Qwen3-TTS-0.6B **only supports voice cloning mode**. There is no standalone TTS mode.

---

## 1. Why "Base Generation" Fails

The Base model requires **speaker embedding** (extracted from `ref_audio`) for synthesis.

```
generate_voice_clone(text, language, ref_audio, ref_text)
    ↓
1. Extract speaker embedding from ref_audio (x-vector, shape [1024])
2. Extract speech codes from ref_audio (codec tokens for ICL mode)
3. Tokenize text via Qwen2 tokenizer
4. Assemble inputs_embeds (NOT input_ids)
5. Talker forward pass → 1st codebook token
6. Code Predictor → remaining 15 codebook tokens
7. Decoder → waveform
```

Calling with `ref_audio=None` fails at step 1 — no speaker embedding can be extracted.

---

## 2. CRITICAL: The Talker Uses `inputs_embeds`, NOT `input_ids`

**The Talker does NOT receive token IDs.** All inputs are pre-embedded and concatenated in embedding space.

### 2.1 Actual Embedding Assembly (from `modeling_qwen3_tts.py`)

```python
# Step 1: Text tokenization (via Qwen2 tokenizer)
text = "<|im_start|>assistant\n{synthesis_text}<|im_end|>\n<|im_start|>assistant\n"
text_ids = processor(text=text)["input_ids"]  # shape: [1, seq_len]

# Step 2: Language codec prefix (converted to EMBEDDINGS, not prepended as IDs)
if language != "auto":
    language_id = config.talker_config.codec_language_id[language.lower()]
    codec_prefix = [codec_think_id, codec_think_bos_id, language_id, codec_think_eos_id]
else:
    codec_prefix = [codec_nothink_id, codec_think_bos_id, codec_think_eos_id]

codec_prefix_embed = talker.get_input_embeddings()(tensor(codec_prefix))
# shape: [1, 3 or 4, d_model=1024]

# Step 3: Speaker embedding injection (as a SINGLE embedding token)
speaker_embed = speaker_encoder(ref_audio)  # shape: [1024]
speaker_embed = speaker_embed.view(1, 1, 1024)  # [1, 1, d_model]

# Step 4: Codec pad/bos tokens
codec_pad_bos_embed = talker.get_input_embeddings()(tensor([codec_pad_id, codec_bos_id]))
# shape: [1, 2, d_model]

# Step 5: Concatenate in embedding space
codec_embed = cat([codec_prefix_embed, speaker_embed, codec_pad_bos_embed], dim=1)
# shape: [1, 6-7, d_model]

# Step 6: Final assembly
talker_input_embeds = cat([role_embed, codec_embed, text_embed, optional_icl_embed], dim=1)
# shape: [1, total_seq_len, d_model=1024]

# Step 7: Talker forward (uses inputs_embeds, NOT input_ids)
output = talker.generate(inputs_embeds=talker_input_embeds, attention_mask=..., ...)
```

### 2.2 Talker Input Sequence Layout

```
Position:  0..2       3..6          7          8..9         10..N         N+1..M
Content:   [role]     [codec_pfx]   [spk_emb]  [pad,bos]    [text]        [icl_codes]
           im_start   think         speaker     codec        synthesis     ref speech
           assistant  +lang_id      embedding   padding      text tokens   (optional)
                      +think_eos
```

**All positions are in embedding space** (d_model=1024). The Talker's `forward()` receives `inputs_embeds` tensor, not `input_ids`.

### 2.3 Language Codec ID Values

These are codec token IDs that get **embedded** (not used as raw IDs in the Talker):

| Language | Codec ID | Config Key |
| :--- | :--- | :--- |
| English | 2050 | `codec_language_id["english"]` |
| Chinese | 2055 | `codec_language_id["chinese"]` |
| Spanish | 2054 | `codec_language_id["spanish"]` |
| German | 2053 | `codec_language_id["german"]` |
| Portuguese | 2071 | `codec_language_id["portuguese"]` |
| Italian | 2070 | `codec_language_id["italian"]` |
| Japanese | 2058 | `codec_language_id["japanese"]` |
| Korean | 2064 | `codec_language_id["korean"]` |
| French | 2061 | `codec_language_id["french"]` |
| Russian | 2069 | `codec_language_id["russian"]` |

### 2.4 Other Special Codec Token IDs

| Token | ID | Purpose |
| :--- | :--- | :--- |
| codec_pad_id | 2148 | Padding |
| codec_bos_id | 2149 | Begin of speech sequence |
| codec_eos_token_id | 2150 | End of speech sequence |
| codec_think_id | 2154 | Think prefix (used when language specified) |
| codec_nothink_id | 2155 | No-think prefix (used for "auto" language) |
| codec_think_bos_id | 2156 | Think section begin |
| codec_think_eos_id | 2157 | Think section end |

---

## 3. The Wrapper Bug

**Current code (`qwen3_tts_wrapper.py` lines 320-325) — BROKEN:**

```python
wavs, sr = self.model.generate_voice_clone(
    text=text,
    language=lang_code,
    ref_audio=None,          # ❌ Cannot extract speaker embedding from None
    ref_text="",             # ❌ No ICL context
)
```

**Fix: Always provide `ref_audio`.**

---

## 4. Solution: Use Reference Audio from HF Dataset

The `indiejoseph/tts20250516` HF dataset contains audio samples that can serve as reference:

```python
# Download one sample as default reference
from datasets import load_dataset
ds = load_dataset("indiejoseph/tts20250516", split="train", streaming=True)
sample = next(iter(ds))
sf.write("assets/default_ref.wav", sample["audio"]["array"], sample["audio"]["sampling_rate"])
```

Then use in capture:

```python
_ = model_wrapper.generate(
    text=pair.text,
    language="English",
    ref_audio="assets/default_ref.wav",
    ref_text="Reference text from dataset.",
    x_vector_only_mode=True,  # Simplest: just speaker embedding, no ICL
)
```

---

## 5. CPU Debug Script

Use `scripts/debug_cpu_capture.py` for local testing without GPU:

```bash
PYTHONPATH="." python scripts/debug_cpu_capture.py \
    --model qwen3tts \
    --lang en \
    --num-samples 1 \
    --max-tokens 64
```

This script:
1. Loads model on CPU (float32)
2. Downloads reference audio from HF dataset automatically
3. Calls `generate_voice_clone()` with proper parameters
4. Captures activations from layers 1-7
5. Prints activation shapes and statistics

---

## 6. Activation Capture Implications

### What the hooks capture

MLP post-activation hooks on layers 1-7 capture tensors of shape `(seq_len, d_model=1024)`.

The sequence includes ALL positions from the assembled `inputs_embeds`:
- **Codec prefix positions** (3-4 tokens): Language embedding signal
- **Speaker embedding position** (1 token): Speaker identity
- **Pad/BOS positions** (2 tokens): Structural markers
- **Text positions** (variable): Phonetic processing — **this is what we want**
- **ICL positions** (optional): Reference speech context

### Filtering activations for phonetic analysis

For SAE training, we should **filter to text positions only**:

```python
# After collecting activations:
# activations shape: (total_seq_len, 1024)
# text_start = len(codec_prefix) + 1 (speaker) + 2 (pad/bos) = ~7
# text_end = text_start + len(text_tokens)
# phonetic_activations = activations[text_start:text_end, :]
```

This separates phonetic features from language/speaker/structural features.

---

## 7. Checklist

- [ ] Provide `ref_audio` to `generate_voice_clone()` (use HF dataset sample)
- [ ] Set language parameter (e.g., "English", "Chinese")
- [ ] Choose mode: `x_vector_only_mode=True` (faster) or `False` (richer)
- [ ] Capture at MLP post-activations (layers 1-7)
- [ ] Filter captured activations to text positions only
- [ ] Store with metadata: language, text, sample_id, text_start_pos

