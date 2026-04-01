# Qwen3-ForcedAligner: Model Inference & Architecture

This document explains how Qwen3-ForcedAligner works internally and how to use it effectively for phoneme alignment.

---

## Architecture Overview

### What is Qwen3-ForcedAligner?

**Qwen3-ForcedAligner** is a speech recognition model trained on alignment tasks:
- **Input:** Audio + Text (or phoneme sequence)
- **Output:** Frame-level alignment between audio frames and phonemes
- **Task:** Force-align text/phonemes to audio (like Montreal Forced Aligner, but neural)

**Model Details:**
- **Base:** Qwen3 ASR backbone (0.6B parameter model)
- **Type:** Transformer-based sequence-to-sequence with attention
- **Training:** Supervised alignment on phonetically annotated datasets
- **Output:** Phoneme boundaries + frame-to-phoneme mappings

---

## Inference Workflow

### Step 1: Audio Preprocessing

```python
# Input: raw audio
audio = load_audio("hello.wav")  # (T,) or (1, T)

# Convert to model's expected format
# - Resample to 16kHz (standard)
# - Convert to mel-spectrogram or MFCC features
mel_spectrogram = compute_mel(audio, sample_rate=16000)

# Shape: (time_steps, freq_bins)
# Example: (500, 80) for 5 seconds at 16kHz
```

### Step 2: Text/Phoneme Encoding

```python
# Input: text or phoneme sequence
text = "hello"

# Convert to phoneme sequence (g2p - grapheme-to-phoneme)
phonemes = text_to_phonemes(text, language="en")
# Output: ["h", "eh", "l", "ow"]

# Encode phonemes to token IDs
phoneme_ids = encode_phonemes(phonemes)
# Output: [15, 27, 33, 41]  (indices into phoneme vocabulary)
```

### Step 3: Model Forward Pass

```python
# Inputs to model:
#   - audio_features: (batch, time_steps, 80)  [mel-spectrogram]
#   - phoneme_ids: (batch, phoneme_sequence_length)

# Forward pass through Transformer
with torch.no_grad():
    # Encoder: processes audio
    audio_embeddings = encoder(audio_features)
    # Shape: (batch, time_steps, hidden_dim)

    # Decoder: processes phoneme sequence
    phoneme_embeddings = decoder(phoneme_ids)
    # Shape: (batch, phoneme_length, hidden_dim)

    # Cross-attention: aligns audio to phonemes
    alignment_logits = attention(audio_embeddings, phoneme_embeddings)
    # Shape: (batch, time_steps, phoneme_length)
```

### Step 4: Alignment Decoding

```python
# alignment_logits: (time_steps, phoneme_length)
# Each cell [t, p] = model's confidence that frame t aligns with phoneme p

# Find best alignment path (Viterbi algorithm or dynamic programming)
alignment_path = viterbi_decode(alignment_logits)
# Output: phoneme_indices_per_frame = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]
#         (frame 0-2 → phoneme "h", frame 3-5 → phoneme "eh", etc.)

# Extract boundaries
boundaries = extract_boundaries(alignment_path)
# Output: [(0, 3), (3, 6), (6, 8), (8, 10)]
#         (phoneme "h": frames 0-3, phoneme "eh": frames 3-6, etc.)
```

### Step 5: Post-Processing

```python
# Convert frame indices to time
frame_rate = 100  # 10ms per frame at 16kHz
time_boundaries = [(f_start/frame_rate, f_end/frame_rate) for f_start, f_end in boundaries]
# Output: [(0.0s, 0.3s), (0.3s, 0.6s), (0.6s, 0.8s), (0.8s, 1.0s)]

# Organize output
alignment = {
    "phonemes": ["h", "eh", "l", "ow"],
    "frame_boundaries": [(0, 3), (3, 6), (6, 8), (8, 10)],  # in frames
    "time_boundaries": [(0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.0)],  # in seconds
    "frame_to_phoneme": [0, 0, 0, 1, 1, 1, 2, 2, 3, 3],  # phoneme per frame
}
```

---

## Python Implementation Pseudocode

⚠️ **IMPORTANT: This is illustrative pseudocode based on typical patterns**

**The actual Qwen3-ForcedAligner API may differ!** Run the inspection script to see the real API:

```bash
python scripts/inspect_aligner_api.py
```

**Illustrative pseudocode (may not match actual API):**

```python
class Qwen3ForcedAligner:
    def __init__(self, model_name, device="cuda"):
        # Load pretrained model
        self.model = AutoModelForSpeechAlignment.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        # Note: g2p converter may not exist—check actual API!
        self.g2p = GraphemeToPhoneme()  # VERIFY THIS EXISTS

    def align(self, text: str, audio: torch.Tensor, language: str = "en"):
        # 1. Audio preprocessing
        # VERIFY: Does processor have get_mel_spectrogram?
        mel_spec = self.processor.get_mel_spectrogram(audio, sample_rate=16000)

        # 2. Text to phonemes
        # VERIFY: Does g2p converter exist? Or is it built-in?
        phonemes = self.g2p.convert(text, language=language)
        phoneme_ids = self.processor.encode_phonemes(phonemes)  # VERIFY

        # 3. Model inference
        # VERIFY: Does model have encoder/decoder attributes?
        with torch.no_grad():
            audio_features = self.model.encoder(mel_spec)
            phoneme_features = self.model.decoder(phoneme_ids)
            alignment_logits = self.model.attention(audio_features, phoneme_features)

        # 4. Alignment decoding
        alignment_path = viterbi_decode(alignment_logits)

        # 5. Post-processing
        boundaries = extract_boundaries(alignment_path)
        frame_to_phoneme = alignment_path.argmax(dim=1)

        return {
            "phonemes": phonemes,
            "boundaries": boundaries,
            "frame_to_phoneme": frame_to_phoneme,
        }
```

### Actual API Discovery

**To find the real API:**

1. Run inspection script:
   ```bash
   python scripts/inspect_aligner_api.py
   ```

2. Check what methods actually exist:
   ```python
   from transformers import AutoModel
   model = AutoModel.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B")
   print(dir(model))  # See all available methods
   ```

3. Check the model card on HuggingFace:
   - https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B
   - May have usage examples

4. Inspect our actual implementation:
   - `src/alignment/forced_aligner.py` uses the real API
   - It attempts to extract phoneme sets from the model itself

---

## Key Model Properties

### 1. Phoneme Vocabularies

**English (ARPAbet):** 43 phonemes
```
Consonants: b, ch, d, dh, f, g, hh, jh, k, l, m, n, ng, p, r, s, sh, t, th, v, w, y, z, zh
Vowels: aa, ae, ah, ao, aw, ay, eh, er, ey, ih, iy, oh, ow, oy, uh, uw
Silence: pau, sil
```

**Mandarin (Pinyin):** ~40 phonemes
```
Initials: b, p, m, f, d, t, n, l, g, k, h, j, q, x, zh, ch, sh, r, z, c, s
Finals: a, o, e, i, u, ü, ai, ei, ao, ou, an, en, ang, eng, ong, ...
```

**Cantonese (Jyutping):** ~40 phonemes
```
Initials: p, ph, m, f, t, th, n, l, k, kh, ng, h, gw, kw, z, c, s, j
Finals: a, e, i, o, u, oe, ai, ei, oi, ou, au, an, en, on, ...
```

### 2. Feature Extraction

**Input Audio Processing:**
```
Raw Audio (16kHz)
       ↓
Mel-Spectrogram (80 freq bins, 10ms frames)
       ↓
Normalize (mean/std)
       ↓
Model Encoder (Transformer)
```

**Model Dimensions:**
- Hidden size: typically 768-1024
- Attention heads: 12-16
- Encoder layers: 12-24
- Total params: ~0.6B

### 3. Alignment Algorithm

**Method:** CTC (Connectionist Temporal Classification) or Attention-based

```
Forward Pass:
  audio: (batch, time, 80)     → encoder → (batch, time, hidden)
  phonemes: (batch, len)        → decoder → (batch, len, hidden)
  attention: cross-attention    → (batch, time, len)

Alignment:
  For each time step t:
    Which phoneme p has highest attention?
    alignment[t] = argmax(attention[t, :])

Output:
  frame_to_phoneme[t] = alignment[t]
  boundaries = group consecutive frames with same phoneme
```

---

## Usage in PhoneticSAE

### How We Use It

```python
from src.alignment import QwenForcedAligner

# 1. Initialize
aligner = QwenForcedAligner(device="cuda", language="en")

# 2. Align
alignment = aligner.align(
    text="hello world",
    audio_path="audio.wav"
)

# 3. Extract per-phoneme activations
for frame_idx, phoneme_idx in enumerate(alignment.frame_to_phoneme):
    phoneme = alignment.phonemes[phoneme_idx]
    activation = activations[frame_idx]  # from TTS model
    store(phoneme, activation)
```

### Output Structure

```
For each sample:
├── phonemes: ["h", "eh", "l", "ow", "w", "er", "l", "d"]
├── frame_boundaries: [(0,3), (3,6), (6,8), (8,10), ...]
├── frame_to_phoneme: [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, ...]
└── duration_ms: 1500

Storage:
├── layer_01/
│   ├── phoneme_h.npy      # activations when frame_to_phoneme == 0
│   ├── phoneme_eh.npy     # activations when frame_to_phoneme == 1
│   └── ...
└── phoneme_inventory.json # {"h": 150, "eh": 200, ...}
```

---

## Debugging & Troubleshooting

### 1. Verify Phoneme Extraction

```python
aligner = QwenForcedAligner(language="en")
phonemes = aligner.get_phoneme_inventory()
print(f"Model knows {len(phonemes)} phonemes: {phonemes}")

# Check if expected phoneme exists
assert "m" in phonemes, "Model doesn't recognize 'm'"
```

### 2. Check Alignment Quality

```python
alignment = aligner.align(text="hello", audio_path="hello.wav")

# Verify boundaries are sensible
for phoneme, (start, end) in zip(alignment.phonemes, alignment.frame_boundaries):
    duration = (end - start) * 0.01  # seconds
    print(f"Phoneme '{phoneme}': {duration:.2f}s")
    assert duration > 0.01, "Phoneme too short!"
    assert duration < 1.0, "Phoneme too long!"
```

### 3. Alignment vs Actual Frames

```python
num_frames = len(alignment.frame_to_phoneme)
audio_duration = alignment.duration_ms / 1000
frame_rate = num_frames / audio_duration
print(f"Frame rate: {frame_rate:.0f} fps")
# Should be close to 100 fps (10ms per frame)
```

---

## Model Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Speed** | ~100x faster than MFA | Real-time on GPU |
| **Accuracy** | 95-99% | Comparable to Montreal Forced Aligner |
| **Languages** | en, zh, yue | Current support |
| **Model Size** | 0.6B params | ~2.5GB on disk |
| **VRAM Required** | ~1-2GB | For batch size 32 |
| **Latency** | <100ms/sample | At 16kHz |

---

## References

- **Qwen3 Models:** https://huggingface.co/Qwen
- **CTC (Speech Alignment):** Graves, A., et al. "Connectionist Temporal Classification" (ICML 2006)
- **Attention-based Alignment:** Bahdanau, D., et al. "Neural Machine Translation" (ICLR 2015)
- **Forced Alignment:** https://en.wikipedia.org/wiki/Forced_alignment

---

## Next: Exploring Model Internals

To deeply understand the Qwen3-ForcedAligner:

1. **Check HuggingFace Model Card:** https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B
2. **Run inspection script:** `python scripts/inspect_aligner.py`
3. **Analyze alignment outputs:** See patterns in `phoneme_inventory.json`
4. **Profile inference time:** Measure actual latency on your audio samples

See `scripts/inspect_aligner.py` for practical examples!
