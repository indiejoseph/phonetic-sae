# Phoneme Alignment Strategy: Audio-First Approach

**Problem:** G2P converters (g2p_en, pypinyin, jyutping) have too many mistakes.

**Better Solution:** Use **forced alignment from audio**, not text→phoneme conversion.

---

## Why Forced Alignment > G2P

| Method | Accuracy | Pros | Cons |
|--------|----------|------|------|
| **G2P** (text→phoneme) | 85-92% | Fast, no audio needed | Many mistakes |
| **Forced Alignment** (audio→frames) | 95%+ | **High accuracy, audio-ground-truth** | Needs good audio |

**Key insight:** With good audio, forced alignment is **far more reliable** than G2P because the audio provides ground truth.

---

## How Forced Alignment Actually Works

```
Input:  text="hello"  +  audio=[...]
         ↓
Step 1: Rough text→phoneme (g2p_en, pypinyin, etc.)
         "hello" → ['h', 'eh', 'l', 'ow']
         ↓
Step 2: Extract acoustic features from audio
         audio → mel-spectrogram features
         ↓
Step 3: Use Viterbi/HMM to find best phoneme-to-frame alignment
         ['h', 'eh', 'l', 'ow'] ↔ [frame_0:frame_150]
         ↓
Step 4: Output: frame-level phoneme labels
         frame_0-45: 'h'
         frame_45-80: 'eh'
         frame_80-120: 'l'
         frame_120-150: 'ow'
```

**The audio corrects G2P mistakes!** Even if G2P is wrong, Viterbi finds where the phonemes actually are in the audio.

---

## Strategy 1: Use Model's Internal Alignment (Best)

Qwen3-ForcedAligner should already do this internally.

### Check if Model Exposes Alignment API

```bash
python scripts/inspect_aligner_api.py --device cpu
```

Look for methods like:
- `.align(text, audio, language, sample_rate)`
- `.get_boundaries(text, audio)`
- `.get_frame_labels(text, audio)`

### If Model Has Alignment API

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("qwen/Qwen3-ForcedAligner")
processor = model.processor  # or AutoProcessor

# Input: text + audio
text = "hello world"
audio_path = "sample.wav"

# Get frame-level phoneme labels
result = model.align(
    text=text,
    audio_path=audio_path,
    language="en",
    sample_rate=16000
)

# Output structure (varies by model, but typically):
# result.phoneme_indices: [0, 0, 0, 1, 1, 2, 2, 3, 3, ...]  (phoneme per frame)
# result.phoneme_names: ['h', 'eh', 'l', 'ow', 'w', 'er', 'l', 'd']
# result.boundaries: [0, 150, 250, 350, ...]  (frame indices where phonemes change)
```

**This is the gold standard.** If the model has this, use it directly.

---

## Strategy 2: Use Forced Alignment Without Model API

If model doesn't expose phoneme conversion, implement forced alignment yourself.

### Requirements
- Text (word-level OK, doesn't need to be phoneme-level)
- Audio (16kHz WAV, mono or stereo)
- Speech recognition model (Whisper, Wav2Vec2, or Qwen's ASR)

### Simple Approach: Use Whisper for Alignment

```python
import librosa
import numpy as np
from transformers import pipeline
from jiwer import cer

# 1. Use Whisper to transcribe and get timestamps
pipe = pipeline("automatic-speech-recognition",
                model="openai/whisper-small")

audio, sr = librosa.load("audio.wav", sr=16000)

# Whisper gives word-level timestamps
result = pipe(audio, return_timestamps=True)

# result = {
#     'text': 'hello world',
#     'chunks': [
#         {'timestamp': (0.0, 0.5), 'text': 'hello'},
#         {'timestamp': (0.5, 1.0), 'text': 'world'},
#     ]
# }

# 2. Convert word timestamps to frame indices (16kHz = 160 samples per frame)
hop_length = 160
frame_per_sec = sr / hop_length  # = 100 frames/sec

for chunk in result['chunks']:
    start_frame = int(chunk['timestamp'][0] * frame_per_sec)
    end_frame = int(chunk['timestamp'][1] * frame_per_sec)
    text = chunk['text']
    print(f"Frames {start_frame}-{end_frame}: {text}")

# 3. For phone-level alignment, use g2p + DTW
# (See below)
```

### Better Approach: Phone-Level with DTW

```python
import librosa
import numpy as np
from dtaidistance import dtw
from scipy.spatial.distance import euclidean

# 1. Extract acoustic features from audio
audio, sr = librosa.load("audio.wav", sr=16000)

# Get mel-spectrogram
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
# Shape: (64, num_frames)

# 2. Convert text to phonemes (accept G2P mistakes - will be corrected by DTW)
phonemes = g2p("hello")  # ['h', 'eh', 'l', 'ow']

# 3. Get phoneme embeddings (pre-trained phone embeddings)
# Option A: Use Wav2Vec2 phone embeddings
# Option B: Use simple learnable embeddings
phone_embeddings = get_phone_embeddings(phonemes)  # (4, embedding_dim)

# 4. Use DTW to align phoneme sequence to acoustic frames
# Create cost matrix: distance between each phoneme and each frame
cost_matrix = np.zeros((len(phonemes), mel_spec.shape[1]))
for i, phone_emb in enumerate(phone_embeddings):
    for j in range(mel_spec.shape[1]):
        cost_matrix[i, j] = euclidean(phone_emb, mel_spec[:, j])

# 5. Find best path through cost matrix (DTW)
path = dtw.warping_path(cost_matrix)  # [(0, 0), (0, 1), (1, 1), ...]

# 6. Convert path to phoneme labels per frame
frame_labels = np.zeros(mel_spec.shape[1], dtype=int)
for phone_idx, frame_idx in path:
    frame_labels[frame_idx] = phone_idx

phoneme_per_frame = [phonemes[idx] for idx in frame_labels]
# Result: ['h', 'h', 'h', 'eh', 'eh', ..., 'ow', 'ow']
```

---

## Strategy 3: Custom Forced Aligner (Advanced)

If you want to implement from scratch:

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class SimplePhoneAligner(nn.Module):
    """Aligns phoneme sequence to audio frames using learned attention."""

    def __init__(self, num_phones: int, mel_dim: int):
        super().__init__()
        self.num_phones = num_phones

        # Phone embeddings
        self.phone_embed = nn.Embedding(num_phones, 128)

        # LSTM to model phone transitions
        self.phone_lstm = nn.LSTM(128, 256, batch_first=True)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(256, num_heads=4, batch_first=True)

    def forward(self, phone_ids, mel_features):
        """
        Args:
            phone_ids: (batch, num_phones) - phone sequence IDs
            mel_features: (batch, num_frames, mel_dim) - acoustic features

        Returns:
            alignment: (batch, num_frames) - phone index for each frame
        """
        # Embed phones
        phone_embed = self.phone_embed(phone_ids)  # (batch, num_phones, 128)

        # LSTM over phones
        phone_repr, _ = self.phone_lstm(phone_embed)  # (batch, num_phones, 256)

        # Attention between phones and frames
        attention_out, attn_weights = self.attention(
            mel_features,  # queries
            phone_repr,    # keys/values
            phone_repr
        )

        # Decode: each frame attends to which phone
        phone_indices = attn_weights.argmax(dim=-1)  # (batch, num_frames)

        return phone_indices, attn_weights

# Training
model = SimplePhoneAligner(num_phones=50, mel_dim=64)

# Would train on parallel alignment data:
# - (text, audio) pairs with ground truth phone alignments
# - Supervised: each frame labeled with its true phoneme
```

---

## Recommended Approach: Strategy 1 + Fallback

```python
class RobustPhonemeAligner:
    """Try model's alignment first, fall back to DTW if needed."""

    def __init__(self, model_name="qwen/Qwen3-ForcedAligner"):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name)
        self.use_dtw_fallback = False

        # Check if model has alignment API
        if not hasattr(self.model, 'align'):
            print("⚠️  Model doesn't expose .align(), will use DTW fallback")
            self.use_dtw_fallback = True

    def align(self, text: str, audio_path: str, language: str = "en"):
        """Align text and audio, using best available method."""

        if not self.use_dtw_fallback:
            # Strategy 1: Use model's built-in alignment
            try:
                result = self.model.align(
                    text=text,
                    audio_path=audio_path,
                    language=language,
                    sample_rate=16000
                )
                return result
            except Exception as e:
                print(f"Model alignment failed: {e}, falling back to DTW")
                self.use_dtw_fallback = True

        # Strategy 2: Use DTW fallback
        return self._align_with_dtw(text, audio_path, language)

    def _align_with_dtw(self, text, audio_path, language):
        """Fallback: DTW-based alignment."""
        # Implementation of Strategy 2 above
        pass
```

---

## What NOT to Do

❌ **Don't rely solely on G2P**
- Errors will propagate to SAE training
- Makes phonetic features unreliable

❌ **Don't use G2P output directly**
- Even if "hello" → ['h', 'ah', 'l', 'oh'] is wrong
- Forced alignment from audio corrects it

✅ **DO use G2P as input to forced alignment**
- G2P gives rough estimate
- Forced alignment refines it using audio
- Result is high-quality

---

## Testing Your Alignment Quality

```python
import json
import numpy as np

# After alignment, verify quality
with open("frame_labels.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        label = json.loads(line)

        text = label['text']
        num_frames = label['num_frames']
        phoneme_indices = label['phoneme_indices']

        # Check: number of unique phonemes vs. expected
        unique_phonemes = len(set(phoneme_indices))
        avg_frames_per_phoneme = num_frames / unique_phonemes

        print(f"{text}:")
        print(f"  Frames: {num_frames}")
        print(f"  Unique phonemes: {unique_phonemes}")
        print(f"  Avg frames/phoneme: {avg_frames_per_phoneme:.1f}")

        # Should be roughly:
        # - 1-2 frames per phoneme minimum (too short = alignment error)
        # - 5-50 frames per phoneme typical
        # - 100+ frames per phoneme = text too short or misaligned

        if avg_frames_per_phoneme < 1:
            print(f"  ⚠️  WARNING: Too few frames per phoneme")
        if avg_frames_per_phoneme > 100:
            print(f"  ⚠️  WARNING: Too many frames per phoneme")
```

---

## Action Items

### Immediate (Before Full Capture)

1. **Verify what Qwen3-ForcedAligner actually exposes**
   ```bash
   python scripts/inspect_aligner_api.py --device cpu
   ```
   Look for: `.align()`, `.get_phoneme_alignment()`, `.get_boundaries()`

2. **If it has alignment API, use it directly**
   - Don't use G2P at all
   - Model's alignment > any fallback

3. **If it doesn't, use DTW instead of G2P**
   ```bash
   pip install dtaidistance --break-system-packages
   ```

### For Now

Replace the G2P fallback with **DTW-based forced alignment** if model doesn't expose phoneme conversion.

---

## Bottom Line

| Approach | Quality | Audio Needed? | Recommendation |
|----------|---------|---------------|-----------------|
| G2P only | 85-92% | No | ❌ Too many errors |
| Model's align() | 95%+ | Yes | ✅ **Best** |
| G2P + DTW | 92-97% | Yes | ✅ **Good fallback** |
| DTW only | 90-95% | Yes | ✅ If no G2P |

**Recommendation:** Use model's alignment API if available, otherwise use DTW, **avoid pure G2P**.

---

## Implementation Priority

1. **First:** Verify `python scripts/inspect_aligner_api.py` output
2. **If model has `.align()`:** Use it (remove G2P entirely)
3. **If model doesn't:** Implement DTW fallback instead of G2P
4. **Result:** Accurate phone-level alignments from audio

This ensures **ground truth from audio**, not unreliable G2P predictions.
