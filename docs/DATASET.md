# Dataset: data/out.jsonl

Overview
- Single-file JSONL where each line is a training example (UTF-8).
- File path: [data/out.jsonl](data/out.jsonl)

Schema
- `text` : string — transcript / prompt for the example.
- `lang` : string — language code. Supported values:
  - `"en"` — English
  - `"zh"` — Mandarin Chinese (普通话)
  - `"yue"` — Cantonese (粵語)
- `speech_token` : list[int] — discrete speech token sequence (CosyVoice2-style token ids).
- `codec` : list[list[int]] — codec codebook indices per audio frame. Each inner list is a vector of length ~16 (16 codebooks), e.g. `[[c0_t0,...,c15_t0], [c0_t1,...,c15_t1], ...]`.

Notes & interpretation
- `speech_token` is the model-side tokenization used by some TTS models; use it when you want to feed text→speech-token supervision or to align token-level activations.
- `codec` is a multi-codebook discrete representation (Qwen3-TTS style). Each timestep/frame contains one integer per codebook. Decoding/codec-specific vocoder logic is required to convert `codec` back to waveform.
- Lengths: `len(codec)` is usually proportional to audio frames; `len(speech_token)` is the speech-token sequence length. They are not always equal — align using timestamps or model tokenizer where available.

## Loading Your Dataset

The project's `CustomDataset` class automatically handles your JSONL format with language codes:

```python
from src.data.dataset_prep import CustomDataset

# Load your dataset (automatically handles "lang" column with "en", "zh", "yue" codes)
dataset = CustomDataset("data/out.jsonl")

# Get language distribution
print(dataset.get_language_distribution())
# Output: {'English': 1000, 'Mandarin': 800, 'Cantonese': 700}

# Filter by language
english_pairs = dataset.filter_by_language("English")
mandarin_pairs = dataset.filter_by_language("Mandarin")
cantonese_pairs = dataset.filter_by_language("Cantonese")

# Iterate in batches
for batch in dataset.iterator(batch_size=32):
    # batch is a list of TextSpeechPair objects
    pass
```

## Recommended Preprocessing for SAE Training

1. Stream-read the JSONL to avoid memory spikes (the `CustomDataset` class handles this).
2. Choose a hook point in the target TTS model (recommended: early residual stream / MLP outputs). See [src/hooks/activation_hook.py](../src/hooks/activation_hook.py) for capture helpers.
3. For each example:
   - Load `text` and `speech_token` to run the model and produce activations.
   - If you need waveform-level supervision, use `codec` with your codec/vocoder to synthesize audio for alignment checks.
   - Save activations in streaming shards (e.g., compressed `.npz` or `.npy` per shard) — see [tools/tts_precompute_activations.py](../tools/tts_precompute_activations.py) for the project precompute workflow.
4. Quantize or cast activations consistently (FP16 recommended for storage/training on GPUs; validate numeric ranges before training).
5. (Optional) Analyze per-language statistics to ensure balanced representation during SAE training.

Practical tips
- Large file: process with a buffered reader and periodic flushes to disk. The capture pipeline in `src/hooks/activation_hook.py` supports streaming flush.
- When matching SAE inputs to activations, store per-activation metadata: example id, token index, layer index, timestamp/frame index.
- Keep a small pilot subset (100–1k examples) to validate end-to-end capture → precompute → MSAE training before scaling.

Example minimal reader (Python)
```py
import json
with open('data/out.jsonl','r',encoding='utf-8') as f:
    for i,line in enumerate(f):
        obj = json.loads(line)
        text = obj.get('text')
        speech_tokens = obj.get('speech_token')
        codec = obj.get('codec')
        if i>1000: break
```

See also
- Capture helpers: [src/hooks/activation_hook.py](src/hooks/activation_hook.py)
- Precompute script: [tools/tts_precompute_activations.py](tools/tts_precompute_activations.py)
- SAE trainer wrapper: [tools/run_msae_train.py](tools/run_msae_train.py)

If you want, I can commit the `requirements-*.txt` files and run a small pilot precompute on 100 examples next.
