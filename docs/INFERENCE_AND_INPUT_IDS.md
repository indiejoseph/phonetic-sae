# Inference Details & Input IDs: Qwen3-TTS vs CosyVoice2

Detailed comparison of inference methods, input token handling, and language-specific configurations for activation mining.

---

## 1. Qwen3-TTS-0.6B Inference

### Target Method: `generate_voice_clone()`

The Base model uses voice cloning inference with speaker embedding and optional In-Context Learning (ICL).

#### Method Signature

```python
def generate_voice_clone(
    self,
    text: Union[str, List[str]],
    language: Union[str, List[str]] = None,
    ref_audio: Optional[Union[AudioLike, List[AudioLike]]] = None,
    ref_text: Optional[Union[str, List[Optional[str]]]] = None,
    x_vector_only_mode: Union[bool, List[bool]] = False,
    voice_clone_prompt: Optional[Union[Dict[str, Any], List[VoiceClonePromptItem]]] = None,
    non_streaming_mode: bool = False,
    **kwargs,
) -> Tuple[List[np.ndarray], int]:
```

#### Input Parameters

| Parameter | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| **text** | str or List[str] | Synthesis text | "Good one. Okay, fine..." |
| **language** | str or List[str] | Language(s) | "Auto", "English", "Chinese" |
| **ref_audio** | AudioLike or List | Reference audio for voice clone | Path, URL, numpy array |
| **ref_text** | str or List[str] | Reference text (for ICL mode) | "Okay. Yeah. I resent you..." |
| **x_vector_only_mode** | bool or List[bool] | Use only speaker embedding | True or False |
| **voice_clone_prompt** | VoiceClonePromptItem | Pre-built prompt object | From `create_voice_clone_prompt()` |
| **non_streaming_mode** | bool | Non-streaming text input | False (default) |

#### Language Support

Qwen3-TTS supports 10 languages, specified as language strings that are mapped to codec IDs:

| Language String | Codec ID | Note |
| :--- | :--- | :--- |
| **English** | 2050 | Full English support |
| **Chinese** | 2055 | Mandarin Chinese |
| **Spanish** | 2054 | Latin American Spanish |
| **German** | 2053 | Standard German |
| **Portuguese** | 2071 | Brazilian Portuguese |
| **Italian** | 2070 | Italian |
| **Japanese** | 2058 | Japanese |
| **Korean** | 2064 | Korean |
| **French** | 2061 | French |
| **Russian** | 2069 | Russian |
| **Auto** | Detected | Auto-detect from model |

**Important:** The language parameter is internally converted to a codec ID (2050-2071) that is prepended to the input token sequence.

#### Voice Cloning Modes

**1. X-Vector Only Mode** (`x_vector_only_mode=True`)
- Uses only speaker embedding (x-vector)
- No text-based context from reference
- Faster inference, simpler input
- Good for speaker identity transfer only

**2. ICL Mode** (`x_vector_only_mode=False`)
- In-Context Learning mode (default)
- Uses reference text AND reference speech codes
- Provides both speaker identity and phonetic context
- Better quality, more context-aware

#### Voice Clone Prompt Structure

```python
@dataclass
class VoiceClonePromptItem:
    ref_code: Optional[torch.Tensor]      # (T, Q) speech tokens from reference
    ref_spk_embedding: torch.Tensor       # (D,) speaker embedding
    x_vector_only_mode: bool              # Mode flag
    icl_mode: bool                        # ICL enabled
    ref_text: Optional[str] = None        # Reference text for display
```

### Input Token Construction (Simplified Flow)

```
1. Text Tokenization:
   - Text → Qwen tokenizer → token_ids (0-151935)

2. Language ID Insertion:
   - Prepend language codec_id (2050-2071) or skip for "Auto"

3. Voice Clone Prompt:
   - Reference audio → speaker embedding (x-vector)
   - Reference audio → speech codes (codec tokens 2048-3071)
   - Reference text → tokenized for ICL context

4. Input to Talker:
   - [language_id] + [text_tokens] + [prompt_tokens]
   - Shape: (batch, seq_len) with dtype=long

5. Code Predictor Output:
   - 15 additional codec tokens (total 16 codebooks)
```

### Special Token IDs in Input

| Token Name | ID | Usage |
| :--- | :--- | :--- |
| **im_start** | 151644 | Marks instruction start |
| **im_end** | 151645 | Marks instruction end |
| **tts_bos** | 151672 | TTS sequence begin |
| **tts_eos** | 151673 | TTS sequence end |
| **tts_pad** | 151671 | Padding |
| **codec_bos** | 2149 | Codec sequence begin |
| **codec_eos** | 2150 | Codec sequence end |
| **codec_pad** | 2148 | Codec padding |
| **codec_language_ids** | 2050-2071 | Language markers (prepended to text) |

### Generation Hyperparameters

Default sampling configuration (from `generation_config.json`):

```python
common_gen_kwargs = dict(
    max_new_tokens=2048,                # Max codec tokens to generate
    do_sample=True,                     # Use sampling (not greedy)
    temperature=0.9,                    # Talker temperature
    top_p=1.0,                          # Nucleus sampling disabled
    top_k=50,                           # Top-K sampling
    repetition_penalty=1.05,            # Slight penalty for repeats
    subtalker_dosample=True,            # Code Predictor sampling
    subtalker_temperature=0.9,          # Code Predictor temperature
    subtalker_top_k=50,                 # Code Predictor top-K
    subtalker_top_p=1.0,                # Code Predictor nucleus disabled
)
```

### Activation Mining Implications for Qwen3-TTS

**Input Characteristics:**
- Text tokens: 0-151,935
- Language codec IDs: 2050-2071 (prepended to input)
- Speech codes (reference): 2048-3071
- Special tokens: 151,644-151,673

**Expected Activation Patterns:**
- **Layer 1-3:** Heavy language ID influence, early phoneme processing
- **Layer 4-7:** Phoneme-level information, minimal language ID effect
- **Layer 8+:** Coarticulation, prosodic features, speaker identity

**Key for Feature Discovery:**
- Language codec IDs appear at sequence position 0 in all inputs
- Can probe "Language Sensitivity" by comparing activations across language_ids
- Reference codes in ICL mode provide phonetically grounded labels for supervision

---

## 2. CosyVoice2-0.5B Inference

### Target Method: `inference_zero_shot()`

CosyVoice2 uses zero-shot inference with speaker prompt and text normalization.

#### Method Signature

```python
def inference_zero_shot(
    self,
    tts_text,
    prompt_text,
    prompt_wav,
    zero_shot_spk_id='',
    stream=False,
    speed=1.0,
    text_frontend=True
):
```

#### Input Parameters

| Parameter | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| **tts_text** | str or Generator | Text to synthesize | '收到好友从远方...' |
| **prompt_text** | str | Speaker prompt text | '希望你以后能够...' |
| **prompt_wav** | str | Speaker audio path | './asset/prompt.wav' |
| **zero_shot_spk_id** | str | Saved speaker ID | 'my_speaker' or '' |
| **stream** | bool | Streaming mode | False (default) |
| **speed** | float | Speaking rate multiplier | 1.0 (default) |
| **text_frontend** | bool | Apply text normalization | True (default) |

#### Language Support

CosyVoice2 uses **implicit language detection** based on text characters. No explicit language parameter.

**Supported Languages:**
- **Chinese** (Simplified/Traditional) — Default
- **English** — Detected from Roman alphabet
- **Japanese** — Detected from Hiragana/Katakana
- **Cantonese (Yue)** — Detected from Cantonese characters
- **Korean** — Detected from Hangul

**Language Markers in Prompts:**
Used in `inference_cross_lingual()` for explicit language control:

```python
'<|en|>English text here'      # English
'<|zh|>中文文本'                 # Chinese
'<|ja|>日本語テキスト'            # Japanese
'<|yue|>粵語文本'                 # Cantonese
'<|ko|>한국어 텍스트'             # Korean
```

### Model Input Construction

#### Zero-Shot Mode Input Dictionary

The `frontend_zero_shot()` method creates:

```python
model_input = {
    # Synthesis text tokens
    'text': tts_text_token,                           # (1, seq_len)
    'text_len': tts_text_token_len,                   # (1,)

    # Prompt tokens (speaker conditioning)
    'prompt_text': prompt_text_token,                 # (1, prompt_seq_len)
    'prompt_text_len': prompt_text_token_len,         # (1,)

    # LLM-level speech conditioning
    'llm_prompt_speech_token': speech_token,          # (1, token_len)
    'llm_prompt_speech_token_len': speech_token_len,  # (1,)

    # Flow-level speech conditioning (mel features)
    'flow_prompt_speech_token': speech_token,         # (1, token_len)
    'flow_prompt_speech_token_len': speech_token_len, # (1,)
    'prompt_speech_feat': speech_feat,                # (1, 80, feat_len)
    'prompt_speech_feat_len': speech_feat_len,        # (1,)

    # Speaker embedding
    'llm_embedding': embedding,                       # (1, 192)
    'flow_embedding': embedding,                      # (1, 192)
}
```

**Key Differences from Zero-Shot without Saved Speaker:**

If `zero_shot_spk_id` is provided (saved speaker):
```python
model_input = {
    **self.spk2info[zero_shot_spk_id],  # Pre-extracted speaker info
    'text': tts_text_token,
    'text_len': tts_text_token_len,
}
```

#### Cross-Lingual Mode Input Dictionary

For `inference_cross_lingual()`, the model_input is modified:

```python
model_input = frontend_zero_shot(...)  # Start with zero-shot
# Then remove LLM-level prompt (to enable language flexibility)
del model_input['prompt_text']
del model_input['prompt_text_len']
del model_input['llm_prompt_speech_token']
del model_input['llm_prompt_speech_token_len']

# Keep flow-level speech and embedding for acoustic consistency
# Now available in model_input:
#   'text', 'text_len'
#   'flow_prompt_speech_token', 'flow_prompt_speech_token_len'
#   'prompt_speech_feat', 'prompt_speech_feat_len'
#   'llm_embedding', 'flow_embedding'
```

### Token Vocabulary

**Text Tokens (Qwen2.5 Tokenizer):**
- Vocabulary size: ~100K+ tokens (Qwen2.5 standard)
- Handles: English, Chinese, Japanese, Korean, Cantonese

**Speech Tokens (Semantic Tokens):**
- Vocabulary size: 6,561 tokens
- Range: 0-6,560
- Extracted from LibriTTS speech via a pre-trained speech tokenizer

### Input Feature Dimensions

| Component | Shape | Description |
| :--- | :--- | :--- |
| **text_tokens** | (batch, seq_len) | Text token IDs |
| **speech_tokens** | (batch, token_len) | Semantic speech tokens |
| **speech_feat (mel)** | (batch, 80, feat_len) | Mel-spectrogram features |
| **speaker_embedding** | (batch, 192) | Speaker embedding from CampPlus |

### Text Normalization & Tokenization Flow

```
1. Input Text (raw)
   ↓
2. Text Normalization (via text_frontend)
   - Language auto-detection
   - Punctuation handling
   - Phoneme conversion (if needed)
   ↓
3. Qwen2.5 Tokenization
   - Convert to token IDs (0-100K+)
   - Handle special tokens for language tags
   ↓
4. Token Tensor
   - Shape: (batch, seq_len)
```

### Special Tokens & Control Markers

CosyVoice2 supports fine-grained control via embedded markers:

| Marker | Purpose | Example |
| :--- | :--- | :--- |
| **[laughter]** | Insert laughter | "He said [laughter] what?" |
| **[breath]** | Insert breath | "I... [breath]... think so" |
| **<strong>text</strong>** | Emphasis/stress | "He showed <strong>great</strong> courage" |
| **<|en\|>** | Language marker | "<\|en\|>Hello world" |
| **<\|endofprompt\|>** | End of instruction prompt | "Instruct: ... <\|endofprompt\|>" |

### Activation Mining Implications for CosyVoice2

**Input Characteristics:**
- Text tokens: 0-100K+ (Qwen2.5)
- Speech tokens: 0-6,560
- Special markers: [laughter], [breath], <strong>...</strong>
- Language auto-detection (no explicit IDs)

**Expected Activation Patterns:**
- **Layer 1-3:** Language detection signals, early phoneme extraction
- **Layer 4-6:** Text understanding, phonetic processing
- **Layer 7+:** Speaker and acoustic conditioning (less directly involved in phonetic task)

**Key for Feature Discovery:**
- Text tokens span much larger vocab → more polysemy expected
- Speech tokens (6,561) provide direct phonetic labels
- Speaker embedding is low-dimensional (192) but injected at both LLM and Flow stages
- No explicit language IDs in tokens → language sensitivity may be more implicit

---

## 3. Comparative Analysis for Activation Mining

### Input Token Ranges

| Component | Qwen3-TTS | CosyVoice2 |
| :--- | :--- | :--- |
| **Text Vocab** | 151,936 | ~100K+ (Qwen2.5) |
| **Speech Token Vocab** | 3,072 (codec) | 6,561 (semantic) |
| **Language IDs** | 2050-2071 (explicit) | Auto-detected (implicit) |
| **Special Tokens** | 151,644-151,673 | Control markers [laughter], <strong> |
| **Padding Token** | 2148 (codec_pad) | 0 or standard |

### Language/Phonetic Signal Representation

**Qwen3-TTS:**
- Explicit language codec IDs (2050-2071) prepended to input
- Can directly measure language sensitivity via codec ID position
- Reference codes (ICL mode) provide phonetic grounding
- 10-language support with dedicated codec IDs

**CosyVoice2:**
- Implicit language detection from text characters
- Language explicit only in `inference_cross_lingual()` via markers
- Speech tokens (6,561) provide richer phonetic representation than codecs (3,072)
- Auto-detection may introduce language-mixing artifacts in mixed-language text

### Activation Hook Timing

**Qwen3-TTS (Talker LLM):**
```
Text Tokens → [Text Embedding]
Language ID → [Codec Embedding]
           ↓
           [Layer 1] ← HOOK HERE (early phoneme processing)
           [Layer 2-7] ← HOOK HERE (phonetic refinement)
           ↓
           Output → Code Predictor
```

**CosyVoice2 (Qwen2.5 LLM):**
```
Text Tokens → [Text Embedding]
           ↓
           [Layer 1] ← HOOK HERE (language detection, phoneme extraction)
           [Layer 2-6] ← HOOK HERE (phonetic processing)
           ↓
           Output → Speech Tokens
           [Speaker Embedding] → Flow Matching
```

### Key Differences for Activation Mining

| Aspect | Qwen3-TTS | CosyVoice2 |
| :--- | :--- | :--- |
| **Language Signal** | Explicit codec IDs | Implicit text characters |
| **Phonetic Labels** | Codec tokens (reference) | Speech tokens (semantic) |
| **Vocab Size** | 151,936 (text) + 3,072 (codec) | 100K+ (text) + 6,561 (speech) |
| **Target Layers** | 1-7 of 28 | 1-6 of 24 |
| **Hidden Size** | 1024 | 896 |
| **Language Diversity** | 10 explicit languages | Multi-lingual implicit |
| **Speaker Conditioning** | Post-Talker (in prompt) | Dual-stage (LLM + Flow) |

---

## 4. Recommended Inference Protocol for Activation Mining

### For Qwen3-TTS Base Model

```python
# Single language, voice clone mode
tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

# Prepare voice clone prompt
prompt_items = tts.create_voice_clone_prompt(
    ref_audio="path/to/reference.wav",
    ref_text="Reference speaker text...",
    x_vector_only_mode=False,  # Use ICL mode for richer phonetic context
)

# Generate with explicit language
for language in ["English", "Chinese"]:
    wavs, sr = tts.generate_voice_clone(
        text="Synthesis text...",
        language=language,
        voice_clone_prompt=prompt_items,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.9,
    )
    # Activations captured during this forward pass
```

**Activation Capture Points:**
- Capture at `model.talker.layers[i].mlp` for i ∈ [1, 7]
- Token sequence includes: [language_id] + [text_tokens] + [prompt_tokens]
- Store with language_id label for later analysis

### For CosyVoice2-0.5B

```python
# Zero-shot mode (default)
cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B')

# Zero-shot inference
for tts_text, prompt_text, prompt_wav in dataset:
    for output in cosyvoice.inference_zero_shot(
        tts_text=tts_text,
        prompt_text=prompt_text,
        prompt_wav=prompt_wav,
        text_frontend=True,  # Enable text normalization
    ):
        # Activations captured during this forward pass
        pass

# Cross-lingual inference for language diversity
for output in cosyvoice.inference_cross_lingual(
    tts_text='<|en|>English text here',
    prompt_wav=prompt_wav,
):
    pass
```

**Activation Capture Points:**
- Capture at `model.llm.layers[i].mlp` for i ∈ [1, 6]
- Token sequence: [text_tokens] (language implicit in characters)
- Store text for language detection post-hoc

---

## 5. Summary Table

| Property | Qwen3-TTS | CosyVoice2 |
| :--- | :--- | :--- |
| **Inference Method** | `generate_voice_clone()` | `inference_zero_shot()` |
| **Language Control** | Explicit parameter | Implicit/auto-detect |
| **Language Marker IDs** | 2050-2071 (codec) | None (text-based) |
| **Phonetic Grounding** | Reference codes (ICL) | Speech tokens (semantic) |
| **Speaker Conditioning** | Speaker embedding | Dual embedding (LLM + Flow) |
| **Target Layers** | 1-7 (of 28) | 1-6 (of 24) |
| **Hook Point** | `talker.layers[i].mlp` | `llm.layers[i].mlp` |
| **Expected Phonetic Features** | High in layers 1-7 | High in layers 1-6 |

---

End of Inference and Input IDs Documentation
