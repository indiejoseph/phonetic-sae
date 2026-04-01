# Phoneme Preparation: When API Discovery Fails

**Problem:** Qwen3-ForcedAligner inspection scripts fail or don't reveal `text_to_phonemes` function.

**Solution:** Use external G2P (Grapheme-to-Phoneme) converters + audio-only alignment.

---

## Quick Decision Tree

```
Does Qwen3-ForcedAligner have text_to_phonemes()?
├─ YES → Use model's internal phoneme conversion (already implemented)
└─ NO → Use external G2P converters (this guide)
    ├─ English → g2p_en library
    ├─ Mandarin → pypinyin library
    └─ Cantonese → jyutping library (limited support)
```

---

## Option 1: External G2P for English (Recommended for EN)

### Install
```bash
pip install g2p_en --break-system-packages
```

### Usage
```python
from g2p_en.g2p import G2p

g2p = G2p()
phonemes = g2p("hello")
print(phonemes)  # ['H', 'AH0', 'L', 'OW1']
```

### Integrate into Phoneme Alignment
```python
from g2p_en.g2p import G2p
from src.alignment import QwenForcedAligner

class TextToPhonemePreprocessor:
    def __init__(self, language: str):
        if language == "en":
            self.g2p = G2p()
            self.language = "en"
        else:
            raise NotImplementedError(f"Language {language} not supported")

    def text_to_phonemes(self, text: str) -> list:
        """Convert text to phonemes using g2p_en."""
        phonemes = self.g2p(text)
        # Clean up: remove stress markers (0, 1, 2)
        return [p.rstrip('0123456789').lower() for p in phonemes]

# Usage
preprocessor = TextToPhonemePreprocessor("en")
phonemes = preprocessor.text_to_phonemes("hello world")
print(phonemes)  # ['h', 'ah', 'l', 'ow', 'w', 'er', 'l', 'd']
```

---

## Option 2: Pinyin for Mandarin Chinese

### Install
```bash
pip install pypinyin --break-system-packages
```

### Usage
```python
from pypinyin import pinyin, TONE, NORMAL

text = "你好"
phonemes = pinyin(text, style=TONE)
print(phonemes)  # [['nǐ'], ['hǎo']]

# Without tones
phonemes_no_tone = pinyin(text, style=NORMAL)
print(phonemes_no_tone)  # [['ni'], ['hao']]
```

### Integrate
```python
from pypinyin import pinyin, NORMAL

class TextToPhonemePreprocessor:
    def __init__(self, language: str):
        if language == "zh":
            self.language = "zh"
        else:
            raise NotImplementedError(f"Language {language} not supported")

    def text_to_phonemes(self, text: str) -> list:
        """Convert Mandarin text to pinyin phonemes."""
        phonemes = pinyin(text, style=NORMAL)
        # Flatten list of lists
        return [p[0].lower() for p in phonemes if p]

# Usage
preprocessor = TextToPhonemePreprocessor("zh")
phonemes = preprocessor.text_to_phonemes("你好世界")
print(phonemes)  # ['ni', 'hao', 'shi', 'jie']
```

---

## Option 3: Jyutping for Cantonese (Limited Support)

### Install
```bash
pip install ToJyutping --break-system-packages
```

### Usage
```python
import ToJyutping

# Get list of (character, jyutping) pairs
ToJyutping.get_jyutping_list('咁啱老世要求佢等陣要開會,剩低嘅嘢我會搞掂㗎喇。')
# => [("咁", "gam3"), ("啱", "ngaam1"), ... , ("。", None)]

# Get inline annotated string
ToJyutping.get_jyutping('咁啱老世要求佢等陣要開會,剩低嘅嘢我會搞掂㗎喇。')
# => '咁(gam3)啱(ngaam1)...喇(laa3)。'

# Get space-separated jyutping text
ToJyutping.get_jyutping_text('咁啱老世要求佢等陣要開會,剩低嘅嘢我會搞掂㗎喇。')
# => 'gam3 ngaam1 lou5 sai3 jiu1 ... gaa3 laa3.'

# Get candidate lists per character
ToJyutping.get_jyutping_candidates('咁啱老世要求佢等陣要開會,剩低嘅嘢我會搞掂㗎喇。')
# => [("咁", ["gam3","gam2","gam1",...]), ...]

# IPA helpers are also available:
ToJyutping.get_ipa_list('咁啱老世要求佢等陣要開會,剩低嘅嘢我會搞掂㗎喇。')
ToJyutping.get_ipa('咁啱老世要求佢等陣要開會,剩低嘅嘢我會搞掂㗎喇。')
ToJyutping.get_ipa_text('咁啱老世要求佢等陣要開會,剩低嘅嘢我會搞掂㗎喇。')
ToJyutping.get_ipa_candidates('咁啱老世要求佢等陣要開會,剩低嘅嘢我會搞掂㗎喇。')
```

### Note on Cantonese
- **ToJyutping** has excellent character coverage for Cantonese
- Better than basic jyutping library
- Returns phonemes with tone numbers (1-9)
- Recommended for production use

---

## Full Fallback Implementation

Create `src/alignment/phoneme_converter.py`:

```python
"""Fallback text-to-phoneme conversion when model API doesn't expose it."""

from typing import List
from abc import ABC, abstractmethod


class PhonemeConverter(ABC):
    """Base class for text-to-phoneme conversion."""

    @abstractmethod
    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert text to list of phonemes."""
        pass


class G2pEnConverter(PhonemeConverter):
    """English text-to-phonemes using g2p_en."""

    def __init__(self):
        try:
            from g2p_en.g2p import G2p
            self.g2p = G2p()
        except ImportError:
            raise ImportError(
                "g2p_en not installed. Install with:\n"
                "pip install g2p_en --break-system-packages"
            )

    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert English text to ARPAbet phonemes."""
        phonemes = self.g2p(text)
        # Remove stress markers and lowercase
        return [p.rstrip('0123456789').lower() for p in phonemes if p]


class PinYinConverter(PhonemeConverter):
    """Mandarin text-to-phonemes using pypinyin."""

    def __init__(self):
        try:
            from pypinyin import pinyin, NORMAL
            self.pinyin_fn = pinyin
            self.NORMAL = NORMAL
        except ImportError:
            raise ImportError(
                "pypinyin not installed. Install with:\n"
                "pip install pypinyin --break-system-packages"
            )

    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert Mandarin text to pinyin phonemes."""
        phonemes = self.pinyin_fn(text, style=self.NORMAL)
        return [p[0].lower() for p in phonemes if p]


class ToJyutpingConverter(PhonemeConverter):
    """Cantonese text-to-phonemes using ToJyutping."""

    def __init__(self):
        try:
            import ToJyutping
            self.ToJyutping = ToJyutping
        except ImportError:
            raise ImportError(
                "ToJyutping not installed. Install with:\n"
                "pip install ToJyutping --break-system-packages"
            )

    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert Cantonese text to jyutping phonemes."""
        phonemes = []
        # Get (character, jyutping) pairs
        jyutping_list = self.ToJyutping.get_jyutping_list(text)
        for char, jyutping_val in jyutping_list:
            if jyutping_val is not None:
                # jyutping_val is already a string like 'nei5', 'hou2'
                phonemes.append(jyutping_val.lower())
        return phonemes


def get_converter(language: str) -> PhonemeConverter:
    """Get appropriate converter for language."""
    converters = {
        "en": G2pEnConverter,
        "english": G2pEnConverter,
        "zh": PinYinConverter,
        "mandarin": PinYinConverter,
        "yue": ToJyutpingConverter,
        "cantonese": ToJyutpingConverter,
    }

    converter_class = converters.get(language.lower())
    if not converter_class:
        raise ValueError(f"Unsupported language: {language}")

    return converter_class()


# Example usage
if __name__ == "__main__":
    # English
    en_converter = get_converter("en")
    print("English 'hello':", en_converter.text_to_phonemes("hello"))

    # Mandarin
    zh_converter = get_converter("zh")
    print("Mandarin '你好':", zh_converter.text_to_phonemes("你好"))

    # Cantonese
    yue_converter = get_converter("yue")
    print("Cantonese '你好':", yue_converter.text_to_phonemes("你好"))
```

---

## Modified Alignment Pipeline

If model doesn't expose `text_to_phonemes`, modify `capture_with_alignment.py`:

```python
from src.alignment import QwenForcedAligner
from src.alignment.phoneme_converter import get_converter

class AlignmentWithFallback:
    def __init__(self, model, language: str):
        self.aligner = QwenForcedAligner(model)
        self.language = language

        # Try to use model's built-in phoneme conversion
        # If not available, use external converter
        try:
            self.text_to_phonemes = self.aligner.text_to_phonemes
        except AttributeError:
            print(f"⚠️  Model doesn't expose text_to_phonemes, using external converter")
            self.converter = get_converter(language)
            self.text_to_phonemes = self.converter.text_to_phonemes

    def align(self, text: str, audio: np.ndarray, sample_rate: int):
        """Align text and audio, handling phonemes internally."""
        # Option 1: If model accepts (text, audio, language)
        try:
            return self.aligner.align(text, audio, self.language, sample_rate)
        except TypeError:
            pass

        # Option 2: If model needs explicit phonemes
        phonemes = self.text_to_phonemes(text)
        return self.aligner.align(
            phonemes=phonemes,
            audio=audio,
            language=self.language,
            sample_rate=sample_rate
        )
```

---

## Testing Fallback Converters

```python
from src.alignment.phoneme_converter import get_converter

# Test each language
test_cases = {
    "en": "hello world",
    "zh": "你好世界",
    "yue": "你好",
}

for lang, text in test_cases.items():
    converter = get_converter(lang)
    phonemes = converter.text_to_phonemes(text)
    print(f"{lang}: {text} → {phonemes}")
```

**Expected output:**
```
en: hello world → ['h', 'ah', 'l', 'ow', 'w', 'er', 'l', 'd']
zh: 你好世界 → ['ni', 'hao', 'shi', 'jie']
yue: 你好 → ['nei', 'hou']  # approximate
```

---

## When to Use Each Option

### Option 1: Use Model's Built-in
- ✅ If `python scripts/inspect_aligner_api.py` shows `text_to_phonemes` method
- ✅ If model accepts (text, audio, language) directly
- ⏱️ Fastest, no extra dependencies

### Option 2: External G2P Converter
- ✅ If model inspection fails or times out
- ✅ If model doesn't expose `text_to_phonemes`
- ✅ If you need explicit phoneme control
- ⏱️ Adds 1-2 dependencies per language

### Option 3: Hybrid (Recommended)
- ✅ Try model's built-in first
- ✅ Fall back to external converter if that fails
- ✅ Gives best of both worlds
- ⏱️ Most robust

---

## Troubleshooting

### Issue: "g2p_en not installed"
```bash
pip install g2p_en --break-system-packages
```

### Issue: "pypinyin not installed"
```bash
pip install pypinyin --break-system-packages
```

### Issue: "Phonemes don't match expected inventory"
- Extract actual phoneme set from converter output
- Compare with `phoneme_inventory.json` from `inspect_aligner.py`
- May need to map converter output to model's expected format

### Issue: "Cantonese phonemes look wrong"
- Jyutping support is limited
- Consider using en/zh first, skip yue until better support available
- Or implement custom mapping

---

## Integration into Existing Code

Update `src/alignment/forced_aligner.py`:

```python
from src.alignment.phoneme_converter import get_converter

class QwenForcedAligner:
    def __init__(self, model, processor, language: str):
        self.model = model
        self.processor = processor
        self.language = language

        # Try model's built-in, fallback to external
        if hasattr(model, 'text_to_phonemes'):
            self.text_to_phonemes = model.text_to_phonemes
        else:
            converter = get_converter(language)
            self.text_to_phonemes = converter.text_to_phonemes

    def align(self, text: str, audio: np.ndarray, sample_rate: int = 16000):
        """Align text and audio with fallback phoneme conversion."""
        # Rest of alignment code...
        phonemes = self.text_to_phonemes(text)
        # ... continue with alignment
```

---

## What to Do Right Now

### 1. Verify What Model Actually Has
```bash
python scripts/inspect_aligner_api.py --device cpu > inspection.txt
cat inspection.txt | grep -i "phoneme\|text"
```

### 2. Install Fallback Converters
```bash
pip install g2p_en pypinyin jyutping --break-system-packages
```

### 3. Test Converters
```python
from src.alignment.phoneme_converter import get_converter

# Test English
en_conv = get_converter("en")
print(en_conv.text_to_phonemes("test"))

# Test Mandarin
zh_conv = get_converter("zh")
print(zh_conv.text_to_phonemes("测试"))
```

### 4. Use Hybrid Approach
- Let model handle phoneme conversion if possible
- Fall back to external converters if needed
- Both approaches produce compatible phoneme sets

---

## Summary

If Qwen3-ForcedAligner doesn't expose `text_to_phonemes`:

1. **Use external G2P converters** ← This is robust
2. **Supported languages:**
   - English: g2p_en (excellent)
   - Mandarin: pypinyin (excellent)
   - Cantonese: jyutping (basic)
3. **Implementation:** Create `phoneme_converter.py` with fallback
4. **Integration:** Update `forced_aligner.py` to try model first, then fallback

This ensures phoneme preparation works **regardless of what Qwen3-ForcedAligner actually exposes**. 🎯
