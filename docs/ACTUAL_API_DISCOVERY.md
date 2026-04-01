# Discovering the Actual Qwen3-ForcedAligner API

**TL;DR:** The pseudocode I provided is illustrative. The real API may be different. Use these methods to discover it.

---

## Why This Matters

Your concern about `text_to_phonemes` is **valid**. I made assumptions based on typical patterns, but:

- ❌ I haven't verified the exact API
- ❌ The `text_to_phonemes` function may not exist as written
- ❌ The input/output format may differ
- ❌ Text-to-phoneme conversion might be built-in or external

This guide helps you find the **actual** API.

---

## Method 1: Run the API Inspector Script

**Quick inspection of model methods and attributes:**

```bash
python scripts/inspect_aligner_api.py
```

**Output will show:**
- All callable methods on the model
- Processor methods and APIs
- Model configuration
- Actual input/output examples
- Saves to `aligner_api_inspection.json`

---

## Method 2: Direct Inspection in Python

```python
from transformers import AutoModel, AutoProcessor

# Load model
model = AutoModel.from_pretrained(
    "Qwen/Qwen3-ForcedAligner-0.6B",
    device_map="cuda",
    trust_remote_code=True,
)

# See ALL attributes and methods
print("All model attributes:")
for attr in dir(model):
    if not attr.startswith("_"):
        print(f"  {attr}")

# Check what the model actually exposes
print("\nCallable methods:")
for attr in dir(model):
    if not attr.startswith("_"):
        obj = getattr(model, attr)
        if callable(obj):
            print(f"  {attr}")

# Check config
print("\nModel config:")
if hasattr(model, "config"):
    print(model.config)

# Try the actual forward pass
import torch
try:
    output = model(torch.randn(1, 16000))  # Try with audio only
    print(f"\nForward pass successful!")
    print(f"Output type: {type(output)}")
    print(f"Output: {output}")
except Exception as e:
    print(f"\nForward pass failed: {e}")
    print("Try different input formats...")
```

---

## Method 3: Check the HuggingFace Model Card

**Go to:** https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B

Look for:
- **Usage examples** (usually in README)
- **Input/output format**
- **Available methods**
- **Required preprocessing**
- **Known issues or gotchas**

---

## Method 4: Look at Our Current Implementation

**Our `src/alignment/forced_aligner.py` already uses the real API:**

```python
# This is the ACTUAL working code
result = self.model.align(
    text=text,
    audio=audio_tensor,
    language=self.language,
    sample_rate=sample_rate,
)
```

**Key observations:**
- The model has an `.align()` method (not separate encoder/decoder)
- It takes `text`, `audio`, `language`, `sample_rate`
- It returns a result object with alignment info

**So the actual API is simpler than the pseudocode!**

---

## What We Need to Verify

### ❓ Question 1: How does text-to-phoneme conversion work?

**Possibilities:**
1. Built into the model (most likely)
2. Via processor (if it has g2p)
3. External g2p library needed

**How to test:**
```python
# Option A: Check if model has phoneme-related methods
if hasattr(model, "text_to_phonemes"):
    print("✅ Model has text_to_phonemes")

# Option B: Check processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B")
print(dir(processor))

# Option C: Check config
if hasattr(model.config, "g2p_model"):
    print(f"✅ Uses g2p model: {model.config.g2p_model}")
```

### ❓ Question 2: What are the input requirements?

**What we know:**
- Requires: audio (16kHz)
- Requires: text
- Optional: language code

**What we need to verify:**
- Audio tensor shape (mono or stereo?)
- Audio dtype (float32 or float16?)
- Text encoding (raw string or token IDs?)

### ❓ Question 3: What does the output contain?

**What we assume:**
- Phoneme sequence
- Frame boundaries
- Frame-to-phoneme mapping

**What we need to verify:**
- Exact output format
- How to access each component
- Are there additional outputs?

---

## Quick Test Script

**Run this to verify the actual API:**

```python
import torch
from transformers import AutoModel, AutoProcessor

print("Loading model...")
model = AutoModel.from_pretrained(
    "Qwen/Qwen3-ForcedAligner-0.6B",
    device_map="cuda",
    trust_remote_code=True,
)

# Test what the model can actually do
print("\n" + "="*70)
print("TESTING ACTUAL MODEL API")
print("="*70)

# 1. Test with audio + text
print("\n1. Testing with audio + text...")
try:
    audio = torch.randn(1, 16000)  # 1 second at 16kHz
    result = model.align(
        text="hello world",
        audio=audio,
        language="en",
        sample_rate=16000,
    )
    print(f"   ✅ Success!")
    print(f"   Output type: {type(result)}")
    print(f"   Output keys: {result.keys() if hasattr(result, 'keys') else 'not a dict'}")
    if hasattr(result, 'keys'):
        for key in result.keys():
            val = result[key]
            print(f"     {key}: {type(val).__name__}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# 2. Test processor
print("\n2. Testing processor...")
try:
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B")
    print(f"   ✅ Processor loaded")
    # Look for phoneme-related methods
    phoneme_methods = [m for m in dir(processor) if "phoneme" in m.lower()]
    if phoneme_methods:
        print(f"   ✅ Phoneme methods found: {phoneme_methods}")
    else:
        print(f"   ⚠️ No phoneme methods in processor")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# 3. List all methods
print("\n3. All model methods (first 20)...")
methods = [m for m in dir(model) if not m.startswith("_") and callable(getattr(model, m))]
for method in methods[:20]:
    print(f"   • {method}")

print("\nDone! Check above for actual API.")
```

---

## What to Do With Your Findings

1. **Document the actual API** in `docs/ACTUAL_API_DISCOVERY.md` (update this file)
2. **Update our implementation** if the API differs from what we have
3. **Update the pseudocode** in `QWEN3_FORCEDALIGNER_INFERENCE.md` to match reality
4. **Share findings** so we can improve the documentation

---

## If the API is Different

**Example: If the model doesn't have `.align()` but has `.forward()`:**

```python
# Update src/alignment/forced_aligner.py
result = self.model(
    audio=audio_tensor,
    input_ids=phoneme_ids,  # or different format
)
```

---

## Summary

| Step | Action | Command |
|------|--------|---------|
| 1 | Run API inspector | `python scripts/inspect_aligner_api.py` |
| 2 | Check model card | Visit HuggingFace URL |
| 3 | Test in Python | Run quick test script above |
| 4 | Document findings | Update this file |
| 5 | Update implementation | Fix `src/alignment/forced_aligner.py` if needed |

**Your concern was spot-on. Let's verify before proceeding!** 🎯
