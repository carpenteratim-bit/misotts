# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Miso TTS 8B is a Python inference library for a text-to-dialogue RVQ Transformer model. It generates speech audio from text and optional audio context using a dual-transformer architecture inspired by the Sesame CSM architecture. This repository contains inference code only — no training code.

## Environment Setup

**Recommended (uv):**
```bash
uv sync --python 3.10
source .venv/bin/activate
```

**Alternative (pip):**
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requires Python >=3.10, <3.13. `bitsandbytes` is Linux-only (optional quantization).

## Running

```bash
uv run python run_misotts.py   # generates full_conversation.wav
```

The first run downloads two models from Hugging Face:
1. `MisoLabs/MisoTTS` — the main TTS model (~16GB)
2. `sony/silentcipher` — the watermarking model

If the watermark download times out, rerun; the HF cache resumes from completed files.

Override the model path with `MISO_TTS_8B_MODEL` env var, or pass `model_path_or_repo_id` to `load_miso_8b()`. Set `NO_TORCH_COMPILE=1` to disable Triton JIT compilation.

## Verifying Watermarks

```bash
python watermarking.py --audio_path output.wav
```

## Architecture

### Dual-Transformer Design

The model uses two Llama 3.2-style transformers:

- **Backbone** (`llama-8B`, 32 layers, 4096 dim): Consumes interleaved text + audio frame embeddings. Predicts codebook 0 (first RVQ codebook) via `codebook0_head`.
- **Decoder** (`llama-300M`, 8 layers, 1536 dim): Autoregressively predicts codebooks 1–31 within each 80ms frame. Takes backbone output projected down via `self.projection` plus the previously sampled codebook embeddings.

The token frame layout is shape `(seq_len, audio_num_codebooks + 1)` where the last column holds text tokens and the first 32 columns hold audio codebook tokens. A corresponding boolean mask distinguishes which slots are populated.

### Token Embedding

Audio tokens are stored in a single flat `audio_embeddings` table of size `audio_vocab_size * audio_num_codebooks`. Codebook `i` tokens are offset by `i * audio_vocab_size` before embedding, so codebooks share one `nn.Embedding` module.

### Generation Loop (`generator.py:generate`)

1. Context segments (prior conversation turns) are tokenized via `_tokenize_segment` — both their text and audio are encoded.
2. The current text prompt is tokenized via `_tokenize_text_segment`.
3. The model calls `generate_frame` autoregressively (one call = one 80ms audio frame).
4. An all-zeros frame signals EOS and stops generation.
5. Collected frame tokens are decoded by the Mimi audio codec back to a waveform.
6. The waveform is watermarked via SilentCipher before being returned.

### Key Files

| File | Role |
|------|------|
| `generator.py` | Public API: `load_miso_8b()`, `Generator`, `Segment` |
| `models.py` | `Model` class, `MISO_TTS_8B_CONFIG`, sampling logic |
| `watermarking.py` | SilentCipher encode/decode; CLI watermark checker |
| `moshi_compat.py` | Monkey-patches bitsandbytes for mixed-precision models |
| `run_misotts.py` | End-to-end multi-speaker conversation example |

### Model Configuration Constants (`models.py:115`)

```
text_vocab_size: 128,256
audio_vocab_size: 2,051
audio_num_codebooks: 32
sample_rate: 24,000 Hz
max_seq_len: 2,048
default dtype: bfloat16
```

## Public API

```python
from generator import load_miso_8b, Segment

generator = load_miso_8b(device="cuda")   # or "cpu" (slow)

# Basic synthesis
audio = generator.generate(
    text="Hello.",
    speaker=0,          # integer speaker ID
    context=[],         # list of prior Segment objects
    max_audio_length_ms=10_000,
    temperature=0.9,    # default
    topk=50,            # default
)
# returns 1-D torch.Tensor at generator.sample_rate (24000 Hz)

# Voice cloning: pass prompt audio as context
context = [Segment(speaker=0, text="Transcript.", audio=prompt_tensor)]
audio = generator.generate(text="Next sentence.", speaker=0, context=context)
```

`Segment.audio` must be a 1-D float tensor at 24,000 Hz (mono). The `Segment` dataclass intentionally has no default for `audio`; pass an empty `context=[]` list for unprompted generation.

## Watermarking Note

All generated audio is watermarked with `MISO_TTS_WATERMARK = [0, 0, 0, 0, 0]` (public demo key). If deploying in another application, replace this key in `watermarking.py` with a private value and keep it secret.
