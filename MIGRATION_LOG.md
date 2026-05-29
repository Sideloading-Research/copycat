# Migration Log — Pocket-TTS + faster-whisper Experiment

## Goal
Replace XTTS v2 (30-90s latency) with faster alternatives for CPU-only inference.

## Changes Applied

### TTS: XTTS v2 → MeloTTS + OpenVoice v2 → Pocket-TTS
- **XTTS v2**: 6 GB model, 30-90s per generation. Best voice cloning quality but unusably slow on CPU.
- **MeloTTS + OpenVoice v2**: Two-step pipeline. MeloTTS generates base audio, OpenVoice does voice conversion. ~5-9s latency. Required patching `se_extractor.py` for CPU. Required NLTK, unidic, and other language deps. OpenVoice v2's VAD and CUDA dependencies made it fragile.
- **Pocket-TTS**: 100M params, ~1-2s latency, single-step voice cloning from WAV. Kyutai's model is gated on HuggingFace (requires terms acceptance + login). Voice cloning weights at `kyutai/pocket-tts`.

### STT: openai-whisper → faster-whisper
- Same Whisper tiny architecture, 4× faster on CPU via CTranslate2. No quality loss.

### RAG: langchain-community removed
- Package was sunset (archived 2026-05-26). Replaced with 50-line `doc_loader.py`.

## Problems Encountered

### Pocket-TTS Voice Cloning

| Issue | Detail |
|-------|--------|
| **Gated model** | `kyutai/pocket-tts` requires HF login + terms acceptance. Without it, silently falls back to preset voices. |
| **New `hf` CLI vs old `huggingface-cli`** | The new `hf` CLI stores tokens in a format `huggingface_hub` (the Python library) doesn't recognize. Must use `huggingface-cli login` (now deprecated) or manually copy token to `~/.huggingface/token`. |
| **Fine-grained token permissions** | HF fine-grained tokens need explicit "Access to public gated repositories" enabled. Classic "Read" tokens work out of the box. |
| **Low-quality reference audio** | Pocket-TTS expects 24 kHz, mono, clean audio. 8 kHz phone recordings (even upsampled) produce degraded voice embeddings. Output is 0.24s of garbled audio or barely audible speech with poor cloning fidelity. |
| **No EOS detection** | With poor conditioning, the model reaches max generation length without EOS, truncating output. |
| **Caching issues** | `export_model_state`/`get_state_for_audio_prompt` caching can corrupt if the WAV files change or are incompatible. |

## Verdict

**Pocket-TTS is excellent for preset voices but unreliable for voice cloning on CPU with typical user recordings.**

- Preset voices (`alba`, `lola`): ✅ Fast (~1s), good quality
- Voice cloning with 24 kHz clean audio: ⚠️ Untested but promising
- Voice cloning with 8 kHz phone recordings: ❌ Produces garbled/truncated output

The fundamental problem: voice cloning on CPU is still an unsolved problem for consumer-grade hardware and recordings. All current approaches (XTTS v2, OpenVoice, Pocket-TTS) either require:
- A GPU (XTTS v2 needs 6 GB+ VRAM)
- Studio-quality reference audio (Pocket-TTS needs 24 kHz clean recordings)
- Complex preprocessing pipelines

**Recommendation:** For reliable voice cloning, revert to XTTS v2 with GPU acceleration, or use a cloud API (ElevenLabs, OpenAI TTS).

## Instructions for Better Results (if continuing with Pocket-TTS)

1. Record reference audio at 24 kHz, mono, 10-15 seconds, clean environment
2. Convert with: `ffmpeg -i input.wav -ac 1 -ar 24000 output.wav`
3. Accept terms at https://huggingface.co/kyutai/pocket-tts
4. Login: `huggingface-cli login` (not `hf auth login`) with a classic Read token
5. Delete cached states: `rm -f voices/*_pocket.safetensors voices/*_24k_mono.wav`
6. Run: `python app.py`
