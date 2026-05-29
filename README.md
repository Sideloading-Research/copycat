# Copycat — Local AI Avatar Chatbot with Lip-Sync

An offline, CPU-only personal AI assistant that talks back with **your own face and voice**.

**Pipeline:** Speech-to-text → RAG on your diary → LLM → Voice cloning → Lip-sync → Playback

![demo](PROJECT.md)

---

## How It Works

```
You speak ──→ faster-whisper ──→ Ollama/Qwen2.5-3B ──→ Pocket-TTS ──→ Wav2Lip ──→ Avatar talks back
                    │                     ↑                  │              │
               (STT) │      ┌──────────────┘    (voice clone)        (lip-sync)
                     │      │
                     ↓      │
               ChromaDB ────┘
               (RAG over
                your diary)
```

1. **STT** — faster-whisper tiny (~4× faster than openai-whisper) transcribes your voice.
2. **RAG** — Your personal diary (`diario/*.md`) is searched for relevant context.
3. **LLM** — Ollama + Qwen2.5-3B generates a short, contextual reply.
4. **TTS** — Pocket-TTS (100M params, 6× real-time on CPU) clones your voice from a reference file and speaks the reply.
5. **Wav2Lip** — Syncs the spoken audio with your face photo (lips move in sync).
6. **Playback** — The avatar speaks back through your speakers + GUI.

---

## What You Need

| Hardware | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 10 GB | 16 GB |
| Storage | 10 GB free | 20 GB free |
| GPU | Not required | Not required |

Compatible with **Linux** (Debian 12 tested), **Windows**, and **macOS**.

---

## Setup

### 1. Clone & prepare

```bash
git clone <this-repo> copycat
cd copycat
```

### 2. Prepare your avatar assets

Place these files in the project root:

```
copycat/
├── face.jpeg          ← A front-facing portrait photo (any size)
├── voices/
│   ├── es.wav         ← Spanish voice reference (6-15s, clean audio)
│   └── en.wav         ← English voice reference (6-15s, clean audio)
└── diario/
    ├── 2024-07-15.md  ← Your personal diary entries (.md files)
    ├── 2024-07-16.md
    └── ...
```

**Voice reference tips:**
- Record 6–15 seconds of **clear, isolated speech** (no background noise).
- Use the same microphone you'll use for chatting.
- Speak naturally — the clone captures your cadence and tone.
- Supported formats: `.wav`, `.mp3`.

**Face photo tips:**
- Front-facing, good lighting, neutral expression.
- Crop to show head and shoulders.
- The Wav2Lip model works best with **256×256** or larger.

**Diary format:**
- Any `.md` file inside `diario/` is automatically indexed at startup.
- Markdown formatting is preserved in the text chunks.
- The more entries you have, the better the RAG context.

### 3. One-command setup

```bash
bash setup.sh
```

This installs system packages, creates a venv, installs all Python dependencies (CPU-only PyTorch, MeloTTS, OpenVoice v2, faster-whisper, Wav2Lip, etc.), downloads unidic + NLTK data for TTS, and patches OpenVoice's `se_extractor.py` for CPU compatibility.

### 4. Run

```bash
source venv/bin/activate
python app.py
```

Wait for _"Listo. Pulsa Escucha o Listen."_ — then press **Escucha** (Spanish) or **Listen** (English) and speak.

> First run downloads ~1 GB of model files (faster-whisper, sentence-transformer, OpenVoice v2 checkpoints). This happens automatically.

---

## Customisation

### Language / Voice

Edit `app.py` paths or swap the files in `voices/`:

| File | Purpose |
|------|---------|
| `voices/es.wav` | Spanish voice reference |
| `voices/en.wav` | English voice reference |
| `voices/*.pth` | Auto-generated latent caches (delete to regenerate) |

### LLM

Edit `OLLAMA_MODEL` in `app.py:79` to use any model from [Ollama library](https://ollama.ai/library). Smaller models (`tinyllama`, `phi`) are faster; larger ones (`llama3`, `mistral`) give better answers.

### Face

Replace `face.jpeg` with your own photo. Wav2Lip works with any front-facing portrait.

### Diary / Knowledge Base

Add or edit `.md` files in `diario/`. The RAG index rebuilds automatically when new files are detected (or delete `chroma_db/` to force a full re-index).

---

## Optimisations Applied

The codebase is tuned for **CPU-only inference on limited RAM**:

| Optimisation | File | Effect |
|---|---|---|---|
| OMP/MKL thread pinning (6 threads) | `app.py:21-25` | Prevents CPU oversubscription |
| `KMP_BLOCKTIME=0` + affinity | `app.py:23-24` | Better cache locality |
| `torch.inference_mode()` | `Wav2Lip/inference.py` | Skip autograd overhead (~10% faster) |
| `num_ctx=1024` for Ollama | `app.py` | Smaller KV cache, faster decoding |
| FFmpeg `-preset veryfast` | `Wav2Lip/inference.py` | Video muxing 2-3× faster |
| Pocket-TTS over MeloTTS+OpenVoice | `tts_manager.py` | 100M params vs 1.5GB; ~1s vs ~30s latency; native voice cloning |
| `gc.collect()` after pipeline | `app.py` | Free memory between turns |
| `doc_loader.py` replaces langchain-community | `app.py:55` | Eliminates dependency on sunsetted package |
| Pocket-TTS quantize=True | `tts_manager.py` | int8 quantization halves memory with negligible quality loss |

---

## Known Limitations

- **Latency:** ~5–15s per response on CPU (Pocket-TTS ~1s, Ollama ~5-10s, Wav2Lip ~15s).
- **Voice cloning requires HF token:** [Accept terms](https://huggingface.co/kyutai/pocket-tts) and login with `hf auth login` for voice cloning. Without a token, preset voices are used (alba/lola).
- **RAM:** ~6–8 GB during inference (Pocket-TTS ~500 MB quantized + Ollama ~3 GB + faster-whisper).
- **Audio playback:** Blocking (no overlap with next pipeline start).
- **Single face:** Only one face photo supported.
- **Lip-sync quality:** Wav2Lip works best with clear, well-lit face images.
- **langchain-community sunset:** `langchain-community` was archived on 2026-05-26. Replaced with `doc_loader.py`.

---

## Troubleshooting

| Symptom | Likely fix |
|---|---|
| _"Face not detected!"_ | Use a front-facing photo, good lighting. Try `--pads 0 20 0 0` in Wav2Lip args. |
| Wav2Lip crashes / OOM | Reduce `--wav2lip_batch_size` (default 128). |
| Pocket-TTS voice cloning not working | Sign in to Hugging Face: `hf auth login` and accept terms at https://huggingface.co/kyutai/pocket-tts |
| Voice doesn't sound like reference | Ensure you're logged in with `hf auth login` and have accepted Pocket-TTS terms on HF. The preset fallback voices won't match your voice. |
| No microphone input | Check `sounddevice` device list: `python -c "import sounddevice; print(sounddevice.query_devices())"` |
| ChromaDB index errors | Delete `chroma_db/` and restart — it will rebuild. |

---

## Tech Stack

| Component | Model | Role |
|---|---|---|---|
| STT | [faster-whisper tiny](https://github.com/SYSTRAN/faster-whisper) | Speech-to-text (~4× faster than openai-whisper) |
| RAG | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) + [ChromaDB](https://www.trychroma.com/) | Semantic search over diary |
| LLM | [Qwen2.5-3B](https://ollama.ai/library/qwen2.5:3b) via [Ollama](https://ollama.ai/) | Response generation |
| TTS | [Pocket-TTS](https://github.com/kyutai-labs/pocket-tts) | Voice cloning (100M params, 6× real-time CPU) |
| Lip-sync | [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) | Audio-to-video synchronisation |
| GUI | [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) | Desktop interface |

---

## Changelog

### 2026-05-29 — CPU-only stabilisation and quality tuning

| Change | Reason |
|--------|--------|
| **`openai-whisper` → `faster-whisper`** | 4× speedup on CPU; `tiny` model with `int8` compute |
| **`se_extractor.py` patched** | `device="cuda"` → `"cpu"`, `float16` → `int8`; no CUDA dependency |
| **Source SE from MeloTTS dummy audio** | Precomputed `en-default.pth` didn't match actual MeloTTS English speakers (`EN-US`). Now generating longer dummy text (3× repeated) to pass VAD and extracting from actual MeloTTS output |
| **`tau` raised to 0.6** | Stronger voice cloning in OpenVoice tone color converter |
| **`speed=0.9` in MeloTTS** | Slower synthesis → better base audio quality for conversion |
| **Voice reference: 30s recordings recommended** | Original 6-15s insufficient for faithful cloning. Provided script for 30s recordings in ES/EN |
| **`unidic` + NLTK data downloaded** | Required by MeloTTS (Japanese tokenizer) and g2p (English POS tagger) |
| **`silero-vad` pre-cached** | Avoids interactive PyTorch trust prompt at runtime |
| **`setup.sh` updated** | Added unidic download, NLTK download, se_extractor.py CPU patch |
| **`langchain-community` removed** | Package was sunset on 2026-05-26. Replaced with `doc_loader.py` (50-line standalone `DirectoryLoader`/`TextLoader`) |
| **`soxr` added to requirements** | Required by `transformers>=4.46` (dependency of `sentence-transformers`) |

### 2026-05-29 (later) — Migrated to Pocket-TTS

| Change | Reason |
|--------|--------|
| **MeloTTS + OpenVoice v2 → Pocket-TTS** | 100M params vs 1.5GB. English TTS: 1s vs 9s, Spanish TTS: 2.5s vs 5s. Voice cloning from raw WAV, no two-stage pipeline. No `se_extractor.py` patching needed. |
| **Voice state caching via safetensors** | `voices/{lang}_pocket.safetensors` caches the extracted voice embedding for near-instant reloading on subsequent runs |
| **Preset voice fallback** | If no Hugging Face token is available, falls back to preset voices (`lola` for ES, `alba` for EN) |
| **`app.py` simplified** | Removed `latent_es_path` / `latent_en_path` parameters (no longer needed) |
| **`requirements.txt` updated** | `pocket-tts` added; `openvoice`, `melotts` and their transitive deps can be removed |

### Known remaining issues

- **Voice cloning requires HF token:** Accept terms at https://huggingface.co/kyutai/pocket-tts and login with `hf auth login`
- **Wav2Lip realism:** Single-photo lip sync has inherent quality limits

---

## License

This project uses several open-source models and libraries under their respective licenses (MIT, Apache 2.0, CC-BY-4.0). Check each component's repository for details.
