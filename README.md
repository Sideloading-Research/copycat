# Copycat — Local AI Avatar Chatbot with Lip-Sync

An offline, CPU-only personal AI assistant that talks back with **your own face and voice**.

**Pipeline:** Speech-to-text → RAG on your diary → LLM → Voice cloning → Lip-sync → Playback

![demo](PROJECT.md)

---

## How It Works

```
You speak ──→ Whisper tiny ──→ Ollama/Qwen2.5-3B ──→ XTTS v2 ──→ Wav2Lip ──→ Avatar talks back
                    │                    ↑                 │            │
              (STT) │     ┌──────────────┘       (voice clone)   (lip-sync)
                    │     │
                    ↓     │
              ChromaDB ───┘
              (RAG over
               your diary)
```

1. **STT** — Whisper tiny transcribes your voice.
2. **RAG** — Your personal diary (`diario/*.md`) is searched for relevant context.
3. **LLM** — Ollama + Qwen2.5-3B generates a short, contextual reply.
4. **TTS** — XTTS v2 clones your voice from a reference file and speaks the reply.
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

### 3. Install dependencies

```bash
# Debian / Ubuntu
sudo apt update
sudo apt install python3-venv ffmpeg git build-essential \
                 portaudio19-dev libsndfile1

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install Python packages
pip install customtkinter pillow sounddevice soundfile numpy \
            opencv-python-headless openai-whisper ollama \
            langchain langchain-community langchain-chroma \
            langchain-huggingface langchain-text-splitters \
            sentence-transformers chromadb TTS

# Download Wav2Lip
git clone https://github.com/Rudrabha/Wav2Lip.git
cd Wav2Lip
pip install -r requirements.txt
# Download model weights (save as checkpoints/wav2lip.pth):
# https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW
cd ..

# Pull the LLM
ollama pull qwen2.5:3b
```

> **Note:** First run will download ~2 GB of model files (XTTS v2, Whisper, sentence-transformer). This happens automatically.

### 4. Precompute voice latents (optional, speeds up first run)

```bash
python -c "
from tts_manager import TTSManager
tm = TTSManager('voices/es.wav', 'voices/en.wav',
                'voices/es_latents.pth', 'voices/en_latents.pth')
print('Latents cached.')
"
```

### 5. Run

```bash
source venv/bin/activate
python app.py
```

Wait for _"Listo. Pulsa Escucha o Listen."_ — then press **Escucha** (Spanish) or **Listen** (English) and speak.

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
|---|---|---|
| OMP/MKL thread pinning (6 threads) | `app.py:21-25` | Prevents CPU oversubscription |
| `KMP_BLOCKTIME=0` + affinity | `app.py:23-24` | Better cache locality |
| `torch.inference_mode()` | `tts_manager.py:83`, `Wav2Lip/inference.py` | Skip autograd overhead (~10% faster) |
| `num_ctx=1024` for Ollama | `app.py` | Smaller KV cache, faster decoding |
| FFmpeg `-preset veryfast` | `Wav2Lip/inference.py` | Video muxing 2-3× faster |
| Precomputed speaker latents | `tts_manager.py` | Avoid 30s re-computation every query |
| `gc.collect()` after pipeline | `app.py` | Free memory between turns |

---

## Known Limitations

- **Latency:** 30–90s per response on CPU (XTTS v2 is the main bottleneck).
- **RAM:** ~10–12 GB used during inference (XTTS ~6 GB + Ollama ~3 GB + Whisper).
- **Audio playback:** Blocking (no overlap with next pipeline start).
- **Single face:** Only one face photo supported.
- **Lip-sync quality:** Wav2Lip works best with clear, well-lit face images.

---

## Troubleshooting

| Symptom | Likely fix |
|---|---|
| _"Face not detected!"_ | Use a front-facing photo, good lighting. Try `--pads 0 20 0 0` in Wav2Lip args. |
| Wav2Lip crashes / OOM | Reduce `--wav2lip_batch_size` (default 128). |
| XTTS v2 fails to load | Check `COQUI_TOS_AGREED=1` and internet for first download. |
| No microphone input | Check `sounddevice` device list: `python -c "import sounddevice; print(sounddevice.query_devices())"` |
| ChromaDB index errors | Delete `chroma_db/` and restart — it will rebuild. |

---

## Tech Stack

| Component | Model | Role |
|---|---|---|
| STT | [Whisper tiny](https://github.com/openai/whisper) | Speech-to-text |
| RAG | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) + [ChromaDB](https://www.trychroma.com/) | Semantic search over diary |
| LLM | [Qwen2.5-3B](https://ollama.ai/library/qwen2.5:3b) via [Ollama](https://ollama.ai/) | Response generation |
| TTS | [XTTS v2](https://github.com/coqui-ai/TTS) | Voice cloning |
| Lip-sync | [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) | Audio-to-video synchronisation |
| GUI | [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) | Desktop interface |

---

## License

This project uses several open-source models and libraries under their respective licenses (MIT, Apache 2.0, Coqui Public Model License). Check each component's repository for details.
