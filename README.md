# Copycat — Local AI Avatar Chatbot with Lip-Sync

An **offline, CPU-only** personal AI assistant that talks back with **your own face and voice**.

**Pipeline:** Speech → Whisper (STT) → RAG (your diary) → LLM (Qwen2.5) → XTTS v2 (voice clone) → Wav2Lip (lip-sync) → Video

---

## Minimum Requirements

| Hardware | Minimum |
|----------|---------|
| CPU | 4 cores |
| RAM | 10 GB |
| Storage | 10 GB free |
| GPU | Not required |

**Linux** (Debian 12 tested), Windows, macOS.

---

## One-command Setup

```bash
git clone <this-repo> copycat
cd copycat
bash setup.sh
```

The script does **everything**:
- Installs system packages (`ffmpeg`, `python3-venv`, etc.)
- Creates a `venv` and installs the ultra-fast `uv` package manager
- Downloads the CPU-only version of PyTorch to save space (~3GB)
- Installs all Python dependencies instantly with `uv`
- Automatically downloads `wav2lip.pth` (~416 MB)
- Pulls the `qwen2.5:3b` model with Ollama
- Creates `voices/` and `diario/` folders

> **Note:** First run downloads ~2 GB of additional models (XTTS v2, Whisper, sentence-transformer) automatically.

---

## Asset Preparation

Before running, place these files in the project root:

```
copycat/
├── face.jpeg          ← Your front-facing portrait (full face, well-lit)
├── voices/
│   ├── es.wav         ← Spanish voice reference (6-15s, clean audio)
│   └── en.wav         ← English voice reference (6-15s, clean audio)
└── diario/
    ├── 2024-07-15.md  ← Your diary entries as .md files
    └── ...
```

**Tips:**
- **Photo:** Front-facing, unobstructed, minimum 256×256
- **Voice:** Record 6–15s with the same microphone you'll use for chatting
- **Diary:** More entries = better RAG responses

**Sample texts for recording your voice reference:**

> **es.wav (español):**
> _"Hoy fue un día bastante productivo, aunque empecé un poco tarde. Por la mañana terminé el informe que tenía pendiente y después aproveché para ordenar las ideas del proyecto nuevo. Me gusta trabajar con calma, escuchando música de fondo, y siento que así rindo mucho mejor. La semana que viene tengo que preparar la presentación, así que voy a dedicarle tiempo el fin de semana para llegar tranquilo."_

> **en.wav (English):**
> _"I've been thinking a lot about how quickly things change around us. One day you're comfortable with your routine, and the next you're learning something completely new. I find that the best way to handle uncertainty is to stay curious and keep asking questions. Every challenge is really just an opportunity to grow, even when it doesn't feel that way at first."_

Read the corresponding text aloud in your natural tone, record with the same microphone you'll use for chatting, and save the file as `voices/es.wav` or `voices/en.wav`. The cloner will capture your characteristic cadence, intonation, and timbre.

---

## Run

```bash
source venv/bin/activate
python3 main.py
```

Wait for _"Listo. Pulsa Escucha o Listen."_ and press the button.

---

## Customization

| Change | Where |
|--------|-------|
| LLM model | `app.py:79` — change `OLLAMA_MODEL` |
| Voice reference | Replace `voices/es.wav` or `voices/en.wav` |
| Photo | Replace `face.jpeg` |
| Knowledge | Add `.md` files to `diario/` (auto-reindexed) |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| _"Face not detected!"_ | Use a well-lit front-facing photo |
| Wav2Lip crashes | Reduce `--wav2lip_batch_size` in `app.py` |
| No microphone | `python -c "import sounddevice; print(sounddevice.query_devices())"` |
| ChromaDB error | Delete `chroma_db/` and restart |

---

## Tech Stack

| Component | Model |
|-----------|-------|
| STT | [Whisper tiny](https://github.com/openai/whisper) |
| RAG | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) + [ChromaDB](https://www.trychroma.com/) |
| LLM | [Qwen2.5-3B](https://ollama.ai/library/qwen2.5:3b) via [Ollama](https://ollama.ai/) |
| TTS | [XTTS v2](https://github.com/coqui-ai/TTS) |
| Lip-sync | [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) |
| GUI | [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) |

---

## Changelog

### [Unreleased]
- **Dependency Installation Optimized:** Switched from `pip` to `uv` in `setup.sh` for blazing-fast dependency resolution, completely avoiding `pip`'s backtracking issues with heavy ML libraries.
- **CPU-Only Optimization:** PyTorch is now explicitly installed from the CPU-only index prior to requirements, saving ~3GB of disk space and bandwidth by avoiding unnecessary CUDA toolkits.
- **Numpy Compatibility:** Restricted `numpy<2.0.0` in `requirements.txt` to prevent breaking changes with older versions of `scipy`, `librosa`, and `TTS`.
- **Transformers Compatibility:** Pinned `transformers<4.40.0` to fix a crash (`cannot import name 'BeamSearchScorer'`) caused by `TTS` incompatibility with newer `transformers` versions.
- **Redundancy Fix:** Removed redundant `TTS` installation from `setup.sh` since it's already managed in `requirements.txt`.
