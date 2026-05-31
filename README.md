# Copycat — Local AI Avatar with Human Personality Simulation

An **offline, CPU-only** personal AI avatar that looks like you, speaks with your cloned voice, and simulates a real human personality using your diary as biographical memory.

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
git clone <repo-url> copycat
cd copycat
bash setup.sh
```

The script does **everything**:
- Installs system packages (`ffmpeg`, `python3-venv`, etc.)
- Creates a `venv` with the ultra-fast `uv` package manager
- Downloads CPU-only PyTorch to save ~3 GB
- Installs all Python dependencies with `uv`
- Downloads `wav2lip.pth` (~416 MB) into `src/Wav2Lip/checkpoints/`
- Pulls `qwen2.5:3b` with Ollama
- Includes Wav2Lip source code
- Creates the required folder structure

> **Note:** First run downloads ~2 GB of additional models (XTTS v2, Whisper, sentence-transformer) automatically.

---

## Asset Preparation

Place these files **before** running:

```
copycat/
├── src/
│   ├── main.py                 ← Entry point
│   ├── core/                   ← STT, RAG, LLM, TTS, Wav2Lip pipeline
│   ├── ui/                     ← CustomTkinter GUI (splash, chat, settings)
│   ├── utils/                  ← Paths, environment tuning
│   ├── Wav2Lip/                ← Lip-sync inference
│   └── doc_loader.py           ← Langchain-community replacements
├── data/
│   ├── picture/face.jpeg       ← Avatar portrait
│   ├── voices/{lang}.wav       ← Voice reference samples
│   ├── behavior/behavior.txt   ← Personality definition
│   ├── journal/*.md            ← Diary entries (RAG knowledge base)
│   └── vector_db/              ← ChromaDB persistent index
├── tmp/                        ← Temporary audio / video files
├── setup.sh                    ← One-command installer
├── run.sh                      ← Launcher
└── requirements.txt            ← Python dependencies
```

**Sample recording texts:**

> **es.wav (español):** _"Hoy fue un día bastante productivo, aunque empecé un poco tarde. Por la mañana terminé el informe que tenía pendiente y después aproveché para ordenar las ideas del proyecto nuevo."_

> **en.wav (English):** _"I've been thinking a lot about how quickly things change around us. One day you're comfortable with your routine, and the next you're learning something completely new."_

---

## Run

```bash
./run.sh
```

Or manually:

```bash
source venv/bin/activate
python3 src/main.py
```

A splash screen shows loading progress. When "Ready!" appears the main window opens — press the microphone button or type a message.

---

## Behaviour / Personality Engineering

Copycat uses a three-layer prompt to make the LLM simulate a human personality:

1. **`data/behavior/behavior.txt`** — defines the core personality (traits, speech patterns, beliefs).
2. **`data/journal/*.md`** — biographical memories retrieved via RAG (ChromaDB + sentence-transformer).
3. **Name detection** — saying *"me llamo X"* or *"my name is X"* sets the persona name for the session.

The system prompt instructs the model to **never** reveal it is an AI, language model, or program. It speaks in first person, drawing from its personality and memories as if they were real life experiences.

---

## Customization

| Change | How |
|--------|-----|
| Personality | Edit `data/behavior/behavior.txt` or use Settings → Save Rules |
| LLM model | Edit `engine.py:100` — change `model="qwen2.5:3b"` |
| Voice reference | Replace `data/voices/es.wav` or `data/voices/en.wav` |
| Face photo | Replace `data/picture/face.jpeg` |
| Knowledge base | Add `.md` files to `data/journal/` (auto-reindexed on next start) |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| _"Wav2Lip not found"_ | Ensure `src/Wav2Lip/` exists (included in repo) |
| Wav2Lip crashes | Reduce batch size in `engine.py:_sync_lips` |
| No microphone | `python -c "import sounddevice; print(sounddevice.query_devices())"` |
| ChromaDB error | Delete `data/vector_db/` and restart |
| No voice output | Check that `data/voices/` has at least one `.wav` file |

---

## Tech Stack

| Component | Model |
|-----------|-------|
| STT | [Whisper tiny](https://github.com/openai/whisper) |
| RAG | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) + [ChromaDB](https://www.trychroma.com/) |
| LLM | [Qwen2.5-3B](https://ollama.ai/library/qwen2.5:3b) via [Ollama](https://ollama.ai/) |
| TTS | [XTTS v2](https://github.com/coqui-ai/TTS) — voice cloning |
| Lip-sync | [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) |
| GUI | [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) |

---

## Architecture & Execution Flow

### Project Structure

```
copycat/
├── src/
│   ├── main.py                 ← Entry point. Splash → model loading → MainWindow
│   ├── core/
│   │   ├── engine.py           ← CopycatEngine: pipeline orchestrator (STT→RAG→LLM→TTS→Wav2Lip)
│   │   ├── tts_manager.py      ← TTSManager: XTTS v2 lifecycle + speaker latent cache
│   │   ├── rag.py              ← RAGManager: ChromaDB + sentence-transformer
│   │   └── audio.py            ← AudioHandler: microphone record + playback
│   ├── ui/
│   │   ├── main_window.py      ← MainWindow: chat UI, avatar, mic button, settings
│   │   ├── settings.py         ← Configuration dialog: face, voice, diario, behavior
│   │   └── splash.py           ← SplashScreen: animated progress during model loading
│   ├── utils/
│   │   ├── paths.py            ← Centralised path registry (PATHS dict)
│   │   └── setup_env.py        ← CPU thread tuning, torch.load patch, warning filters
│   ├── Wav2Lip/                ← Lip-sync inference
│   ├── doc_loader.py           ← Standalone replacements for deprecated langchain-community
│   └── assets/logo.png         ← Splash icon
├── data/
│   ├── picture/face.jpeg       ← Avatar portrait
│   ├── voices/{lang}.wav       ← Voice reference samples
│   ├── behavior/behavior.txt   ← Personality definition
│   ├── journal/*.md            ← Diary entries (RAG knowledge base)
│   └── vector_db/              ← ChromaDB persistent index
├── tmp/                        ← Temporary audio / video files
├── setup.sh                    ← One-command installer
├── run.sh                      ← Launcher
└── requirements.txt            ← Python dependencies
```

### Execution Flow

```
User clicks 🎤 (or types)
        │
        ▼
┌─────────────────┐
│  AudioHandler    │  Record 16 kHz PCM → save to tmp/_tmp_user.wav
│  (sounddevice)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CopycatEngine  │  run_pipeline(lang, manual_text, status_cb, chat_cb)
│                  │
│  1. STT          │  Whisper tiny → transcribe tmp/_tmp_user.wav
│                  │    (auto-detects spoken language, result.language)
│                  │
│  2. Name Detect  │  Regex "me llamo X" / "my name is X" → self.persona_name
│                  │
│  3. RAG Search   │  ChromaDB → similarity_search(user_text, k=2)
│                  │    returns diary excerpts as context string
│                  │
│  4. Build Prompt │  _build_persona_prompt(behavior, context, user_text,
│                  │                          output_lang, spoken_lang)
│                  │    Forces output language regardless of input
│                  │    Injects personality + memories + role-play rules
│                  │
│  5. LLM          │  Ollama → qwen2.5:3b → generate(prompt)
│                  │    Bot response in output language
│                  │
│  6. TTS          │  TTSManager → XTTS v2 → generate_tts(text, lang)
│                  │    Uses cached speaker latents → 24 kHz WAV
│                  │
│  7. Wav2Lip      │  Subprocess → inference.py → lip-synced video
│  8. Playback     │  sounddevice (audio) + after-loop (video frames)
│                  │
│  9. Cleanup      │  Delete tmp/* before next run
└─────────────────┘
```

### Threading Model

```
main thread (Tkinter)         background threads
─────────────────────         ──────────────────
main.py:
  root.withdraw()
  SplashScreen(root)           _load_all_models(splash)
  root.after(100, check)         ├── Whisper (tiny)
  root.mainloop()                 ├── RAG (ChromaDB init)
                                  ├── XTTS v2 (~6 GB model)
                                  └── splash.close()

  _reveal(root, engine)
  MainWindow(engine)             (no more bg threads — already loaded)
    │
    ├── _send_text_manual()      _process_pipeline(lang, text)
    ├── _toggle_voice()           ├── run_pipeline()
    │                              ├── play_audio()  [sd.play + sd.wait]
    │                              └── _play_video()  [after-loop on main thread]
    └── _on_closing()
         └── save_session_log()
```

---

## Code Analysis — Issues Detected & Fixed

| CLI Symptom | Root Cause | Fix |
|---|---|---|
| `"Loading vector database from…"` printed twice | `rag.initialize()` called in both `main.py` and `engine.load_models()` | Removed duplicate call; `load_models()` now only calls `self.rag.initialize()` once |
| `invalid command name "…check_dpi_scaling"` | `set_progress()` called `self.update()` from background thread (Tcl not thread-safe) | Changed to `self.after(0, lambda: …)` + `update_idletasks()` |
| `FutureWarning: resume_download is deprecated` | `huggingface_hub` warns about deprecated param | Suppressed via `warnings.filterwarnings` in `setup_env.py` |
| Noise: `(80, 305)` `Reading video frames…` etc. | Wav2Lip + ffmpeg write debug info to stdout/stderr | Subprocess now runs with `stdout=DEVNULL, stderr=STDOUT` |
| Stale audio from previous failed pipeline | Temp files accumulated | Added `_cleanup_temp()` at start of each `run_pipeline()` |
| `after` loop kept firing after window close | No `winfo_exists()` guard in `_play_video()` loop | Added `if not self.winfo_exists(): return` |
| Whisper forced to output language | `transcribe(…, language=lang)` used output lang for ASR | Changed to auto-detect: removed `language=` param |
| Duplicate RAGManager instances | `load_models()` created a 2nd RAGManager | `load_models()` reuses `self.rag` from `__init__` |

---

## Future Proposals & Known Limitations

| Area | Issue | Proposed Solution |
|---|---|---|
| **Language mismatch** | Voice cloning only has `en` + `es`; user may speak other langs | Map output to closest available voice; add language auto-select |
| **Single face frame** | Wav2Lip gets only 1 frame → output is a short loop | Generate multiple face variants or use a video with idle motion |
| **Session context** | LLM has no conversation memory between turns | Add chat history buffer with sliding window (last N exchanges) |
| **Voice selection** | Settings always saves to `es.wav` regardless of file chosen | Detect language from filename or add a language dropdown |
| **Model unload** | XTTS + Whisper stay in RAM until process exit | Add `unload_models()` method; free GPU-less RAM via `del` + `gc` |
| **Error recovery** | Pipeline crash leaves UI buttons disabled | Add timeout + auto-re-enable buttons after N seconds |
| **Multi-turn RAG** | Each query re-searches the same diary | Cache recent contexts; only re-query on new topics |
| **Progress feedback** | Wav2Lip runs silently for ~5s (blocking subprocess) | Show indeterminate progress bar or "Generating video…" timer |
| **Voice latents** | No UI to trigger latent recomputation | Add "Refresh latents" button in Settings |
| **Logging** | All output goes to stdout; no log file | Add `logging` module with file rotation |

---

## Changelog

### [Unreleased]
- **Persona Engineering:** Three-layer prompt (behavior + RAG memories + name detection) forces human role-play; model never reveals it is an AI.
- **Language Coercion:** Model responds in the selected output voice language regardless of input language.
- **Name Detection:** Saying "me llamo X" or "my name is X" sets the persona name for the session.
- **RAG Deduplication:** Fixed double initialization of vector database.
- **Thread Safety:** Splash `set_progress` uses `after()` to avoid Tcl/Tk race conditions.
- **Temp File Cleanup:** Pipeline now clears stale temp files before each run.
- **Wav2Lip Noise Suppressed:** Subprocess output hidden; FutureWarning silenced.
- **Whisper Auto-Detect:** Speech-to-text now auto-detects spoken language instead of forcing output language.
- **Video Loop Guard:** `_play_video` stops `after` callbacks when window is destroyed.
- **Splash Loading:** Splash now waits until all models (Whisper, RAG, XTTS) finish loading.
- **Settings UX:** Added "Save Rules" button with `Ctrl+Enter` shortcut and visual feedback.
- **Codebase Cleanup:** Removed legacy `acarrero.py`, translated all Spanish text to English.
- **TTSManager Simplification:** Removed unused `latents_dir` parameter.
- **Bug Fixes:** Fixed Tkinter `TclError` on save, missing `Path` import, incorrect Wav2Lip paths.
