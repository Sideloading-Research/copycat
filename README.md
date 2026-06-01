# Copycat — Local AI Avatar

An **offline, CPU-only** personal AI avatar that looks like you, speaks with your cloned voice, and answers using your diary as memory.

**Pipeline:** Mic → Whisper (STT) → ChromaDB (RAG memory) → Ollama (LLM) → XTTS v2 (voice clone) → Wav2Lip (lip-sync) → Avatar video

---

## Quick Start

```bash
git clone https://github.com/Sideloading-Research/copycat
cd copycat
bash setup.sh          # ~15 min (models download)
# Prepare your assets (see below)
./run.sh
```

### Assets you must provide

| File | Purpose | How |
|------|---------|-----|
| `data/picture/face.jpeg` | Avatar portrait (256×256 min, no glasses, good lighting) | Crop a selfie to square |
| `data/voices/es.wav` | Spanish voice reference (6-15 s, clean audio) | Record: _"Hoy fue un día bastante productivo..."_ |
| `data/voices/en.wav` | English voice reference (6-15 s, clean audio) | Record: _"I've been thinking a lot about how quickly things change..."_ |
| `data/journal/*.md` | Diary entries — your memory (RAG knowledge base) | One `.md` file per day |
| `data/behavior/behavior.txt` | Personality definition for the LLM to role-play | Free text describing who you are |

### Requirements

- **CPU:** 4+ cores
- **RAM:** 10+ GB
- **Storage:** 10 GB free
- **OS:** Linux, macOS, Windows

---

## How to use

1. Run `./run.sh` — a splash screen shows loading progress.
2. When the main window opens, **type a message** or press the **microphone button** 🎤 to speak.
3. Select the output language (`EN` / `ES`) with the radio buttons on the right.
4. Say *"me llamo X"* or *"my name is X"* to set your name for the session.
5. The avatar responds with your face and cloned voice, lips synced.

### Controls

| Button | Action |
|--------|--------|
| 🎤 (red) | Start/stop recording |
| ⚙ | Open settings (upload face, voice, diary files; edit personality) |
| Text entry + Enter | Send typed message |

---

## Architecture

```
src/
├── main.py              Entry point: splash → model loading → UI
├── core/
│   ├── engine.py        Pipeline orchestrator (STT→RAG→LLM→TTS→lip-sync)
│   ├── rag.py           ChromaDB + sentence-transformer (incremental indexing)
│   ├── tts_manager.py   XTTS v2 voice cloning with cached speaker latents
│   └── audio.py         Microphone recording + audio playback
├── ui/
│   ├── main_window.py   Chat UI, avatar, controls
│   ├── splash.py        Animated loading screen
│   └── settings.py      Configuration dialog
└── utils/
    ├── paths.py         Centralised path definitions
    └── setup_env.py     CPU thread limits, torch.load compat patch
```

### Pipeline flow

```
Startup: behavior.txt → journal/00_behavior.md (auto-synced)

User speaks/types
  → Whisper (speech-to-text)
  → RAG priority search (1 chunk behavior + 2 diary, max 3000 chars)
  → Build prompt (500-char behavior summary + RAG context + rules)
  → Ollama / gemma3:4b (LLM generates reply)
  → XTTS v2 (voice cloning, 24 kHz WAV)
  → Wav2Lip (lip-sync on face photo → MP4)
  → Play audio + show video
  → Save stats to data/logs/chats.jsonl
```

### Incremental RAG

The vector DB tracks changes via per-file MD5 hashes (`.file_hashes.json`):

- **New file** → only that file is chunked and added.
- **Modified file** → old vectors deleted, new ones added.
- **Deleted file** → vectors removed.
- **No changes** → instant load (no re-indexing).

On each restart, `behavior.txt` is copied to `journal/00_behavior.md`. RAG detects the hash change and re-indexes it automatically.

### Per-chat logging

Every pipeline run appends a JSON line to `data/logs/chats.jsonl`:

```json
{"timestamp":"...","lang":"es","input_text":"...","output_text":"...",
 "tt_rag_ms":45,"tt_llm_ms":32000,"tt_tts_ms":8500,"tt_lipsync_ms":5200,
 "tt_total_ms":45800,"n_chunks":3,"n_inferences":5}
```

See [likeFiveYearsOld.md](likeFiveYearsOld.md) for a deep-dive explanation.

---

## Customization

| Change | How |
|--------|-----|
| Personality | Edit `data/behavior/behavior.txt` or use Settings → Save Rules |
| LLM model | Change `model=` in `src/core/engine.py:162` |
| Voice samples | Replace `data/voices/{lang}.wav` |
| Face photo | Replace `data/picture/face.jpeg` |
| Knowledge base | Add `.md` files to `data/journal/` (auto-indexed on next start) |

### Prompt structure

1. **`behavior.txt`** — personality definition. A 500-char summary is in the prompt; the full file is indexed in RAG via `journal/00_behavior.md` (auto-synced on every restart).
2. **RAG context** — retrieves 1 priority chunk from behavior + 2 from diary entries (ChromaDB). Total up to 3000 chars.
3. **Name detection** — *"me llamo X"* or *"my name is X"* sets the persona name.

The system prompt instructs the model to never reveal it is an AI and to reply in first person.

---

## Optional performance tuning

### Thread limits (already built-in)

Copycat sets these automatically on startup to prevent CPU oversaturation:

| Variable | Value | Why |
|----------|-------|-----|
| `OMP_NUM_THREADS` | 2 | Prevents thermal throttling |
| `OLLAMA_NUM_THREADS` | 2 | LLM uses 2 cores instead of all 8 |
| `KMP_BLOCKTIME` | 0 | Threads sleep immediately |

### CPU frequency governor (Linux only)

If the laptop overheats during model loading:

```bash
# Before launching Copycat — lock to 2 GHz conservative
for i in {0..7}; do
  sudo cpufreq-set -c $i -u 2.00GHz -g conservative
done

# After loading — restore normal scaling
for i in {0..7}; do
  sudo cpufreq-set -c $i -g schedutil
done
```

Requires `cpufrequtils` (`sudo apt install cpufrequtils`) and passwordless `sudo`.

### Smaller LLM

```bash
ollama pull gemma3:1b    # 815 MB, faster but less capable
# Then edit model name in src/core/engine.py
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| _"Failed to load audio: No such file"_ | Ensure `tmp/` directory exists (created automatically on first import) |
| Wav2Lip crashes | Check that `src/Wav2Lip/checkpoints/wav2lip.pth` exists |
| No microphone detected | `python -c "import sounddevice; print(sounddevice.query_devices())"` |
| ChromaDB error | Delete `data/vector_db/` and restart (will rebuild) |
| No voice output | Ensure `data/voices/` has at least one `.wav` file |
| LLM very slow | Check RAM usage — if swapping, use a smaller model (`gemma3:1b`) |

---

## Tech Stack

| Component | Model |
|-----------|-------|
| STT | [Whisper tiny](https://github.com/openai/whisper) |
| RAG embedding | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Vector DB | [ChromaDB](https://www.trychroma.com/) |
| LLM | [gemma3:4b](https://ollama.com/library/gemma3:4b) via [Ollama](https://ollama.com/) |
| TTS | [XTTS v2](https://github.com/coqui-ai/TTS) |
| Lip-sync | [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) |
| GUI | [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) |

---

## Roadmap / TODO

### Diary raw search
Beyond RAG (vector similarity), add a **literal full-text search** mode that reads diary entries directly (grep / keyword match) and injects the matching passage verbatim into the prompt. This gives the LLM access to exact quotations the user wrote, not just semantically similar fragments.

### Modular architecture (planned)
Current `CopycatEngine` is tightly coupled to every component. Future refactoring:

- **Abstract interfaces** for STT, LLM, TTS, VectorDB, LipSync — swap any backend without editing engine.py
- **`PipelineOrchestrator`** as the single coordinator, with STT/LLM/TTS/etc injected via constructor
- **`PromptBuilder`** extracted from engine.py — builds the persona prompt from pluggable sections
- **`Config` dataclass** with all hardcoded values (model names, paths, thread counts, chunk sizes)
- **`PipelineController`** separated from `MainWindow` UI — enables CLI/server mode

### Other ideas
- Conversation memory (sliding window of last N exchanges)
- Model unload / reload to free RAM between turns
- Voice activity detection (VAD) for automatic recording stop
- Hybrid RAG (dense + sparse BM25 retriever)
- Progress bar during Wav2Lip (currently silent for ~5s)
- Language auto-detection for output voice (no manual EN/ES toggle)
- GPU acceleration detection + fallback
