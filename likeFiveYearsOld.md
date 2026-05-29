# Copycat — Explained Like You're Five

## The Lego Project

Copycat is a program that **clones you** to talk on your behalf.
You say something, it searches your diary, thinks of a reply,
and answers back with **your face and your voice** moving your lips.

---

## The Pipeline — Step by Step

```
[1] MICROPHONE    →  turns air into electricity
       │                  (your recorded voice)
       ↓
[2] Whisper        →  turns sound into text (STT)
       │                  (Speech-To-Text)
       ↓
[3] ChromaDB       →  searches your diary for the most
       │                  relevant context (RAG)
       ↓
[4] Ollama LLM     →  reads context + your question
       │                  and writes a short answer
       ↓
[5] Pocket-TTS     →  clones your voice and speaks
        │                  the answer (Text-To-Speech)
       ↓
[6] Wav2Lip        →  moves your photo's lips
       │                  to sync with the audio
       ↓
[7] SPEAKERS       →  talks back with your face and voice
```

---

## Each Piece and Why It's Here

| Piece | What it does | Why this one? | Alternatives you could use |
|-------|-------------|---------------|----------------------------|
| **Whisper tiny** | Turns your voice into text | The smallest → runs fast on CPU | `whisper base/small` (more accurate but slower), `faster-whisper` (same model, 2-3x faster CPU), `speechbrain`, `vosk` |
| **ChromaDB** | Stores your diary as numerical "memory" for searching | Vector database. Stores sentences as numbers to find "the most similar" | `FAISS` (faster, no automatic persistence), `Qdrant`, `Pinecone` (online, not free), `sqlite-vss` |
| **all-MiniLM-L6-v2** | Turns text into numbers (vectors) | The lightest embedding model that works well (80 MB, CPU) | `all-mpnet-base-v2` (better quality, 3x slower), `e5-small`, `bge-small` |
| **Ollama + Qwen2.5:3b** | Reads diary context + question and writes an answer | Ollama manages the model. Qwen2.5:3b is small (3B params) but bilingual | `llama3.2:3b`, `phi3:mini` (faster), `mistral:7b` (better but slower), `gemma2:2b` |
| **Pocket-TTS** | Clones your voice and says the answer | Only 100M params (fits on CPU, 1 second per sentence). Clones from 5 seconds of audio | `XTTS v2` (bigger, slower), `bark` (robotic), `elevenlabs` (online, not free) |
| **Wav2Lip** | Moves your photo's lips | The open-source standard for lip-sync with a single photo | `wav2lip-hq` (better quality), `sadtalker` (more modern, more resources) |
| **CustomTkinter** | Draws the window with buttons, avatar, text | Modernised Tkinter with dark theme. Ships with Python | `PyQt6`, `DearPyGui`, `PyWebView` (for web interface) |

---

## What's Fast and What's Slow?

| Step | Typical time | Reasonable? |
|------|-------------|-------------|
| Whisper tiny (STT) | 1-3 seconds | ✅ Very fast |
| ChromaDB (search) | < 0.5 seconds | ✅ Instant |
| Ollama (response) | 5-15 seconds | ✅ Acceptable |
| **Pocket-TTS (voice clone)** | **1-2 seconds** | ✅ **No longer the bottleneck** |
| Wav2Lip (sync lips) | 15-25 seconds | 🐢 Now it's the slowest |

**We used to use XTTS v2**, which took 30-90 seconds to clone your voice.
It had a GPT-2 inside generating audio token by token, like a chat
writing letter by letter. On CPU it was inevitably slow.

**Now we use Pocket-TTS**, a tiny model (100M params) built
specifically for CPU. It clones your voice from 5 seconds of audio
and generates a sentence in 1-2 seconds. That's 30-50 times faster than XTTS v2.

For voice cloning to work, you need:
1. Accept the terms at https://huggingface.co/kyutai/pocket-tts
2. Install the HuggingFace CLI: `curl -LsSf https://hf.co/cli/install.sh | bash`
3. Log in: `hf auth login`

Without logging in, Pocket-TTS uses preset voices (like "Lola" for Spanish
or "Alba" for English) that sound good but are NOT your voice.

---

## Important: STT ≠ TTS

**Whisper** (STT) and **Pocket-TTS** (TTS) are two completely different things:

```
Whisper     → WRITES what it hears   (speech → text)   → STT
Pocket-TTS  → SPEAKS what it reads   (text → speech)   → TTS
```

**faster-whisper** does NOT clone voices. It's just a faster version of Whisper.
Where Whisper takes 3s to transcribe, faster-whisper takes 1s.
But it has nothing to do with cloning your voice — that's what Pocket-TTS does.

---

## The Key Optimisation Trick

AI models on CPU spend their time multiplying huge tables of numbers
(matrix operations). That's where these settings come from:

| Variable | What it does |
|----------|-------------|
| `OMP_NUM_THREADS=6` | Uses 6 out of 8 cores for matrix multiplication |
| `MKL_NUM_THREADS=6` | Same for Intel MKL (if PyTorch uses it) |
| `KMP_BLOCKTIME=0` | Threads sleep when done → no fighting |
| `KMP_AFFINITY=compact` | Each thread pinned to a fixed core → better cache |

Setting 6 on an 8-core CPU leaves 2 cores free so the operating system,
sound, and GUI don't fight with the AI.

---

## Useful Searches to Find Alternatives Yourself

| You need | Search on Google |
|----------|-----------------|
| Faster STT on CPU | "faster-whisper vs whisper CPU benchmark 2025" |
| Lightweight embeddings | "best lightweight sentence transformers CPU 2025" |
| Offline TTS with Spanish | "Pocket TTS vs XTTS v2 CPU benchmark 2026" |
| Lip-sync with photo | "sadTalker vs wav2Lip CPU speed comparison" |
| Vector database | "chromaDB vs FAISS comparison 2025" |
| Small fast local LLM | "best small LLM ollama CPU 2025 reddit" |

The golden rule: always search `"X vs Y CPU benchmark 2025"`
and check how much RAM each one needs. With 14 GB you can't have two models
that each weigh 6 GB.

---

## The Program Viewed from Above

```
app.py (the boss)
│
├── On startup (in background)
│   ├── Whisper (STT) ──── IN PARALLEL ──── Embeddings (RAG)
│   ├── ChromaDB (diary index)
│   └── Pocket-TTS (the little guy, 500 MB)
│
├── When you press a button
│   ├── Records audio from microphone
│   └── Launches pipeline in a separate thread:
│       ├── Whisper  →  text
│       ├── ChromaDB →  diary context
│       ├── Ollama   →  reply
│       ├── Pocket-TTS  →  audio .wav file
│       ├── Wav2Lip  →  video .mp4 with synced lips
│       └── Play audio + video
│
└── GUI (CustomTkinter)
    ├── Avatar (your photo or the lip-synced video)
    ├── Status bar (green = ready, red = error)
    ├── Log box (text showing what's happening)
    └── Buttons: Escucha (Spanish) / Listen (English)
```

---

## Changelog

### 2026-05-29 — From XTTS v2 to Pocket-TTS

| Change | Why |
|--------|-----|
| **XTTS v2 → Pocket-TTS** | XTTS weighed 6 GB and took 30-90s. Pocket-TTS is 100M params, takes 1-2s. Runs 6× faster than real-time on CPU |
| **openai-whisper → faster-whisper** | Same Whisper tiny model but 4× faster on CPU (uses CTranslate2 instead of PyTorch) |
| **MeloTTS + OpenVoice v2 removed** | No longer need the two-step pipeline. Pocket-TTS clones voice directly from the reference audio |
| **OMP_NUM_THREADS=6** | Leave 2 cores free so the system doesn't fight with the AI |
| **langchain-community removed** | It was sunset (archived). Replaced with `doc_loader.py`, a 50-line standalone replacement |
| **Voice pre-caching in .safetensors** | First run extracts voice from .wav (slow), saves as .safetensors, subsequent runs load instantly |

### Things We Tried That Didn't Work

| Attempt | Problem |
|---------|---------|
| OpenVoice v2 with `vad=True` | MeloTTS dummy audio was too short (1.9s) and VAD failed |
| OpenVoice v2 with `vad=False` | Used Whisper medium on CUDA (no GPU available) |
| Precomputed embeddings `en-default.pth` | MeloTTS EN_V2 model has no "EN-Default" speaker, so the tone didn't match |
| `enable_watermark=False` in OpenVoice | Parameter didn't exist in the installed version |

### What's Missing

- **Real voice cloning:** You need to accept terms on HuggingFace and log in with `hf auth login`. Without it, it uses preset voices (not yours)
- **Wav2Lip is slow:** 15-25s per response. It's the new bottleneck
- **No streaming:** You have to wait for the entire pipeline to finish before hearing the response
