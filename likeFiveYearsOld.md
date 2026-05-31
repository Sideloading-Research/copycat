# Copycat — Explained Like You're Five

## The project built with Lego bricks

Copycat is a program that **clones you** to talk on your behalf.
You say something, it searches your diary, thinks of a reply,
and answers with **your face and your voice** moving the lips.

---

## The pipeline — step by step

```
[1] MICROPHONE     →  turns air into electricity
       │                  (your recorded voice)
       ↓
[2] Whisper        →  turns sound into text (STT)
       ↓
[3] ChromaDB       →  searches your diary for the closest
       │                  match to what you asked (RAG)
       ↓
[4] Ollama LLM     →  reads context + your question
       │                  and writes a short reply
       ↓
[5] XTTS v2        →  clones your voice and speaks
       │                  the reply (Text-To-Speech)
       ↓
[6] Wav2Lip        →  moves the lips of your photo
       │                  to sync with the audio
       ↓
[7] SPEAKERS       →  talks back with your face and voice
```

---

## Each piece and why it's there

| Piece | What does it do? | Why this one? | Alternatives you could use |
|-------|-----------------|---------------|----------------------------|
| **Whisper tiny** | Turns your voice into text | The smallest → runs fast on CPU | `whisper base/small` (better accuracy but slower), `faster-whisper` (same model, 2-3x faster on CPU), `speechbrain`, `vosk` |
| **ChromaDB** | Stores your diary as numerical "memory" for searching | Vector database. Stores sentences as numbers to find "the closest match" | `FAISS` (faster, no automatic persistence), `Qdrant`, `Pinecone` (online, not free), `sqlite-vss` |
| **all-MiniLM-L6-v2** | Turns text into numbers (vectors) | The lightest embedding model that works well (80 MB, CPU) | `all-mpnet-base-v2` (better quality, 3x slower), `e5-small`, `bge-small` |
| **Ollama + Qwen2.5:3b** | Reads diary context + question and writes a reply | Ollama handles the model. Qwen2.5:3b is small (3B params) but bilingual | `llama3.2:3b`, `phi3:mini` (faster), `mistral:7b` (better but slower), `gemma2:2b` |
| **XTTS v2** | Clones your voice and speaks the reply | The only open-source one that sounds good and supports Spanish+English | `bark` (sounds robotic), `styletts2` (English only), `elevenlabs` (online, not free) |
| **Wav2Lip** | Moves the lips of your photo | The open-source standard for lip-sync with a single photo | `wav2lip-hq` (better quality), `sadtalker` (more modern, more demanding) |
| **CustomTkinter** | Draws the window with buttons, avatar, text | Modernized Tkinter with dark theme. Ships with Python | `PyQt6`, `DearPyGui`, `PyWebView` (for web interface) |

---

## What's slow and what's fast

| Step | Typical time | Reasonable? |
|------|-------------|-------------|
| Whisper tiny (STT) | 1-3 seconds | ✅ Very fast |
| ChromaDB (search) | < 0.5 seconds | ✅ Instant |
| Ollama (response) | 5-15 seconds | ✅ Acceptable |
| **XTTS v2 (voice clone)** | **30-90 seconds** | 🐢 **The bottleneck** |
| Wav2Lip (lip sync) | 15-25 seconds | 🐢 Acceptable |

**XTTS v2 is NOT bad.** It's the best offline voice cloner available.
But it has a GPT-2 inside that generates audio token by token,
like a chat typing letter by letter. On CPU that's unavoidably slow.

To speed it up you'd need:
- A **GPU** (NVIDIA with 6GB+ VRAM) → XTTS runs 10x faster
- Or accept **lower voice quality** with another model

---

## Important clarification: STT ≠ TTS

**Whisper** (STT) and **XTTS** (TTS) are two completely different things:

```
Whisper     → WRITES what it hears   (speech → text)   → STT
XTTS v2     → SPEAKS what it reads   (text → speech)   → TTS
```

**faster-whisper** does NOT clone voices. It's just a faster version of Whisper.
Where Whisper takes 3s to transcribe, faster-whisper takes 1s.
But it has nothing to do with cloning your voice — that's what XTTS v2 does.

**If you switched to faster-whisper:** transcription would be faster,
but voice cloning (XTTS) would still take the same time.

---

## The key optimization trick

AI models on CPU spend their time multiplying huge tables of numbers
(matrix operations). That's where these settings come from:

| Variable | What does it do? |
|----------|-----------------|
| `OMP_NUM_THREADS=6` | Uses 6 out of 8 cores for matrix multiplication |
| `MKL_NUM_THREADS=6` | Same for Intel MKL (if PyTorch uses it) |
| `KMP_BLOCKTIME=0` | Threads sleep when done → no fighting |
| `KMP_AFFINITY=compact` | Each thread pinned to a fixed core → better cache |

Setting 6 on an 8-core CPU leaves 2 cores free so the operating system,
audio, and GUI don't compete with the AI.

---

## Useful searches to find alternatives yourself

| You need | Search on Google |
|----------|-----------------|
| Faster STT on CPU | "faster-whisper vs whisper CPU benchmark 2025" |
| Lightweight embeddings | "best lightweight sentence transformers CPU 2025" |
| Offline TTS that speaks Spanish | "XTTS v2 alternatives offline TTS spanish 2025" |
| Lip-sync with photo | "sadTalker vs wav2Lip CPU speed comparison" |
| Vector database | "chromaDB vs FAISS comparison 2025" |
| Small fast local LLM | "best small LLM ollama CPU 2025 reddit" |

The golden rule: always search `"X vs Y CPU benchmark 2025"`
and check how much RAM each one needs. With 14 GB you can't run two models
that weigh 6 GB each.

---

## The program seen from above

```
app.py (the boss)
│
├── On startup (in background)
│   ├── Whisper (STT) ──── IN PARALLEL ──── Embeddings (RAG)
│   ├── ChromaDB (diary index)
│   └── XTTS v2 (the fat one, 6 GB)
│
├── When you press a button
│   ├── Records audio from microphone
│   └── Launches pipeline in a separate thread:
│       ├── Whisper  →  text
│       ├── ChromaDB →  diary context
│       ├── Ollama   →  response
│       ├── XTTS v2  →  .wav audio file
│       ├── Wav2Lip  →  .mp4 video with synced lips
│       └── Play audio + video
│
└── GUI (CustomTkinter)
    ├── Avatar (your photo or the lip-sync video)
    ├── Status bar (green = ready, red = error)
    ├── Log box (text of what's happening)
    └── Buttons: Escucha (Spanish) / Listen (English)
```
