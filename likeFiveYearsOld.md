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

## Thread tuning (already built-in)

AI models on CPU spend their time multiplying huge tables of numbers
(matrix operations). Copycat already sets these for you on startup:

| Variable | Value | Why |
|----------|-------|-----|
| `OMP_NUM_THREADS` | 2 | Prevents thermal throttling on small laptops |
| `MKL_NUM_THREADS` | 2 | Same for Intel Math Kernel Library |
| `KMP_BLOCKTIME` | 0 | Threads sleep immediately → less fighting |
| `KMP_AFFINITY` | compact | Each thread sticks to one core → better cache |
| `OLLAMA_NUM_THREADS` | 2 | LLM uses 2 cores instead of all 8 |

On an 8-core laptop, 2 threads per model leaves 6 cores free for the OS,
audio drivers, and GUI, preventing audio crackling and UI freezes.

## Optional: CPU frequency governor (Linux only)

If your laptop overheats during model loading (fans go crazy), you can pin
the CPU to a lower speed with these commands **before** launching Copycat:

```bash
for i in {0..7}; do
  sudo cpufreq-set -c $i -u 2.00GHz -g conservative
done
```

After Copycat finishes loading, restore normal speed:

```bash
for i in {0..7}; do
  sudo cpufreq-set -c $i -g schedutil
done
```

This is **optional** — Copycat doesn't touch your CPU governor settings.
You need `cpufrequtils` installed (`sudo apt install cpufrequtils`) and
passwordless `sudo` for the commands above.

---

---

## How the memory (vector DB) learns new things without starting over

Imagine you have a **giant box of Lego bricks** that represents everything
you know (your journal).  To answer questions, the program turns every
paragraph into a special number-code (a *vector*) and stores it in a
second box called ChromaDB.

**Before (old way — slow):** Every time you added ONE new page to your
diary, the program would:
1. Dump ALL the Lego bricks on the floor.
2. Turn EVERY page into number-codes again.
3. Put everything back in the box.

Even if you only added one sentence, it re-read all 100 pages.
That's like re-packing your whole suitcase because you added one sock.

**Now (new way — fast):** The program keeps a **cheat-sheet** that says
*"this page → this code"* for every page (it's a tiny file called
``.file_hashes.json``).  When you start the program:

1. It looks at each page of your diary and checks its cheat-sheet.
2. If the page is **new**: it turns only that page into number-codes
   and adds it.  *(Like adding that one sock to the suitcase.)*
3. If the page **changed**: it throws away the old codes for *that page*
   and does it again.  *(Like taking the old sock out and putting a
   new one in.)*
4. If the page is **exactly the same**: it does nothing — skip!
   *(You don't even open the suitcase.)*
5. If a page **disappeared**: it throws away its codes.  *(Like taking
   a sock out.)*

**The cheat-sheet trick works because of maths:** each page's content
is turned into a short fingerprint (an MD5 hash — a 32-character code
like ``"a1b2c3..."``).  If the content changes *even by one letter*,
the fingerprint changes.  So the program knows instantly whether a page
is new, modified, or the same, without reading the whole book.

---

## Does incremental make it slower?

**No — it makes it faster, and never slower.**  Here's why:

- If you **didn't change anything**: the program checks fingerprints
  (takes 0.01 seconds) and says "all good".  Same speed as before.
- If you **added one file**: the program fingerprints all files
  (0.1 seconds for 100 files), finds the one new one, and processes
  *only that one*.  Much faster than rebuilding everything.
- If you **changed every file**: it does the same work as the old way,
  plus 0.1 seconds of fingerprint checking.  **Negligible overhead.**

The fingerprint check is like glancing at a book's cover to see if
it's the same book — it doesn't read the whole book.

**In short: incremental is always better.**  The more files you have,
the bigger the savings when you only add a few.

---

| You need | Search on Google |
|----------|-----------------|
| Faster STT on CPU | "faster-whisper vs whisper CPU benchmark 2025" |
| Lightweight embeddings | "best lightweight sentence transformers CPU 2025" |
| Offline TTS that speaks Spanish | "XTTS v2 alternatives offline TTS spanish 2025" |
| Lip-sync with photo | "sadTalker vs wav2Lip CPU speed comparison" |
| Vector database | "chromaDB vs FAISS comparison 2025" |
| Small fast local LLM | "best small LLM ollama CPU 2025 reddit" |

---

## Why everything slows down when the memory (journal) grows

This is the most important thing to understand about Copycat:

**The RAG search itself does NOT get slower as the journal grows.**
ChromaDB uses approximate nearest-neighbour search — it finds the
top-3 matches out of a million documents in the same time it takes
to find them out of 10 documents.  Cost: O(log N) or better.

**What gets slower is the LLM (Ollama).**  When the RAG returns more
context (more diary pages), the LLM prompt gets longer.  A longer
prompt means:
- More RAM to hold the prompt in memory
- More tokens for the LLM to "read" before generating a reply
- Longer time-to-first-token

Before the fix, the code retrieved **6 chunks** of up to 800
characters each = **4800 characters** of context, with no upper limit.
Over time, as you add more chat history, that context keeps growing.

---

## What we changed (the optimisations)

| Tweak | Before | After | Why |
|-------|--------|-------|-----|
| **RAG chunks (k)** | 6 chunks | 3 chunks | Half the context for the LLM = half the prompt size |
| **Context cap** | Unlimited | 2000 characters | Hard stop so the prompt never bloat |
| **LLM threads** | All 8 cores (default) | 2 threads | Stops the LLM from stealing every CPU core |
| **Prompt wording** | 150+ words | ~50 words | Every word in the prompt costs tokens |
| **Thread limits** | Rag set 4, Wav2Lip set 6, others 2 — fighting each other | All set to 2 globally | No more core-juggling between models |

**Result:** The LLM now generates a reply in about half the time
because it has less text to read and doesn't have to fight 7 other
threads for CPU time.  Memory usage drops because the prompt is
smaller.

---

## The golden rule

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
