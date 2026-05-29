"""
Copycat — Local AI Avatar Chatbot with Lip-Sync

Pipeline:
  Whisper tiny (STT) → ChromaDB + RAG → Ollama/Qwen2.5-3B (LLM)
  → OpenVoice v2 (voice cloning TTS) → Wav2Lip-GAN (lip-sync) → Playback

Runs fully offline on CPU (no GPU required).
Customise your avatar by placing:
  - face.jpeg       → a front-facing portrait photo
  - voices/es.wav   → Spanish voice reference (6-15s, clean)
  - voices/en.wav   → English voice reference (6-15s, clean)
  - diario/*.md     → personal diary entries (RAG knowledge base)
"""

import os

# ── CPU THREAD TUNING (MUST be before PyTorch import) ────────────────────
# Ryzen 7 4700U: 8 cores. Reserve 2 cores for GUI/system responsiveness.
# PyTorch, MKL, and Ollama read these env vars at library-init time.
os.environ.setdefault("OMP_NUM_THREADS",         "6")
os.environ.setdefault("MKL_NUM_THREADS",         "6")
os.environ.setdefault("KMP_BLOCKTIME",           "0")
os.environ.setdefault("KMP_AFFINITY",            "granularity=fine,compact,1,0")
os.environ.setdefault("OLLAMA_NUM_THREADS",      "6")

import sys
import subprocess
import gc
import threading
import queue
import torch
# Patch torch.load globally for PyTorch 2.6 weights_only=True default compatibility
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load
torch.set_num_threads(6)
torch.set_num_interop_threads(1)

import tkinter as tk
import customtkinter as ctk
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import ollama
from PIL import Image, ImageTk

# langchain-community sunset 2026-05 — using standalone replacement
from doc_loader import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
DIARIO    = BASE / "diario"
VOICES    = BASE / "voices"
CHROMA_DB = BASE / "chroma_db"
WAV2LIP   = BASE / "Wav2Lip"

VOICE_ES     = str(VOICES / "es.wav")
VOICE_EN     = str(VOICES / "en.wav")
FACE_IMG     = str(BASE / "face.jpeg")
WAV2LIP_PTH  = str(WAV2LIP / "checkpoints" / "wav2lip_gan.pth")

TMP_USER  = str(BASE / "_tmp_user.wav")
TMP_BOT   = str(BASE / "_tmp_bot.wav")
TMP_VIDEO = str(BASE / "_tmp_lip.mp4")

OLLAMA_MODEL    = "qwen2.5:3b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLE_RATE     = 16_000

# ── THEME ──────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

C_IDLE   = "#1f538d"
C_ACTIVE = "#b71c1c"
C_OK     = "#2e7d32"
C_WARN   = "#e65100"
C_ERR    = "#c62828"
C_INFO   = "#0277bd"


class CopycatApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Copycat – Local Avatar")
        self.geometry("480x700")
        self.resizable(False, False)

        # runtime state
        self.recording    = False
        self.active_lang  = None   # "es" | "en"
        self.audio_chunks = []
        self.stream       = None

        # AI handles (loaded async)
        self.whisper_m    = None
        self.tts_manager  = None
        self.vector_store = None

        self._build_ui()
        self._show_static()
        threading.Thread(target=self._load_models, daemon=True).start()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        """Build the main window: avatar display, status, log, language buttons."""
        self.frm_avatar = ctk.CTkFrame(self, width=340, height=340, corner_radius=12)
        self.frm_avatar.pack(pady=18)
        self.lbl_avatar = tk.Label(self.frm_avatar, bg="#121212")
        self.lbl_avatar.pack(expand=True, fill="both", padx=6, pady=6)

        # status
        self.lbl_status = ctk.CTkLabel(
            self, text="Loading AI...",
            font=("Courier New", 12, "bold"), text_color=C_WARN
        )
        self.lbl_status.pack(pady=4)

        # log
        self.log = ctk.CTkTextbox(self, width=430, height=100, state="disabled",
                                   font=("Courier New", 11))
        self.log.pack(pady=6)

        # buttons
        frm_btns = ctk.CTkFrame(self, fg_color="transparent")
        frm_btns.pack(pady=14)

        self.btn_es = ctk.CTkButton(
            frm_btns, text="Escucha", width=190, height=54,
            font=("Arial", 17, "bold"), state="disabled",
            fg_color=C_IDLE,
            command=lambda: self._toggle("es")
        )
        self.btn_es.grid(row=0, column=0, padx=10)

        self.btn_en = ctk.CTkButton(
            frm_btns, text="Listen", width=190, height=54,
            font=("Arial", 17, "bold"), state="disabled",
            fg_color=C_IDLE,
            command=lambda: self._toggle("en")
        )
        self.btn_en.grid(row=0, column=1, padx=10)

    def _show_static(self):
        if not os.path.exists(FACE_IMG):
            ph = np.zeros((320, 320, 3), dtype=np.uint8)
            cv2.putText(ph, "face.jpeg missing", (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            cv2.imwrite(FACE_IMG, ph)
        img = Image.open(FACE_IMG).resize((328, 328))
        itk = ImageTk.PhotoImage(img)
        self.lbl_avatar.configure(image=itk)
        self.lbl_avatar.image = itk

    # ── MODEL LOADING (async daemon thread) ──────────────────────────────────

    def _load_models(self):
        """Load all AI models in background. Embeddings load parallel to Whisper."""
        try:
            # Kick off embedding model load in a parallel thread
            self._status("Loading models…", C_WARN)
            embed_ready = threading.Event()
            t = threading.Thread(target=self._load_embeddings,
                                 args=(embed_ready,), daemon=True)
            t.start()

            # Load Whisper while embeddings load in background
            self._status("Loading Whisper tiny (faster-whisper)…", C_WARN)
            self.whisper_m = WhisperModel("tiny", device="cpu", compute_type="int8")

            # By now embeddings should be ready; init ChromaDB instantly
            embed_ready.wait()
            self._status("Indexing diario/ (RAG)…", C_WARN)
            self._init_rag()

            # Pocket-TTS is lightweight (~100M params, CPU-native) — loads last
            self._status("Loading Pocket-TTS (voice cloner)…", C_WARN)
            from tts_manager import TTSManager
            self.tts_manager = TTSManager(
                voice_es_path=VOICE_ES,
                voice_en_path=VOICE_EN,
            )

            self.after(0, self._enable_btns)
            self._status("Ready. Press Escucha or Listen.", C_OK)
            self._log("[SYSTEM] Models loaded. Pocket-TTS active.")
        except Exception as exc:
            self._status(f"Load error: {exc}", C_ERR)
            self._log(f"[ERROR] {exc}")

    def _load_embeddings(self, ready: threading.Event):
        """Load the sentence-transformer model (runs in parallel thread)."""
        self.embed_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        ready.set()

    def _init_rag(self):
        """Initialise ChromaDB — fast disk-load or first-time full index."""
        DIARIO.mkdir(exist_ok=True)
        if CHROMA_DB.exists() and any(CHROMA_DB.iterdir()):
            self._log("[RAG] Loading persistent Chroma database from disk.")
            self.vector_store = Chroma(
                persist_directory=str(CHROMA_DB),
                embedding_function=self.embed_model
            )
        else:
            self._log("[RAG] Database not found. Indexing diario/ with parallel loader...")
            loader = DirectoryLoader(
                str(DIARIO), glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                silent_errors=True,
                use_multithreading=True,
                show_progress=True,
            )
            docs = loader.load()
            if not docs:
                docs = [Document(page_content="Diary empty. Add .md files to diario/")]

            # Larger chunks = fewer vectors = faster index & search
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks   = splitter.split_documents(docs)

            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embed_model,
                persist_directory=str(CHROMA_DB),
            )
            self._log(f"[RAG] Created database: {len(docs)} files · {len(chunks)} chunks.")

    def _enable_btns(self):
        self.btn_es.configure(state="normal")
        self.btn_en.configure(state="normal")

    def _disable_btns(self):
        self.btn_es.configure(state="disabled")
        self.btn_en.configure(state="disabled")

    # ── AUDIO RECORDING ────────────────────────────────────────────────────────

    def _toggle(self, lang: str):
        if self.recording and self.active_lang == lang:
            self._stop_recording()
        elif not self.recording:
            self._start_recording(lang)

    def _start_recording(self, lang: str):
        self.recording    = True
        self.active_lang  = lang
        self.audio_chunks = []

        stop_lbl = "Detener ■" if lang == "es" else "Stop ■"
        if lang == "es":
            self.btn_es.configure(text=stop_lbl, fg_color=C_ACTIVE)
            self.btn_en.configure(state="disabled")
        else:
            self.btn_en.configure(text=stop_lbl, fg_color=C_ACTIVE)
            self.btn_es.configure(state="disabled")

        label = "Escuchando…" if lang == "es" else "Listening…"
        self._status(label, C_INFO)

        def _cb(indata, frames, time, status):
            self.audio_chunks.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            dtype="float32", callback=_cb
        )
        self.stream.start()

    def _stop_recording(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        lang = self.active_lang
        self.btn_es.configure(text="Escucha", fg_color=C_IDLE, state="disabled")
        self.btn_en.configure(text="Listen",  fg_color=C_IDLE, state="disabled")
        self._status("Processing…", C_WARN)

        audio = np.concatenate(self.audio_chunks, axis=0)
        sf.write(TMP_USER, audio, SAMPLE_RATE)
        threading.Thread(target=self._pipeline, args=(lang,), daemon=True).start()

    # ── AI PIPELINE (STT → RAG → LLM → TTS → Wav2Lip → Play) ────────────────

    def _pipeline(self, lang: str):
        try:
            # 1. STT (faster-whisper ~4x faster on CPU)
            self._status("Transcribing…", C_WARN)
            segments, _ = self.whisper_m.transcribe(TMP_USER, language=lang, beam_size=1)
            user_text   = " ".join(s.text.strip() for s in segments).strip()
            if not user_text:
                self._status("No speech detected.", C_WARN)
                self.after(0, self._enable_btns)
                return
            self._log(f"[{lang.upper()}] You: {user_text}")

            # 2. RAG
            self._status("Querying diary…", C_WARN)
            hits    = self.vector_store.similarity_search(user_text, k=2)
            context = "\n---\n".join(h.page_content for h in hits)

            # 3. LLM
            self._status("Generating response…", C_WARN)
            lang_instr = (
                "Responde siempre en español. Máximo 20 palabras."
                if lang == "es"
                else "Always answer in English. Maximum 20 words."
            )
            prompt = (
                f"{lang_instr}\n"
                f"Diary context:\n{context}\n\n"
                f"Question: {user_text}\nAnswer:"
            )
            resp     = ollama.generate(model=OLLAMA_MODEL, prompt=prompt,
                                         options={"num_ctx": 1024})
            bot_text = resp["response"].strip()
            self._log(f"Bot: {bot_text}")

            # 4. TTS – OpenVoice v2 voice cloning using precomputed latents
            self._status("Generating voice (OpenVoice v2)…", C_WARN)
            self.tts_manager.generate_tts(
                text=bot_text,
                language=lang,
                output_path=TMP_BOT
            )

            # 5. Wav2Lip – lip sync (runs as subprocess to free main memory)
            self._status("Syncing lips…", C_WARN)
            cmd = (
                f"{sys.executable} {WAV2LIP}/inference.py"
                f" --checkpoint_path {WAV2LIP_PTH}"
                f" --face {FACE_IMG}"
                f" --audio {TMP_BOT}"
                f" --outfile {TMP_VIDEO}"
                f" --nosmooth"
            )
            subprocess.run(cmd, shell=True, check=True,
                           env={**os.environ, "OMP_NUM_THREADS": "6"})

            # 6. Play audio + video
            self._status("Responding…", C_OK)
            threading.Thread(target=self._play_audio, daemon=True).start()
            self.after(0, self._play_video)

        except Exception as exc:
            self._status(f"Error pipeline: {exc}", C_ERR)
            self._log(f"[ERROR] {exc}")
            self.after(0, self._enable_btns)
        finally:
            gc.collect()

    def _play_audio(self):
        data, fs = sf.read(TMP_BOT)
        sd.play(data, fs)
        sd.wait()

    def _play_video(self):
        if not os.path.exists(TMP_VIDEO):
            self._show_static()
            self.after(0, self._enable_btns)
            return
        cap = cv2.VideoCapture(TMP_VIDEO)

        def _next():
            ret, frame = cap.read()
            if ret:
                resized = cv2.resize(frame, (328, 328))
                rgba  = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
                img   = Image.fromarray(rgba)
                itk   = ImageTk.PhotoImage(img)
                self.lbl_avatar.configure(image=itk)
                self.lbl_avatar.image = itk
                self.after(40, _next)          # ~25 fps
            else:
                cap.release()
                self._show_static()
                self._status("Ready. Press Escucha or Listen.", C_OK)
                self.after(0, self._enable_btns)

        _next()

    # ── HELPERS ────────────────────────────────────────────────────────────────

    def _status(self, text: str, color: str):
        """Update the status bar (thread-safe via after())."""
        self.after(0, lambda: self.lbl_status.configure(text=text, text_color=color))

    def _log(self, text: str):
        def _upd():
            self.log.configure(state="normal")
            self.log.insert("end", text + "\n")
            self.log.see("end")
            self.log.configure(state="disabled")
        self.after(0, _upd)


if __name__ == "__main__":
    app = CopycatApp()
    app.mainloop()
