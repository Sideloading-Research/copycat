"""Central configuration for the Copycat application.

All tunable parameters live here so that no other module
hard-codes model names, chunk sizes, thread counts, etc.
Import the singleton ``cfg`` and use its attributes::

    from src.config import cfg
    print(cfg.llm_model)
"""

from dataclasses import dataclass
from src.utils.paths import PATHS


# Supported languages (ISO 639-1 code -> English name).
# Only codes with a matching ``data/voices/<code>.wav`` are used at runtime.
ALL_LANGS: dict[str, str] = {
    "en": "English", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "pt": "Portuguese",
    "nl": "Dutch", "ru": "Russian", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese", "ar": "Arabic",
    "hi": "Hindi", "tr": "Turkish", "pl": "Polish",
    "sv": "Swedish", "da": "Danish", "fi": "Finnish",
    "cs": "Czech", "hu": "Hungarian", "ro": "Romanian",
    "el": "Greek", "he": "Hebrew", "th": "Thai",
    "vi": "Indonesian", "id": "Indonesian", "ms": "Malay",
    "uk": "Ukrainian", "ca": "Catalan", "gl": "Galician",
    "eu": "Basque", "bn": "Bengali", "ta": "Tamil",
    "te": "Telugu", "mr": "Marathi", "gu": "Gujarati",
}


@dataclass
class Config:
    # ── LLM ─────────────────────────────────────────────────────
    llm_model: str = "gemma3:4b"
    llm_threads: int = 2
    llm_host: str = "http://localhost:11434"   # Ollama default

    # ── STT (Whisper) ───────────────────────────────────────────
    stt_model: str = "tiny"
    stt_fp16: bool = False        # CPU-only → no half-precision
    stt_sample_rate: int = 16000
    stt_silence_threshold: float = 0.02   # RMS below this = silence, skip Whisper

    # ── TTS (XTTS v2) ──────────────────────────────────────────
    tts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts_threads: int = 2
    tts_output_sample_rate: int = 24000

    # ── RAG (ChromaDB + sentence-transformers) ──────────────────
    rag_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 100
    rag_k: int = 3                  # top-k general search
    rag_priority_k: int = 1         # chunks forced from priority source
    rag_max_chars: int = 2000       # total context truncation
    rag_priority_max_chars: int = 3000  # context limit with priority

    # ── Lip-sync (Wav2Lip) ──────────────────────────────────────
    lipsync_threads: int = 6
    lipsync_nosmooth: bool = True

    # ── Location / geo ──────────────────────────────────────────
    internet_cache_ttl: float = 60.0
    geo_cache_ttl: float = 3600.0

    # ── Pipeline ────────────────────────────────────────────────
    max_behavior_summary_chars: int = 500
    prompt_rules: str = (
        "- Never mention AI, language model, or programming.\n"
        "- Reply in {out_lang}.\n"
        "- Keep it under 40 words.\n"
        "- Be natural and conversational."
    )

    # ── Thread limits (global env vars) ─────────────────────────
    omp_num_threads: str = "2"
    mkl_num_threads: str = "2"
    openblas_num_threads: str = "2"
    ollama_num_threads: str = "2"
    kmp_blocktime: str = "0"
    kmp_affinity: str = "granularity=fine,compact,1,0"

    # ── Paths (derived from PATHS dict) ─────────────────────────
    # Convenience references so callers can write ``cfg.face_img``.
    @property
    def face_img(self): return PATHS["face_img"]

    @property
    def voices_dir(self): return PATHS["voices_dir"]

    @property
    def journal(self): return PATHS["journal"]

    @property
    def vector_db(self): return PATHS["vector_db"]

    @property
    def behavior(self): return PATHS["behavior"]

    @property
    def behavior_name(self): return "00_behavior.md"

    @property
    def tmp_user(self): return PATHS["tmp_user"]

    @property
    def tmp_bot(self): return PATHS["tmp_bot"]

    @property
    def tmp_video(self): return PATHS["tmp_video"]

    @property
    def wav2lip_script(self): return PATHS["wav2lip_script"]

    @property
    def wav2lip_pth(self): return PATHS["wav2lip_pth"]

    @property
    def logs_dir(self): return PATHS["logs_dir"]


# Singleton — import ``cfg`` everywhere instead of instantiating Config.
cfg = Config()
