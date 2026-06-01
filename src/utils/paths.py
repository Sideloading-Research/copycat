"""Centralised path definitions for the Copycat project.

All file-system paths used by the application are defined here so that
other modules only ever import ``PATHS["key"]`` instead of hard-coding
directory traversal.
"""

from pathlib import Path

# Project root (three levels up from this file: utils/paths.py → src/ → root).
ROOT = Path(__file__).resolve().parent.parent.parent

SRC = ROOT / "src"
DATA = ROOT / "data"

WAV2LIP_DIR = SRC / "Wav2Lip"

# Journal: plain-text .md files that serve as the chatbot's long-term memory.
JOURNAL = DATA / "journal"
# Voice clips (one .wav per language, file name = language code).
VOICES = DATA / "voices"
# Chroma vector database (persisted on disk).
CHROMA_DB = DATA / "vector_db"
# Behaviour/personality prompt (free-form text).
BEHAVIOR_FILE = DATA / "behavior" / "behavior.txt"

# Per-chat execution logs (appended on every pipeline run).
LOGS = DATA / "logs"
LOGS.mkdir(exist_ok=True)

# Temporary working directory (created on import if missing).
TMP = ROOT / "tmp"
TMP.mkdir(exist_ok=True)

PATHS = {
    "journal": JOURNAL,
    "vector_db": CHROMA_DB,
    "logs_dir": LOGS,
    "icon_img": SRC / "assets" / "icon.png",
    "voices_dir": VOICES,
    "voice_es": VOICES / "es.wav",
    "voice_en": VOICES / "en.wav",
    "behavior": BEHAVIOR_FILE,
    "face_img": DATA / "picture" / "face.jpeg",
    "wav2lip_pth": WAV2LIP_DIR / "checkpoints" / "wav2lip.pth",
    # Temp files (re-created on every pipeline run).
    "tmp_user": TMP / "_tmp_user.wav",
    "tmp_bot": TMP / "_tmp_bot.wav",
    "tmp_video": TMP / "_tmp_lip.mp4",
    "wav2lip_script": WAV2LIP_DIR / "inference.py",
}
