from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data"
WAV2LIP_DIR = SRC / "Wav2Lip"
JOURNAL = DATA / "journal"
VOICES = DATA / "voices"
CHROMA_DB = DATA / "vector_db"
BEHAVIOR_FILE = DATA / "behavior" / "behavior.txt"

TMP = ROOT / "tmp"
TMP.mkdir(exist_ok=True)

PATHS = {
    "journal": JOURNAL,
    "vector_db": CHROMA_DB,
    "voices_dir": VOICES,
    "voice_es": VOICES / "es.wav",
    "voice_en": VOICES / "en.wav",
    "behavior": BEHAVIOR_FILE,
    "face_img": DATA / "picture" / "face.jpeg",
    "wav2lip_pth": WAV2LIP_DIR / "checkpoints" / "wav2lip.pth",
    "tmp_user": TMP / "_tmp_user.wav",
    "tmp_bot": TMP / "_tmp_bot.wav",
    "tmp_video": TMP / "_tmp_lip.mp4",
    "wav2lip_script": WAV2LIP_DIR / "inference.py"
}
