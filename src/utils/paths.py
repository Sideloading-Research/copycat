from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data"
JOURNAL = DATA / "journal"
VOICES = DATA / "voices"
CHROMA_DB = DATA / "vector_db"
# Definimos el archivo TXT dentro de la carpeta
BEHAVIOR_FILE = DATA / "behavior" / "behavior.txt"

TMP = ROOT / "tmp"
TMP.mkdir(exist_ok=True)

PATHS = {
    "journal": JOURNAL,
    "vector_db": CHROMA_DB,
    "voices_dir": VOICES, # Carpeta para escaneo
    "behavior": BEHAVIOR_FILE, # El archivo de texto
    "face_img": DATA / "picture" / "face.jpeg",
    "wav2lip_pth": DATA / "checkpoints" / "wav2lip.pth",
    "tmp_user": TMP / "_tmp_user.wav",
    "tmp_bot": TMP / "_tmp_bot.wav",
    "tmp_video": TMP / "_tmp_lip.mp4",
    "wav2lip_script": ROOT / "Wav2Lip" / "inference.py"
}