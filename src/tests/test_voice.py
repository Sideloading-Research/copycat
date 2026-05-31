"""
test_voice.py – Quick voice clone verification for Copycat.

Generates two short audio files using XTTS v2:
  - test_es.wav  (cloned from voices/es.wav, text in Spanish)
  - test_en.wav  (cloned from voices/en.wav, text in English)

Then plays each one so you can judge quality subjectively.

Usage (inside venv):
    python3 test_voice.py
"""

import os
import sys


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(BASE_DIR, "voices")

ES_REF = os.path.join(VOICES_DIR, "es.wav")
EN_REF = os.path.join(VOICES_DIR, "en.wav")

TEST_ES_OUT = os.path.join(BASE_DIR, "test_es.wav")
TEST_EN_OUT = os.path.join(BASE_DIR, "test_en.wav")

# Short sentences that exercise different phonemes.
TEST_TEXT_ES = (
    "Hola, soy tu asistente personal. "
    "Hoy el clima es agradable y me siento con mucha energía."
)
TEST_TEXT_EN = (
    "Hello, I am your personal assistant. "
    "Today the weather is pleasant and I feel very energetic."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_reference_files() -> None:
    """Abort early if reference wav files are missing."""
    missing = [p for p in (ES_REF, EN_REF) if not os.path.isfile(p)]
    if missing:
        print("[ERROR] Missing reference files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)


def load_tts_model():
    """Load XTTS v2 model (downloads ~1.87 GB on first run)."""
    print("[INFO] Loading XTTS v2 model (first run may take several minutes)…")
    from TTS.api import TTS  # noqa: PLC0415  imported here to delay heavy import
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    print("[INFO] Model loaded.")
    return model


def synthesize(model, text: str, speaker_wav: str, language: str, out_path: str) -> None:
    """Generate a wav file cloning the given speaker reference."""
    print(f"[INFO] Synthesising [{language}]: \"{text[:60]}…\"")
    model.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
        file_path=out_path,
    )
    size_kb = os.path.getsize(out_path) // 1024
    print(f"[OK]  Saved: {out_path}  ({size_kb} KB)")


def play_wav(path: str) -> None:
    """Play a wav file using sounddevice + soundfile."""
    import sounddevice as sd  # noqa: PLC0415
    import soundfile as sf    # noqa: PLC0415

    data, rate = sf.read(path, dtype="float32")
    print(f"[PLAY] {os.path.basename(path)}  ({len(data)/rate:.1f} s) …")
    sd.play(data, rate)
    sd.wait()
    print("[PLAY] Done.")


def prompt_play(path: str) -> None:
    """Ask user before playing (they may prefer to open files manually)."""
    answer = input(f"\nPlay {os.path.basename(path)}? [y/N] ").strip().lower()
    if answer == "y":
        play_wav(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Copycat – Voice Clone Test")
    print("=" * 60)

    check_reference_files()

    model = load_tts_model()

    # --- Spanish ---
    print("\n--- Spanish clone ---")
    synthesize(model, TEST_TEXT_ES, ES_REF, "es", TEST_ES_OUT)

    # --- English ---
    print("\n--- English clone ---")
    synthesize(model, TEST_TEXT_EN, EN_REF, "en", TEST_EN_OUT)

    # --- Playback ---
    print("\n" + "=" * 60)
    print("  Synthesis complete. Listening tests:")
    print("=" * 60)
    prompt_play(TEST_ES_OUT)
    prompt_play(TEST_EN_OUT)

    print("\n[DONE] Files saved:")
    print(f"  {TEST_ES_OUT}")
    print(f"  {TEST_EN_OUT}")
    print("\nIf the clone sounds right, the voices/ files are ready for main.py.")
    print("If quality is poor, re-record with less background noise (≥10 s).")


if __name__ == "__main__":
    main()
