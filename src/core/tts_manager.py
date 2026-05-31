"""
XTTS v2 Voice-Cloning Manager.

Loads the Coqui XTTS v2 model once, computes speaker conditioning latents from
reference audio on first run, then caches them to disk for near-instant reuse.

Usage:
    tts = TTSManager("voices/")
    tts.generate_tts("Hello world", "en", "output.wav")
"""

import os
from pathlib import Path
import torch
import soundfile as sf


class TTSManager:
    """Manages XTTS v2 model lifecycle and cached speaker-latent inference."""

    def __init__(self, voices_dir):
        self.voices_dir = Path(voices_dir)
        self.latents = {}
        self._init_model()
        self.refresh_voices()

    def _init_model(self):
        """Initializes Coqui TTS wrapper and extracts the core XTTS model."""
        os.environ["COQUI_TOS_AGREED"] = "1"
        from TTS.api import TTS as CoquiTTS

        # Load the TTS model wrapper
        tts_wrapper = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
        self.xtts_model = tts_wrapper.synthesizer.tts_model

    def _get_single_latent(self, voice_path: str, latent_path: str) -> dict:
        """Retrieves single speaker latent dictionary (from disk or computed)."""
        p = Path(latent_path)
        if p.exists():
            return torch.load(p)

        return self._compute_and_save_latent(voice_path, p)

    def _compute_and_save_latent(self, voice_path: str, latent_path: Path) -> dict:
        """Computes speaker conditioning latents and saves them to disk."""
        gpt_cond_latent, speaker_embedding = self.xtts_model.get_conditioning_latents(
            audio_path=[voice_path]
        )
        latent_dict = {
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding
        }
        torch.save(latent_dict, latent_path)
        return latent_dict

    def generate_tts(self, text: str, language: str, output_path: str):
        """Synthesise speech from text using cached speaker latents.

        Args:
            text: Text string to vocalise.
            language: Language code ('es' or 'en').
            output_path: Path for the resulting 24 kHz WAV file.
        """
        if language not in self.latents:
            raise ValueError(f"Language '{language}' not configured in TTSManager.")

        lang_latents = self.latents[language]
        with torch.inference_mode():
            out = self.xtts_model.inference(
                text=text,
                language=language,
                gpt_cond_latent=lang_latents["gpt_cond_latent"],
                speaker_embedding=lang_latents["speaker_embedding"]
            )
        sf.write(output_path, out["wav"], 24000)

    def refresh_voices(self):
        """Load only .wav files existing in data/voices/ folder."""
        self.latents = {}
        for wav_path in self.voices_dir.glob("*.wav"):
            lang = wav_path.stem
            latent_path = wav_path.with_suffix(".pth")
            try:
                self.latents[lang] = self._get_single_latent(str(wav_path), latent_path)
            except Exception as e:
                print(f"Error loading voice {lang}: {e}")
