"""XTTS v2 voice-cloning manager.

Loads the Coqui XTTS v2 model once, computes speaker-conditioning
latents from reference audio on first use, then caches them to disk
for near-instant reuse on subsequent runs.

Usage::

    tts = TTSManager("data/voices/")
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
        self.latents: dict[str, dict] = {}  # language → latent dict
        self._init_model()
        self.refresh_voices()

    def _init_model(self):
        """Load the Coqui TTS wrapper and extract the underlying XTTS model."""
        os.environ["COQUI_TOS_AGREED"] = "1"
        from TTS.api import TTS as CoquiTTS

        tts_wrapper = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
        self.xtts_model = tts_wrapper.synthesizer.tts_model

    def _get_single_latent(self, voice_path: str, latent_path: str) -> dict:
        """Return speaker latents for one voice, loading from cache if possible."""
        p = Path(latent_path)
        if p.exists():
            return torch.load(p)
        return self._compute_and_save_latent(voice_path, p)

    def _compute_and_save_latent(self, voice_path: str, latent_path: Path) -> dict:
        """Compute speaker-embedding / GPT-cond latents and write them to disk."""
        gpt_cond_latent, speaker_embedding = self.xtts_model.get_conditioning_latents(
            audio_path=[voice_path]
        )
        latent_dict = {
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding,
        }
        torch.save(latent_dict, latent_path)
        return latent_dict

    def generate_tts(self, text: str, language: str, output_path: str):
        """Synthesise speech from *text* using the cached latents for *language*.

        Args:
            text: Text to vocalise.
            language: Language code (``"es"`` or ``"en"``).
            output_path: Destination path for the 24 kHz mono WAV.
        """
        if language not in self.latents:
            raise ValueError(f"Language '{language}' not configured in TTSManager.")

        lang_latents = self.latents[language]
        with torch.inference_mode():
            out = self.xtts_model.inference(
                text=text,
                language=language,
                gpt_cond_latent=lang_latents["gpt_cond_latent"],
                speaker_embedding=lang_latents["speaker_embedding"],
            )
        sf.write(output_path, out["wav"], 24000)

    def refresh_voices(self):
        """Scan ``data/voices/`` and load latents for every ``.wav`` found.

        The file stem (e.g. ``en`` from ``en.wav``) is used as the language key.
        """
        self.latents = {}
        for wav_path in self.voices_dir.glob("*.wav"):
            lang = wav_path.stem
            latent_path = wav_path.with_suffix(".pth")
            try:
                self.latents[lang] = self._get_single_latent(
                    str(wav_path), str(latent_path)
                )
            except Exception as e:
                print(f"Error loading voice {lang}: {e}")
