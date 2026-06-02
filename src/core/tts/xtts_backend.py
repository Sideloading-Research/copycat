"""XTTS v2 TTS backend — voice cloning via Coqui XTTS.

Speaker-conditioning latents are cached as ``.pth`` files beside each
``.wav`` reference so re-loading is near-instant on subsequent runs.
"""

import os
import torch
import soundfile as sf
from pathlib import Path
from src.config import cfg


class XTTSBackend:
    """Text-to-speech with voice cloning using Coqui XTTS v2.

    Usage::

        tts = XTTSBackend()
        tts.load_model()
        tts.synthesize("Hello", "en", "/tmp/out.wav")
    """

    def __init__(self, voices_dir: str | Path | None = None):
        self.voices_dir = Path(voices_dir or cfg.voices_dir)
        self._model = None
        self.latents: dict[str, dict] = {}

    def load_model(self):
        """Load the underlying XTTS v2 model and scan voice files.

        Agrees to the Coqui License Agreement (``COQUI_TOS_AGREED``)
        which is required for XTTS v2.
        """
        if self._model is not None:
            return
        os.environ["COQUI_TOS_AGREED"] = "1"
        from TTS.api import TTS as CoquiTTS

        wrapper = CoquiTTS(cfg.tts_model_name)
        self._model = wrapper.synthesizer.tts_model
        self._refresh_voices()

    def _get_single_latent(self, voice_path: str, latent_path: str) -> dict:
        p = Path(latent_path)
        if p.exists():
            return torch.load(p, map_location="cpu")
        gpt_cond, speaker_embed = self._model.get_conditioning_latents(
            audio_path=[voice_path]
        )
        latent_dict = {
            "gpt_cond_latent": gpt_cond,
            "speaker_embedding": speaker_embed,
        }
        torch.save(latent_dict, p)
        return latent_dict

    def _refresh_voices(self):
        self.latents = {}
        for wav_path in self.voices_dir.glob("*.wav"):
            lang = wav_path.stem
            latent_path = wav_path.with_suffix(".pth")
            try:
                self.latents[lang] = self._get_single_latent(
                    str(wav_path), str(latent_path)
                )
            except Exception as e:
                print(f"  Error loading voice {lang}: {e}")

    def synthesize(self, text: str, language: str, output_path: str) -> None:
        if language not in self.latents:
            raise ValueError(
                f"Language '{language}' not configured. "
                f"Available: {list(self.latents)}"
            )
        lang_latents = self.latents[language]
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(cfg.tts_threads)
        try:
            with torch.inference_mode():
                out = self._model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=lang_latents["gpt_cond_latent"],
                    speaker_embedding=lang_latents["speaker_embedding"],
                )
            sf.write(output_path, out["wav"], cfg.tts_output_sample_rate)
        finally:
            torch.set_num_threads(prev_threads)
