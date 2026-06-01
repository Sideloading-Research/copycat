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


# XTTS v2 inference spawns many threads by default and can easily
# saturate all CPU cores, causing thermal throttling on laptops.
# We clamp it to 2 threads so it does not compete with Ollama or
# Whisper.  The user can raise this if they have a desktop with
# adequate cooling.
_TTS_THREADS = 2


class TTSManager:
    """Manages XTTS v2 model lifecycle and cached speaker-latent inference.

    Key design decisions
    --------------------
    - The raw XTTS model is extracted from the Coqui ``TTS`` wrapper
      (``tts_wrapper.synthesizer.tts_model``) instead of being loaded
      directly, because the Coqui API handles model-path resolution
      and checkpoint download automatically.
    - Speaker latents are computed once per voice file and cached as
      ``.pth`` next to the ``.wav``.  Loading from disk is ~100× faster
      than re-computing.
    """

    def __init__(self, voices_dir):
        self.voices_dir = Path(voices_dir)
        # language → latent dict  (gpt_cond_latent + speaker_embedding)
        self.latents: dict[str, dict] = {}
        self._init_model()
        self.refresh_voices()

    def _init_model(self):
        """Load the Coqui TTS wrapper and extract the underlying XTTS model.

        Agreeing to the Coqui License Agreement (``COQUI_TOS_AGREED``)
        is required for XTTS v2.  See:
        https://coqui.ai/cpml
        """
        os.environ["COQUI_TOS_AGREED"] = "1"
        from TTS.api import TTS as CoquiTTS

        # The wrapper is a thin entry point that downloads the model if
        # missing and sets up the full TTS pipeline (vocoder, denoiser,
        # etc.).  We only keep the underlying XTTS model for direct
        # ``inference()`` calls, which saves the vocoder forward-pass
        # overhead.
        tts_wrapper = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
        self.xtts_model = tts_wrapper.synthesizer.tts_model

    def _get_single_latent(self, voice_path: str, latent_path: str) -> dict:
        """Return speaker latents for one voice, loading from cache if possible.

        Parameters
        ----------
        voice_path : str
            Path to the reference ``.wav`` file (6-15 s, clean audio).
        latent_path : str
            Path to the cached ``.pth`` file.  If it exists it is loaded
            directly; otherwise the latents are computed and saved.

        Returns
        -------
        dict
            Contains ``"gpt_cond_latent"`` and ``"speaker_embedding"``.
        """
        p = Path(latent_path)
        if p.exists():
            # ``map_location="cpu"`` ensures latents saved on a GPU
            # system can be loaded on a CPU-only machine without error.
            return torch.load(p, map_location="cpu")
        return self._compute_and_save_latent(voice_path, p)

    def _compute_and_save_latent(self, voice_path: str, latent_path: Path) -> dict:
        """Compute speaker-embedding / GPT-cond latents and write them to disk.

        This is called once per voice file on the first run.  The
        resulting ``.pth`` is ~130 KB and contains two tensors:
        ``gpt_cond_latent`` (6, 1024) and ``speaker_embedding`` (768,).
        """
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

        Parameters
        ----------
        text : str
            Text to vocalise.
        language : str
            Language code (``"es"`` or ``"en"``).  Must match one of the
            voice files in ``data/voices/``.
        output_path : str
            Destination path for the 24 kHz mono WAV file.

        Notes
        -----
        - Threads are clamped to ``_TTS_THREADS`` during inference to
          avoid CPU contention with the other loaded models (Ollama,
          Whisper).
        - The output sample rate is fixed at 24 kHz.
        """
        if language not in self.latents:
            raise ValueError(f"Language '{language}' not configured in TTSManager.")

        lang_latents = self.latents[language]

        # Limit XTTS thread usage so it does not steal cores from
        # Ollama (which is already running with 2 threads).
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(_TTS_THREADS)

        try:
            with torch.inference_mode():
                out = self.xtts_model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=lang_latents["gpt_cond_latent"],
                    speaker_embedding=lang_latents["speaker_embedding"],
                )
            sf.write(output_path, out["wav"], 24000)
        finally:
            # Restore the original thread count so other callers are
            # not affected.
            torch.set_num_threads(prev_threads)

    def refresh_voices(self):
        """Scan ``data/voices/`` and load latents for every ``.wav`` found.

        The file stem (e.g. ``en`` from ``en.wav``) is used as the
        language key.  Latents that were already cached are loaded from
        disk; uncached voices are computed and saved.

        This method is called once during construction.  If the user
        adds a new voice file at runtime they should call
        ``refresh_voices()`` again.
        """
        self.latents = {}
        for wav_path in self.voices_dir.glob("*.wav"):
            lang = wav_path.stem
            # Cached latents live next to the source .wav with a .pth
            # extension (e.g. ``en.wav`` → ``en.pth``).
            latent_path = wav_path.with_suffix(".pth")
            try:
                self.latents[lang] = self._get_single_latent(
                    str(wav_path), str(latent_path)
                )
                print(f"  Voice '{lang}' loaded ({wav_path.name})")
            except Exception as e:
                print(f"Error loading voice {lang}: {e}")
