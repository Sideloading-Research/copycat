"""Whisper STT backend — wraps openai-whisper for CPU inference."""

import whisper
from src.config import cfg


class WhisperSTT:
    """Speech-to-text via OpenAI Whisper (``tiny`` by default for CPU speed).

    Usage::

        stt = WhisperSTT()
        stt.load_model()
        text = stt.transcribe("/tmp/audio.wav")
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or cfg.stt_model
        self._model = None
        self.last_lang: str | None = None

    def load_model(self):
        """Load the Whisper model into memory.  Called once at startup."""
        if self._model is None:
            self._model = whisper.load_model(self.model_name)

    def transcribe(self, audio_path: str, fp16: bool = False) -> str:
        if self._model is None:
            self.load_model()
        result = self._model.transcribe(audio_path, fp16=fp16 or cfg.stt_fp16)
        self.last_lang = result.get("language")
        return result["text"].strip()
