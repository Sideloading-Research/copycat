"""Abstract interfaces for every replaceable component in the pipeline.

Each backend is defined as a ``Protocol`` so that consumers accept
any object satisfying the interface — no inheritance required.

Usage in ``PipelineOrchestrator``::

    from src.core.interfaces import STTBackend, LLMBackend, ...
    from src.core.stt.whisper_backend import WhisperSTT
    from src.core.llm.ollama_backend import OllamaLLM

    orchestrator = PipelineOrchestrator(
        stt=WhisperSTT(),
        llm=OllamaLLM(),
        ...
    )
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class STTBackend(Protocol):
    """Speech-to-text.  Transcribe an audio file to a string."""

    def transcribe(self, audio_path: str, fp16: bool = False) -> str:
        """Return the transcribed text from *audio_path*.

        The implementation may also return a language hint via the
        return value or an attribute; pipeline code reads ``.last_lang``
        if available.
        """
        ...

    @property
    def last_lang(self) -> str | None:
        """ISO 639-1 code of the detected language (if available)."""
        ...


@runtime_checkable
class LLMBackend(Protocol):
    """Large Language Model.  Generate a reply from a prompt."""

    def generate(self, prompt: str, **kwargs) -> str:
        """Return the model's response string."""
        ...


@runtime_checkable
class TTSBackend(Protocol):
    """Text-to-speech with voice cloning.  Synthesise a WAV file."""

    def synthesize(self, text: str, language: str, output_path: str) -> None:
        """Clone the speaker's voice and write a 24 kHz mono WAV to *output_path*."""
        ...


@runtime_checkable
class VectorDB(Protocol):
    """Vector database for RAG retrieval."""

    def search(self, query: str, k: int = 3, max_chars: int = 2000) -> str:
        """Return concatenated, truncated context from the top-*k* results."""
        ...

    def search_priority(
        self,
        query: str,
        priority_source: str,
        k: int = 3,
        max_chars: int = 2000,
        priority_k: int = 1,
    ) -> str:
        """Return context that always includes *priority_k* chunks from
        *priority_source* plus the best remaining results."""
        ...

    def initialize(self) -> None:
        """Load or build the index.  Called once at startup."""
        ...


@runtime_checkable
class LipSyncBackend(Protocol):
    """Lip-sync a face image to an audio track, producing an MP4 video."""

    def render(self, face_path: str, audio_path: str, output_path: str) -> None:
        """Create a lip-synced video at *output_path*."""
        ...
