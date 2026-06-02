"""Refactored CopycatEngine — dependency-injected pipeline coordinator.

Keeps the same public API so ``main.py`` and ``main_window.py``
require minimal changes, but internally delegates to:

- ``core.stt.whisper_backend.WhisperSTT``
- ``core.llm.ollama_backend.OllamaLLM``
- ``core.tts.xtts_backend.XTTSBackend``
- ``core.vector_db.chroma_backend.ChromaVectorDB``
- ``core.lipsync.wav2lip_backend.Wav2LipBackend``
- ``orchestrator.pipeline.PipelineOrchestrator``
- ``orchestrator.prompt_builder.PromptBuilder``
- ``orchestrator.name_detector.NameDetector``
- ``orchestrator.session_logger.SessionLogger``
"""

from src.config import cfg
from src.utils.paths import PATHS
from src.core.stt.whisper_backend import WhisperSTT
from src.core.llm.ollama_backend import OllamaLLM
from src.core.tts.xtts_backend import XTTSBackend
from src.core.vector_db.chroma_backend import ChromaVectorDB
from src.core.lipsync.wav2lip_backend import Wav2LipBackend
from src.orchestrator.pipeline import PipelineOrchestrator
from src.orchestrator.prompt_builder import PromptBuilder
from src.orchestrator.name_detector import NameDetector
from src.orchestrator.session_logger import SessionLogger


class CopycatEngine:
    """Facade that wires backends together and exposes the same
    public API as the original monolithic engine.

    Usage::

        engine = CopycatEngine()
        engine.load_models()
        engine.run_pipeline("es", manual_text="Hola")
    """

    def __init__(self):
        self.persona_name: str | None = None

        # Instantiate backends (all CPU-optimised defaults from config).
        self._stt = WhisperSTT()
        self._llm = OllamaLLM()
        self._tts = XTTSBackend()
        self._vdb = ChromaVectorDB()
        self._lipsync = Wav2LipBackend()
        self._prompt_builder = PromptBuilder()
        self._name_detector = NameDetector()
        self._logger = SessionLogger()

        self._orchestrator = PipelineOrchestrator(
            stt=self._stt,
            llm=self._llm,
            tts=self._tts,
            vector_db=self._vdb,
            lipsync=self._lipsync,
            prompt_builder=self._prompt_builder,
            name_detector=self._name_detector,
            logger=self._logger,
        )

    # ── asset check ─────────────────────────────────────────────

    @staticmethod
    def check_assets():
        """Return ``(has_critical_misses, details_dict)`` for startup validation."""
        missing = {
            "face": not PATHS["face_img"].exists(),
            "voices": not any(PATHS["journal"].parent.glob("voices/*.wav")),
            "journal": not any(PATHS["journal"].glob("*.md")),
        }
        is_critical = missing["face"] or missing["voices"] or missing["journal"]
        return is_critical, missing

    # ── model loading ────────────────────────────────────────────

    def load_models(self):
        """Load Whisper, initialise ChromaDB, boot XTTS v2.

        This method exists to preserve the old API used by ``main.py``.
        """
        try:
            self._sync_behavior_to_journal()
            self._stt.load_model()
            self._vdb.initialize()
            self._tts.load_model()
            print("All models loaded successfully.")
        except Exception as e:
            print(f"Critical model load error: {e}")

    # ── behaviour sync ──────────────────────────────────────────

    @staticmethod
    def _sync_behavior_to_journal():
        """Copy ``behavior.txt`` → ``journal/00_behavior.md`` for RAG indexing."""
        behavior_file = PATHS["behavior"]
        if not behavior_file.exists():
            print("No behavior.txt found — skipping sync.")
            return
        content = behavior_file.read_text(encoding="utf-8")
        wrapped = "# Personality Profile\n\n" + content.strip()
        dest = PATHS["journal"] / "00_behavior.md"
        dest.write_text(wrapped, encoding="utf-8")
        print(f"Behaviour synced → {dest}")

    # ── main pipeline ───────────────────────────────────────────

    def run_pipeline(self, lang, manual_text=None, mic_enabled=True, status_cb=None, chat_cb=None):
        """Execute the full STT → RAG → LLM → TTS → lip-sync pipeline.

        When *mic_enabled* is ``False`` the TTS and lip-sync steps are
        skipped entirely to avoid unnecessary computation.

        Returns ``True`` on success, ``False`` on error.
        """
        # Sync persona_name so the orchestrator picks it up.
        self._orchestrator.persona_name = self.persona_name

        result = self._orchestrator.run_turn(
            lang=lang,
            manual_text=manual_text,
            mic_enabled=mic_enabled,
            status_cb=status_cb,
            chat_cb=chat_cb,
        )

        # Sync back.
        self.persona_name = self._orchestrator.persona_name
        return result

    # ── session persistence ─────────────────────────────────────

    def save_session_log(self, history_list):
        """Write the full chat log to a timestamped markdown file."""
        self._logger.save_session(history_list)
