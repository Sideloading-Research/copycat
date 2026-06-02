"""PipelineController — handles UI events and orchestrates pipeline threads.

Separated from ``MainWindow`` so the same controller can be reused
from a CLI, a web API, or any other frontend.
"""

import threading
from src.core.engine import CopycatEngine
from src.core.audio import AudioHandler
from src.utils.paths import PATHS


class PipelineController:
    """Mediates between the UI and the engine/audio layers.

    Owns the ``CopycatEngine`` and ``AudioHandler`` instances.
    Emits status and chat updates via callbacks that the view provides.

    Usage::

        ctrl = PipelineController()
        ctrl.load_models_async(on_ready=app.deiconify)

        # Later, when user presses Enter or the mic button:
        ctrl.process_text("Hello")
        ctrl.toggle_recording("es", on_stop=ctrl.process_voice)
    """

    def __init__(self, engine=None):
        self.engine = engine or CopycatEngine()
        self.audio = AudioHandler()
        self.recording = False
        self.mic_enabled = True

    # ── lifecycle ────────────────────────────────────────────────

    def load_models_async(self, on_ready=None, status_cb=None):
        """Load all AI models in a background thread.

        Calls ``on_ready()`` on the main thread when done.
        """

        def _load():
            try:
                if status_cb:
                    status_cb("Initialising Folders...", "#0277bd")
                PATHS["vector_db"].mkdir(parents=True, exist_ok=True)
                PATHS["journal"].mkdir(parents=True, exist_ok=True)
                PATHS["voices_dir"].mkdir(parents=True, exist_ok=True)

                if status_cb:
                    status_cb("Loading AI Models...", "#e65100")
                self.engine.load_models()

                if status_cb:
                    status_cb("System Ready", "#2e7d32")
                if on_ready:
                    on_ready()
            except Exception as e:
                print(f"Critical initialisation error: {e}")
                if status_cb:
                    status_cb("Engine Error", "#c62828")

        threading.Thread(target=_load, daemon=True).start()

    # ── text input ───────────────────────────────────────────────

    def process_text(self, text: str, lang: str, chat_cb=None, status_cb=None, on_complete=None):
        """Submit a typed message to the pipeline (runs in a daemon thread)."""
        if chat_cb:
            chat_cb("user", text)
        threading.Thread(
            target=self._run_pipeline,
            args=(lang, text, status_cb, chat_cb, on_complete),
            daemon=True,
        ).start()

    # ── voice input ──────────────────────────────────────────────

    def start_recording(self, lang: str, status_cb=None):
        """Begin microphone capture."""
        self.recording = True
        self.audio.start_recording()
        if status_cb:
            status_cb(f"Listening ({lang})...", "#b71c1c")

    def stop_recording(self, lang: str, status_cb=None, chat_cb=None, on_complete=None):
        """Stop microphone and process the captured audio."""
        self.recording = False
        self.audio.stop_recording(str(PATHS["tmp_user"]))
        if status_cb:
            status_cb("Processing voice...", "#0277bd")
        threading.Thread(
            target=self._run_pipeline,
            args=(lang, None, status_cb, chat_cb, on_complete),
            daemon=True,
        ).start()

    def toggle_mic(self, status_cb=None):
        """Toggle the master audio cut-off on/off.

        Returns the new state (``True`` = enabled).
        """
        self.mic_enabled = not self.mic_enabled
        if self.mic_enabled:
            if status_cb:
                status_cb("Microphone enabled", "#2e7d32")
        else:
            if self.recording:
                self.recording = False
                self.audio.stop_recording(str(PATHS["tmp_user"]))
            if status_cb:
                status_cb("Microphone disabled — audio I/O cut", "#c62828")
        return self.mic_enabled

    # ── pipeline runner ──────────────────────────────────────────

    def _run_pipeline(self, lang, manual_text, status_cb=None, chat_cb=None, on_complete=None):
        """Execute the pipeline and play audio/video on success (if mic enabled)."""
        success = self.engine.run_pipeline(
            lang,
            manual_text=manual_text,
            mic_enabled=self.mic_enabled,
            status_cb=status_cb,
            chat_cb=chat_cb,
        )
        if on_complete:
            on_complete(success)

    # ── session ──────────────────────────────────────────────────

    def save_session(self, history):
        self.engine.save_session_log(history)
