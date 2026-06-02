"""Application entry point — splash screen, model loading, UI launch.

Updated to use the refactored ``PipelineController`` with proper
error handling so splash never hangs on failure.
"""

import sys
import os
import threading
import customtkinter as ctk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ui.splash import SplashScreen
from src.ui.main_window import MainWindow, C_OK, C_ERR


def _load_all_models(splash, app):
    """Load everything during splash so UI appears only when ready.

    Wrapped in try/except so a model failure does not leave the
    splash screen stuck forever.
    """
    try:
        splash.set_progress(0.05, "Configuring CPU ...")
        import src.utils.setup_env  # noqa: F401 — side-effects only

        splash.set_progress(0.15, "Loading Whisper, RAG & XTTS ...")

        # PipelineController is already created by MainWindow.
        # We re-use it to load models and capture the ready callback.
        ctrl = app.controller

        def on_ready():
            splash.set_progress(0.85, "Loading audio engine ...")
            import src.core.audio  # noqa: F401
            splash.set_progress(1.0, "Ready!")
            splash.after(800, splash.close)

        def on_status(msg, colour):
            # Forward status to splash progress
            if "Ready" in msg:
                on_ready()
            elif "Error" in msg:
                raise RuntimeError(msg)

        ctrl.load_models_async(
            on_ready=on_ready,
            status_cb=lambda msg, colour: splash.set_progress(0.5, msg),
        )

    except Exception as e:
        error_msg = f"Failed to load models: {e}"
        print(error_msg)
        splash.set_progress(0.0, error_msg)
        splash.after(3000, splash.close)


def main():
    """Create the main window (hidden), show splash, load models, reveal UI."""
    ctk.set_appearance_mode("Dark")

    app = MainWindow(auto_init=False)   # No auto-load — splash handles it.
    app.withdraw()

    splash = SplashScreen(app)

    def load_task():
        _load_all_models(splash, app)

    threading.Thread(target=load_task, daemon=True).start()

    def check_splash():
        if not splash.winfo_exists():
            _reveal(app)
        else:
            app.after(100, check_splash)

    app.after(100, check_splash)
    app.mainloop()


def _reveal(app):
    """Hand-off from splash to main UI without destroying the window."""
    app.after(0, app._refresh_languages)
    app._update_status("System Ready", C_OK)
    app.deiconify()


if __name__ == "__main__":
    main()
