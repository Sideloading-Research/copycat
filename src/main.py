"""Application entry point — splash screen, model loading, UI launch."""

import sys
import os
import customtkinter as ctk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ui.splash import SplashScreen
from src.ui.main_window import MainWindow, C_OK


def _load_all_models(splash):
    """Load everything during splash so UI appears only when ready."""
    splash.set_progress(0.05, "Configuring CPU ...")
    import src.utils.setup_env

    splash.set_progress(0.15, "Loading Whisper, RAG & XTTS ...")
    from src.core.engine import CopycatEngine

    engine = CopycatEngine()
    engine.load_models()

    splash.set_progress(0.85, "Loading audio engine ...")
    import src.core.audio

    splash.set_progress(1.0, "Ready!")
    splash.after(800, splash.close)
    return engine


def main():
    """Create the main window (hidden), show splash, load models, reveal UI."""
    ctk.set_appearance_mode("Dark")

    # Single CTk instance for the entire app lifetime — destroying and
    # recreating CTk causes Tcl/Tk ``invalid command name`` errors because
    # the Tcl interpreter is killed.
    app = MainWindow(auto_init=False)
    app.withdraw()

    splash = SplashScreen(app)
    engine_holder = {}

    def load_task():
        engine_holder["engine"] = _load_all_models(splash)

    import threading

    threading.Thread(target=load_task, daemon=True).start()

    def check_splash():
        if not splash.winfo_exists():
            _reveal(app, engine_holder.get("engine"))
        else:
            app.after(100, check_splash)

    app.after(100, check_splash)
    app.mainloop()


def _reveal(app, engine):
    """Hand-off from splash to main UI without destroying the window."""
    if engine:
        app.engine = engine
        app._refresh_languages()
    app._update_status("System Ready", C_OK)
    app.deiconify()


if __name__ == "__main__":
    main()
