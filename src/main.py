import sys
import os
import customtkinter as ctk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ui.splash import SplashScreen
from src.ui.main_window import MainWindow


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
    ctk.set_appearance_mode("Dark")
    root = ctk.CTk()
    root.withdraw()

    splash = SplashScreen(root)
    engine_holder = {}

    def load_task():
        engine_holder["engine"] = _load_all_models(splash)

    import threading
    threading.Thread(target=load_task, daemon=True).start()

    def check_splash():
        if not splash.winfo_exists():
            _reveal(root, engine_holder.get("engine"))
        else:
            root.after(100, check_splash)

    root.after(100, check_splash)
    root.mainloop()


def _reveal(root, engine):
    root.destroy()
    app = MainWindow(engine=engine)
    app.mainloop()


if __name__ == "__main__":
    main()
