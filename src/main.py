import sys
import os
import customtkinter as ctk

# Asegurar que src esté en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ui.splash import SplashScreen
from src.ui.main_window import MainWindow


def _load_modules(splash: SplashScreen) -> None:
    """Deferred import of heavy modules."""
    splash.set_progress(0.1, "Config CPU ...")
    import src.utils.setup_env  # Aquí mueves tus os.environ y parches de torch

    splash.set_progress(0.3, "Loading audio engine ...")
    import src.core.audio

    splash.set_progress(0.5, "Start Wishper & RAG ...")
    import src.core.rag

    splash.set_progress(0.8, "Loading XTTS...")
    import src.core.engine

    splash.set_progress(1.0, "Ready!")
    splash.after(1000, splash.close)


def main():
    ctk.set_appearance_mode("Dark")
    root = ctk.CTk()
    root.withdraw()


    splash = SplashScreen(root)
    # Run separate thread upload so as not to block splash animation
    import threading
    threading.Thread(target=lambda: _load_modules(splash), daemon=True).start()

    # Wait for the splash to close to reveal the app
    def check_splash():
        if not splash.winfo_exists():
            _reveal(root)
        else:
            root.after(100, check_splash)

    root.after(100, check_splash)
    root.mainloop()


def _reveal(root):
    root.destroy()
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()