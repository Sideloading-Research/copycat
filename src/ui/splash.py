"""Splash screen — displayed while the IDE modules load."""

import customtkinter as ctk
from PIL import Image, ImageTk
from pathlib import Path

ASSET_DIR = Path(__file__).parent.parent / "assets"


class SplashScreen(ctk.CTkToplevel):
    """Animated splash window shown on startup."""

    def __init__(self, parent: ctk.CTk) -> None:
        super().__init__(parent)
        self.title("")
        self.geometry("520x320")
        self.resizable(False, False)
        self.overrideredirect(True)  # No window decorations
        self._centre()
        self.configure(fg_color="#0A0E1A")
        self._build_ui()

    def _centre(self) -> None:
        """Place the window in the centre of the screen."""
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - 520) // 2
        y = (sh - 320) // 2
        self.geometry(f"520x320+{x}+{y}")

    def _build_ui(self) -> None:
        """Construct splash widgets."""
        # Icon
        icon_path = ASSET_DIR / "logo.png"
        if icon_path.exists():
            img = Image.open(icon_path).resize((120, 120))
            self._icon_img = ctk.CTkImage(light_image=img, size=(120, 120))
            ctk.CTkLabel(self, image=self._icon_img, text="").pack(pady=(30, 8))

        # Title
        ctk.CTkLabel(
            self,
            text="Copycat",
            font=ctk.CTkFont(family="Courier New", size=32, weight="bold"),
            text_color="#00FF88",
        ).pack()

        # Subtitle
        ctk.CTkLabel(
            self,
            text="Local AI Avatar",  #
            font=ctk.CTkFont(size=12),
            text_color="#546E7A",
        ).pack(pady=(2, 4))

        ctk.CTkLabel(
            self,
            text="v0.1.0",  # Versión inicial de Copycat[cite: 11]
            font=ctk.CTkFont(size=10),
            text_color="#1F6FEB",
        ).pack(pady=(0, 16))

        # Progress bar
        self._progress = ctk.CTkProgressBar(
            self, width=360, height=10,
            fg_color="#1A1F2E", progress_color="#00FF88",
        )
        self._progress.set(0)
        self._progress.pack(pady=(0, 8))

        # Status label
        self._status = ctk.CTkLabel(
            self, text="Loading…",
            font=ctk.CTkFont(size=10), text_color="#546E7A",
        )
        self._status.pack()

    def set_progress(self, value: float, message: str = "") -> None:
        """Update the progress bar (0.0–1.0) and status text."""
        self._progress.set(value)
        if message:
            self._status.configure(text=message)
        self.update()

    def close(self) -> None:
        """Destroy the splash window."""
        self.destroy()
