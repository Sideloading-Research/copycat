"""Main application window — pure View layer.

Layout: chat transcript (left) + avatar + controls (right).
All logic is delegated to ``PipelineController``.
"""

import threading
import customtkinter as ctk
import tkinter as tk
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
from src.ui.controller import PipelineController
from src.utils.paths import PATHS

# Status-bar colour palette.
C_IDLE = "#1f538d"
C_ACTIVE = "#b71c1c"
C_OK = "#2e7d32"
C_WARN = "#e65100"
C_ERR = "#c62828"
C_INFO = "#0277bd"
C_MIC_OFF = "#0d2b5c"


class MainWindow(ctk.CTk):
    """Single-window GUI for Copycat.

    This is a **View only**.  All user actions are forwarded to
    ``PipelineController`` which owns the engine and audio.
    """

    def __init__(self, auto_init=True):
        super().__init__()
        self.title("Copycat - Local AI Avatar")
        self.geometry("1000x750")
        self.minsize(800, 600)

        icon_path = PATHS.get("icon_img")
        if icon_path and icon_path.exists():
            icon_img = ImageTk.PhotoImage(file=str(icon_path))
            self.wm_iconphoto(True, icon_img)

        self.lang_var = ctk.StringVar(value="en")
        self.full_session_log: list[dict] = []

        # Controller owns engine + audio.
        self.controller = PipelineController()
        self.recording = False
        self.mic_enabled = True

        self._build_ui()
        self._load_avatar_static()

        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        if auto_init:
            self.controller.load_models_async(
                on_ready=self._on_engine_ready,
                status_cb=self._update_status,
            )

    # ── lifecycle ────────────────────────────────────────────────

    def _on_closing(self):
        self._update_status("Saving session...", C_INFO)
        if self.full_session_log:
            self.controller.save_session(self.full_session_log)
        self.destroy()

    def _on_engine_ready(self):
        self.after(0, self._refresh_languages)
        self._update_status("System Ready", C_OK)

    # ── UI construction ──────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # ── left: chat panel ─────────────────────────────────────
        self.chat_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.chat_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.chat_frame.grid_rowconfigure(0, weight=1)
        self.chat_frame.grid_columnconfigure(0, weight=1)

        self.chat_display = ctk.CTkTextbox(self.chat_frame, state="disabled")
        self.chat_display.grid(row=0, column=0, sticky="nsew", pady=(0, 10))

        self.input_area = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        self.input_area.grid(row=1, column=0, sticky="ew")
        self.input_area.grid_columnconfigure(0, weight=1)

        self.entry_text = ctk.CTkEntry(
            self.input_area, placeholder_text="Type a message..."
        )
        self.entry_text.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.entry_text.bind("<Return>", self._on_enter_key)

        self.btn_send = ctk.CTkButton(
            self.input_area,
            text="Send",
            width=80,
            command=self._send_text_manual,
        )
        self.btn_send.grid(row=0, column=1)

        # ── right: avatar + controls ─────────────────────────────
        self.right_frame = ctk.CTkFrame(self, width=380)
        self.right_frame.grid(row=0, column=1, sticky="ns", padx=20, pady=20)
        self.right_frame.grid_propagate(False)

        self.lbl_avatar = tk.Label(self.right_frame, bg="#121212")
        self.lbl_avatar.pack(pady=(20, 5))

        ctk.CTkLabel(
            self.right_frame, text="Active Voice:", font=("Arial", 10, "bold")
        ).pack()
        self.radio_container = ctk.CTkFrame(
            self.right_frame, fg_color="transparent"
        )
        self.radio_container.pack(pady=5)

        self.lbl_status = ctk.CTkLabel(self.right_frame, text="Starting...")
        self.lbl_status.pack(pady=5)

        self.actions_frame = ctk.CTkFrame(
            self.right_frame, fg_color="transparent"
        )
        self.actions_frame.pack(pady=20)

        self.btn_mic = ctk.CTkButton(
            self.actions_frame,
            text="\U0001f3a4",
            width=80,
            height=80,
            corner_radius=40,
            command=self._toggle_voice_interaction,
        )
        self.btn_mic.pack(side="left", padx=10)

        self.btn_mic_toggle = ctk.CTkButton(
            self.actions_frame,
            text="\U0001f50a",
            width=40,
            height=40,
            corner_radius=20,
            fg_color=C_OK,
            hover_color="#1b5e20",
            command=self._toggle_mic,
        )
        self.btn_mic_toggle.pack(side="left", padx=5)

        self.btn_config = ctk.CTkButton(
            self.actions_frame,
            text="\u2699",
            width=40,
            height=40,
            corner_radius=20,
            fg_color="#444444",
            command=self._open_settings,
        )
        self.btn_config.pack(side="left", padx=5)

    # ── helpers ──────────────────────────────────────────────────

    def _refresh_languages(self):
        """Rebuild the language radio buttons from ``data/voices/*.wav``."""
        for widget in self.radio_container.winfo_children():
            widget.destroy()

        voices = list(PATHS["voices_dir"].glob("*.wav"))
        if not voices:
            ctk.CTkLabel(
                self.radio_container, text="No voices found", text_color=C_ERR
            ).pack()
            return

        for v in voices:
            lang = v.stem
            ctk.CTkRadioButton(
                self.radio_container,
                text=lang.upper(),
                variable=self.lang_var,
                value=lang,
            ).pack(side="left", padx=5)

    def _append_to_chat(self, sender, text):
        """Add a line to the chat transcript."""
        self.chat_display.configure(state="normal")
        tag = f"[{sender.upper()}]: "
        self.chat_display.insert("end", f"\n{tag}{text}\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")
        self.full_session_log.append({"role": sender, "content": text})

    def _update_status(self, text, color="#FFFFFF"):
        """Thread-safe status-bar update."""
        if not self.mic_enabled and text in ("Transcribing voice...", "Syncing lips..."):
            return
        self.after(0, lambda: self.lbl_status.configure(text=text, text_color=color))

    def _load_avatar_static(self):
        """Show the static face image (or a blank square) in the avatar area."""
        img_path = str(PATHS["face_img"])
        if not os.path.exists(img_path):
            img = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        else:
            img = Image.open(img_path)
        itk = ImageTk.PhotoImage(img.resize((328, 328)))
        self.lbl_avatar.configure(image=itk)
        self.lbl_avatar.image = itk

    def _play_video(self):
        """Play back the lip-synced video frame by frame on the avatar label."""
        video_path = str(PATHS["tmp_video"])
        if not os.path.exists(video_path):
            return
        cap = cv2.VideoCapture(video_path)

        def _next():
            if not self.winfo_exists():
                cap.release()
                return
            ret, frame = cap.read()
            if ret:
                rgba = cv2.cvtColor(
                    cv2.resize(frame, (328, 328)), cv2.COLOR_BGR2RGBA
                )
                itk = ImageTk.PhotoImage(Image.fromarray(rgba))
                self.lbl_avatar.configure(image=itk)
                self.lbl_avatar.image = itk
                self.after(40, _next)
            else:
                cap.release()
                self._load_avatar_static()

        _next()

    # ── input handling ───────────────────────────────────────────

    def _on_enter_key(self, event):
        self._send_text_manual()
        return "break"

    def _send_text_manual(self):
        msg = self.entry_text.get().strip()
        if msg:
            self.entry_text.delete(0, "end")
            self.controller.process_text(
                msg,
                self.lang_var.get(),
                chat_cb=self._append_to_chat,
                status_cb=self._update_status,
                on_complete=self._on_pipeline_complete,
            )

    def _toggle_mic(self):
        self.mic_enabled = self.controller.toggle_mic(status_cb=self._update_status)
        if self.mic_enabled:
            self.btn_mic_toggle.configure(fg_color=C_OK, text="\U0001f50a")
            self.btn_mic.configure(fg_color=C_IDLE)
        else:
            self.btn_mic_toggle.configure(fg_color=C_MIC_OFF, text="\U0001f507")
            self.btn_mic.configure(fg_color=C_MIC_OFF)

    def _toggle_voice_interaction(self):
        if not self.mic_enabled:
            self._update_status("Microphone is disabled", C_ERR)
            return
        lang = self.lang_var.get()
        if not self.recording:
            self.recording = True
            self.btn_mic.configure(fg_color=C_ACTIVE)
            self.controller.start_recording(lang, status_cb=self._update_status)
        else:
            self.recording = False
            self.btn_mic.configure(fg_color=C_IDLE)
            self.controller.stop_recording(
                lang,
                status_cb=self._update_status,
                chat_cb=self._append_to_chat,
                on_complete=self._on_pipeline_complete,
            )

    def _on_pipeline_complete(self, success: bool):
        """Called when pipeline finishes. Plays audio/video if successful and mic enabled."""
        if success and self.mic_enabled:
            threading.Thread(
                target=self.controller.audio.play_audio,
                args=(str(PATHS["tmp_bot"]),),
                daemon=True,
            ).start()
            self._play_video()

    def _open_settings(self):
        from src.ui.settings import configuration
        configuration(on_complete_callback=self._refresh_languages)
