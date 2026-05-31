import customtkinter as ctk
import tkinter as tk
import cv2
import threading
import os
import numpy as np
from PIL import Image, ImageTk
from src.core.engine import CopycatEngine
from src.core.audio import AudioHandler
from src.utils.paths import PATHS

# Color states
C_IDLE, C_ACTIVE, C_OK, C_WARN, C_ERR, C_INFO = "#1f538d", "#b71c1c", "#2e7d32", "#e65100", "#c62828", "#0277bd"


class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Copycat – Multi-Lang AI")
        self.geometry("1000x750")
        self.minsize(800, 600)
        # Estado
        self.lang_var = ctk.StringVar(value="en")
        self.full_session_log = []
        self.engine = CopycatEngine()
        self.audio = AudioHandler()
        self.recording = False

        self._build_ui()
        self._load_avatar_static()
        # Permitir que las columnas se expandan
        self.grid_columnconfigure(0, weight=1)  # Chat se estira
        self.grid_columnconfigure(1, weight=0)  # Controles fijos
        self.grid_rowconfigure(0, weight=1)
        # Carga asíncrona de modelos pesados
        threading.Thread(target=self._init_engine, daemon=True).start()

    def _on_closing(self):
        """Gestión de cierre: Guarda el log y limpia recursos."""
        self._update_status("Saving session...", C_INFO)

        # Guardamos el log de la sesión a través del motor
        if self.full_session_log:
            self.engine.save_session_log(self.full_session_log)

        # Limpieza de hilos y cierre
        self.destroy()

    def _init_engine(self):
        """Carga modelos de forma segura y asegura la creación de carpetas."""
        try:
            self._update_status("Initializing Folders...", C_INFO)
            # Forzamos la creación de la estructura de datos antes de nada
            PATHS["vector_db"].mkdir(parents=True, exist_ok=True)
            PATHS["journal"].mkdir(parents=True, exist_ok=True)
            PATHS["voices_dir"].mkdir(parents=True, exist_ok=True)

            self._update_status("Loading AI Models...", C_WARN)
            # Llamamos a la carga del motor
            self.engine.load_models()

            # Una vez cargado, refrescamos la UI
            self.after(0, self._refresh_languages)
            self._update_status("System Ready", C_OK)

        except Exception as e:
            # Si algo falla aquí, al menos lo veremos en la consola y no se congelará
            print(f"❌ Error crítico en la inicialización: {e}")
            self._update_status("Engine Error", C_ERR)

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # --- IZQUIERDA: CHAT ---
        self.chat_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.chat_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.chat_frame.grid_rowconfigure(0, weight=1)
        self.chat_frame.grid_columnconfigure(0, weight=1)

        self.chat_display = ctk.CTkTextbox(self.chat_frame, state="disabled")
        self.chat_display.grid(row=0, column=0, sticky="nsew", pady=(0, 10))

        self.input_area = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        self.input_area.grid(row=1, column=0, sticky="ew")

        self.entry_text = ctk.CTkEntry(self.input_area, placeholder_text="Escribe...")
        self.entry_text.pack(side="left", expand=True, fill="x", padx=(0, 10))
        self.entry_text.bind("<Return>", lambda e: self._send_text_manual())

        # --- DERECHA: AVATAR Y CONTROLES ---
        self.right_frame = ctk.CTkFrame(self, width=380)
        self.right_frame.grid(row=0, column=1, sticky="ns", padx=20, pady=20)
        self.right_frame.grid_propagate(False)

        # 1. Avatar
        self.lbl_avatar = tk.Label(self.right_frame, bg="#121212")
        self.lbl_avatar.pack(pady=(20, 5))

        # 2. Selector de Idioma (Debajo del avatar)
        ctk.CTkLabel(self.right_frame, text="Voz Activa:", font=("Arial", 10, "bold")).pack()
        self.radio_container = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.radio_container.pack(pady=5)

        # 3. Status
        self.lbl_status = ctk.CTkLabel(self.right_frame, text="Iniciando...")
        self.lbl_status.pack(pady=5)

        # 4. Botonera (Mic y Settings al lado)
        self.actions_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.actions_frame.pack(pady=20)

        self.btn_mic = ctk.CTkButton(self.actions_frame, text="🎤", width=80, height=80,
                                     corner_radius=40, command=self._toggle_voice_interaction)
        self.btn_mic.pack(side="left", padx=10)

        self.btn_config = ctk.CTkButton(self.actions_frame, text="⚙", width=40, height=40,
                                        corner_radius=20, fg_color="#444444", command=self._open_settings)
        self.btn_config.pack(side="left", padx=10)


    def _refresh_languages(self):
        """Busca archivos .wav en data/voices/ y crea los radio botones."""
        for widget in self.radio_container.winfo_children():
            widget.destroy()

        voices = list(PATHS["voices_dir"].glob("*.wav"))
        if not voices:
            ctk.CTkLabel(self.radio_container, text="No voices found", text_color=C_ERR).pack()
            return

        for v in voices:
            lang = v.stem
            ctk.CTkRadioButton(self.radio_container, text=lang.upper(),
                               variable=self.lang_var, value=lang).pack(side="left", padx=5)

    def _append_to_chat(self, sender, text):
        """Añade texto al chat y guarda en el log de sesión."""
        self.chat_display.configure(state="normal")
        tag = f"[{sender.upper()}]: "
        self.chat_display.insert("end", f"\n{tag}{text}\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")
        self.full_session_log.append({"role": sender, "content": text})

    def _send_text_manual(self):
        msg = self.entry_text.get().strip()
        if msg:
            self.entry_text.delete(0, "end")
            self._append_to_chat("user", msg)
            lang = self.lang_var.get()
            threading.Thread(target=self._process_pipeline, args=(lang, msg), daemon=True).start()

    def _toggle_voice_interaction(self):
        lang = self.lang_var.get()
        if not self.recording:
            self.recording = True
            self.btn_mic.configure(fg_color=C_ACTIVE)
            self._update_status(f"Listening ({lang})...", C_ACTIVE)
            self.audio.start_recording()
        else:
            self.recording = False
            self.btn_mic.configure(fg_color=C_IDLE)
            self.audio.stop_recording(str(PATHS["tmp_user"]))
            self._update_status("Processing voice...", C_INFO)
            threading.Thread(target=self._process_pipeline, args=(lang, None), daemon=True).start()

    def _process_pipeline(self, lang, manual_text):
        """Ejecuta el pipeline completo y gestiona la respuesta."""
        success = self.engine.run_pipeline(
            lang,
            manual_text=manual_text,
            status_cb=self._update_status,
            chat_cb=self._append_to_chat
        )
        if success:
            threading.Thread(target=self.audio.play_audio, args=(str(PATHS["tmp_bot"]),), daemon=True).start()
            self._play_video()

    def _update_status(self, text, color="#FFFFFF"):
        self.after(0, lambda: self.lbl_status.configure(text=text, text_color=color))

    def _load_avatar_static(self):
        img_path = str(PATHS["face_img"])
        if not os.path.exists(img_path):
            img = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        else:
            img = Image.open(img_path)

        itk = ImageTk.PhotoImage(img.resize((328, 328)))
        self.lbl_avatar.configure(image=itk)
        self.lbl_avatar.image = itk

    def _play_video(self):
        video_path = str(PATHS["tmp_video"])
        if not os.path.exists(video_path): return
        cap = cv2.VideoCapture(video_path)

        def _next():
            ret, frame = cap.read()
            if ret:
                rgba = cv2.cvtColor(cv2.resize(frame, (328, 328)), cv2.COLOR_BGR2RGBA)
                itk = ImageTk.PhotoImage(Image.fromarray(rgba))
                self.lbl_avatar.configure(image=itk)
                self.lbl_avatar.image = itk
                self.after(40, _next)
            else:
                cap.release()
                self._load_avatar_static()

        _next()

    def _open_settings(self):
        from src.ui.settings import configuration
        configuration(on_complete_callback=self._refresh_languages)