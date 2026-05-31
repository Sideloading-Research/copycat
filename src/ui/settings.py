import customtkinter as ctk
from tkinter import filedialog
import shutil
from src.utils.paths import PATHS


def configuration(on_complete_callback):
    window = ctk.CTkToplevel()  # Usamos Toplevel para que no muera la app principal
    window.geometry("600x700")
    window.title("Personality and Configuration")
    window.attributes("-topmost", True)  # Que aparezca encima

    # --- LÓGICA DE BOTONES ---
    def select_face():
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            # Skill: Copiar archivos y renombrar para que el sistema los encuentre
            PATHS["face_img"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, PATHS["face_img"])
            lbl_face.configure(text="✅ Face loaded", text_color="green")

    def select_voice():
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav")])
        if path:
            # Por defecto lo guardamos como es.wav para el primer arranque
            target = PATHS["voice_es"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, target)
            lbl_voice.configure(text="✅ Voice loaded", text_color="green")

    def load_mds():
        paths = filedialog.askopenfilenames(filetypes=[("Markdown files", "*.md")])
        for p in paths:
            shutil.copy(p, PATHS["journal"] / Path(p).name)
        lbl_md.configure(text=f"✅ {len(paths)} archivos cargados", text_color="green")

    def save_behavior():
        content = txt_behavior.get("1.0", "end-current")
        PATHS["behavior"].parent.mkdir(parents=True, exist_ok=True)
        with open(PATHS["behavior"], "w", encoding="utf-8") as f:
            f.write(content)
        window.destroy()
        on_complete_callback()  # Avisamos que terminamos

    # --- UI LAYOUT ---
    ctk.CTkLabel(window, text="Configuración Inicial", font=("Arial", 20, "bold")).pack(pady=20)

    # Botones de archivos
    btn_face = ctk.CTkButton(window, text="Cargar Rostro (face.jpeg)", command=select_face).pack(pady=5)
    lbl_face = ctk.CTkLabel(window, text="Requerido", text_color="red")
    lbl_face.pack()

    btn_voice = ctk.CTkButton(window, text="Cargar Voz Base (.wav)", command=select_voice).pack(pady=5)
    lbl_voice = ctk.CTkLabel(window, text="Requerido", text_color="red")
    lbl_voice.pack()

    btn_md = ctk.CTkButton(window, text="Importar documentos al Diario (.md)", command=load_mds).pack(pady=5)
    lbl_md = ctk.CTkLabel(window, text="Mínimo uno sugerido", text_color="gray")
    lbl_md.pack()

    # Behavior Box
    ctk.CTkLabel(window, text="Conducta y Personalidad (Behavior):").pack(pady=(20, 0))
    txt_behavior = ctk.CTkTextbox(window, width=500, height=200)
    txt_behavior.pack(pady=10)
    txt_behavior.insert("0.0", "Eres un asistente amable y técnico...")  # Default

    btn_save = ctk.CTkButton(window, text="FINALIZAR Y ARRANCAR", fg_color="green", command=save_behavior)
    btn_save.pack(pady=30)
