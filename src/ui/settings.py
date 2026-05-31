import customtkinter as ctk
from tkinter import filedialog
from pathlib import Path
import shutil
from src.utils.paths import PATHS


def configuration(on_complete_callback=None):
    window = ctk.CTkToplevel()
    window.geometry("600x700")
    window.title("Personality and Configuration")
    window.attributes("-topmost", True)

    # --- BUTTON LOGIC ---
    def select_face():
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            PATHS["face_img"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, PATHS["face_img"])
            lbl_face.configure(text="Face loaded", text_color="green")

    def select_voice():
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav")])
        if path:
            target = PATHS["voice_es"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, target)
            lbl_voice.configure(text="Voice loaded", text_color="green")

    def load_mds():
        paths = filedialog.askopenfilenames(filetypes=[("Markdown files", "*.md")])
        for p in paths:
            shutil.copy(p, PATHS["journal"] / Path(p).name)
        lbl_md.configure(text=f"{len(paths)} files loaded", text_color="green")

    def save_rules_only():
        content = txt_behavior.get("1.0", "end-1c")
        PATHS["behavior"].parent.mkdir(parents=True, exist_ok=True)
        with open(PATHS["behavior"], "w", encoding="utf-8") as f:
            f.write(content)
        lbl_saved.configure(text="Rules saved!", text_color="green")
        window.after(2000, lambda: lbl_saved.configure(text=""))

    def save_and_close():
        save_rules_only()
        window.destroy()
        if on_complete_callback:
            on_complete_callback()

    # --- UI LAYOUT ---
    ctk.CTkLabel(window, text="Initial Configuration", font=("Arial", 20, "bold")).pack(pady=20)

    # File buttons
    btn_face = ctk.CTkButton(window, text="Load Face (face.jpeg)", command=select_face).pack(pady=5)
    lbl_face = ctk.CTkLabel(window, text="Required", text_color="red")
    lbl_face.pack()

    btn_voice = ctk.CTkButton(window, text="Load Base Voice (.wav)", command=select_voice).pack(pady=5)
    lbl_voice = ctk.CTkLabel(window, text="Required", text_color="red")
    lbl_voice.pack()

    btn_md = ctk.CTkButton(window, text="Import .md Documents to Journal", command=load_mds).pack(pady=5)
    lbl_md = ctk.CTkLabel(window, text="At least one recommended", text_color="gray")
    lbl_md.pack()

    # Behavior Box
    ctk.CTkLabel(window, text="Behavior and Personality:").pack(pady=(20, 0))
    txt_behavior = ctk.CTkTextbox(window, width=500, height=200)
    txt_behavior.pack(pady=10)

    # Load existing behavior if present
    if PATHS["behavior"].exists():
        existing = PATHS["behavior"].read_text(encoding="utf-8")
        txt_behavior.insert("0.0", existing)
    else:
        txt_behavior.insert("0.0", "You are a friendly and technical assistant...")

    # Bind Ctrl+Enter to save without closing
    txt_behavior.bind("<Control-Return>", lambda e: save_rules_only())

    # Save feedback label
    lbl_saved = ctk.CTkLabel(window, text="", text_color="green")
    lbl_saved.pack()

    # Button row: save rules + finish
    btn_frame = ctk.CTkFrame(window, fg_color="transparent")
    btn_frame.pack(pady=20)

    btn_save_rules = ctk.CTkButton(btn_frame, text="Save Rules  (Ctrl+Enter)", command=save_rules_only)
    btn_save_rules.pack(side="left", padx=10)

    btn_finish = ctk.CTkButton(btn_frame, text="FINISH AND START", fg_color="green", command=save_and_close)
    btn_finish.pack(side="left", padx=10)
