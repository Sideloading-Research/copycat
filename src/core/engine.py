import os
import sys
import subprocess
import ollama
import whisper
from src.core.rag import RAGManager
from src.core.tts_manager import TTSManager
import datetime
from src.utils.paths import PATHS


class CopycatEngine:
    def __init__(self):
        self.whisper_m = None
        self.tts_manager = whisper.load_model("tiny")
        self.rag = RAGManager(
            journal_path=PATHS["journal"],
            db_path=PATHS["vector_db"],
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def check_assets(self):
        """
            Check the minimum resources to boot.
            A dictionary returns with what is missing.
        """
        missing = {
            "face": not PATHS["face_img"].exists(),
            "voices": not any(PATHS["journal"].parent.glob("voices/*.wav")),  # Busca cualquier wav en voices
            "journal": not any(PATHS["journal"].glob("*.md"))
        }

        # Es crítico solo si falta la cara o no hay ninguna voz
        is_critical = missing["face"] or missing["voices"] or missing["journal"]
        return is_critical, missing

    def load_models(self):
        try:
            # 1. Cargar Whisper primero (es lo más estable)
            import whisper
            self.whisper_m = whisper.load_model("tiny")

            # 2. Cargar RAG
            self.rag = RAGManager(
                journal_path=PATHS["journal"],
                db_path=PATHS["vector_db"],
                model_name="all-MiniLM-L6-v2"  # O el modelo que prefieras
            )
            self.rag.initialize()

            # 3. Cargar TTS de forma segura
            from src.core.tts_manager import TTSManager
            # Pasamos la carpeta de voces para que escanee lo que hay
            self.tts_manager = TTSManager(PATHS["voices_dir"])

            print("✅ Todos los modelos cargados (Voz pasiva si no hay archivos).")
        except Exception as e:
            print(f"❌ Error crítico en carga de modelos: {e}")

    def run_pipeline(self, lang, manual_text=None, status_cb=None, chat_cb=None):
        """
        Pipeline unificado de Copycat: Maneja voz (Whisper) o texto manual,
        RAG, LLM (Ollama) y síntesis multimedia.
        """
        try:
            # 1. Obtención de Texto (STT o Manual)
            if manual_text:
                user_text = manual_text
            else:
                if status_cb: status_cb("Transcribing voice...", "#e65100")
                # Usamos tu referencia correcta: self.whisper_m con fp16=False
                result = self.whisper_m.transcribe(str(PATHS["tmp_user"]), fp16=False, language=lang)
                user_text = result["text"].strip()
                # Notificar al chat visual si la entrada fue por voz
                if chat_cb: chat_cb("user", user_text)

            if not user_text:
                return False

            # 2. RAG + Behavior (Memoria y Conducta)
            if status_cb: status_cb("Searching memory...", "#0277bd")
            context = self.rag.search(user_text)

            behavior = ""
            if PATHS["behavior"].exists():
                behavior = PATHS["behavior"].read_text(encoding="utf-8")

            # 3. LLM (Ollama Qwen2.5:3b)
            if status_cb: status_cb("Thinking...", "#e65100")

            # Inyección de etiquetas de idioma solicitadas[cite: 9]
            prompt = (
                f"SYSTEM BEHAVIOR:\n{behavior}\n\n"
                f"CONTEXT FROM JOURNAL:\n{context}\n\n"
                f"USER [{lang.upper()}]: {user_text}\n"
                f"ASSISTANT [{lang.upper()}]:"
            )

            resp = ollama.generate(model="qwen2.5:3b", prompt=prompt)
            bot_text = resp["response"].strip()

            if chat_cb: chat_cb("bot", bot_text)

            # 4. TTS (XTTS v2)
            if status_cb: status_cb("Cloning voice...", "#2e7d32")
            # Asegúrate de que el método en tts_manager se llame generate_tts o generate
            self.tts_manager.generate_tts(bot_text, lang, str(PATHS["tmp_bot"]))

            # 5. Wav2Lip (Sincronización Labial)
            if status_cb: status_cb("Syncing lips...", "#2e7d32")
            # Llamamos a tu método interno de sincronización
            self._sync_lips()

            if status_cb: status_cb("Ready", "#2e7d32")
            return True

        except Exception as e:
            error_msg = f"Pipeline Error: {str(e)}"
            if chat_cb: chat_cb("system", error_msg)
            if status_cb: status_cb("Error", "#c62828")
            print(error_msg)
            return False

    def _sync_lips(self):
        cmd = (
            f"{sys.executable} {PATHS['wav2lip_script']} "
            f"--checkpoint_path {PATHS['wav2lip_pth']} "
            f"--face {PATHS['face_img']} "
            f"--audio {PATHS['tmp_bot']} "
            f"--outfile {PATHS['tmp_video']} --nosmooth"
        )
        subprocess.run(cmd, shell=True, check=True, env={**os.environ, "OMP_NUM_THREADS": "6"})

    def _build_prompt(self, lang, context, text):
        instr = "Responde en español. Máximo 20 palabras." if lang == "es" else "English. Max 20 words."
        return f"{instr}\nContext:\n{context}\n\nQ: {text}\nA:"

    def save_session_log(self, history_list):
        """
        Toma la lista de diccionarios [{'role':..., 'content':...}]
        y la guarda en data/journal/chatlog-YYYYMMDD_HHMMSS.md
        """
        if not history_list:
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chatlog-{timestamp}.md"
        filepath = PATHS["journal"] / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# Chat Session Log - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
                for entry in history_list:
                    role = entry['role'].upper()
                    content = entry['content']
                    f.write(f"**{role}**: {content}\n\n")
            print(f"✅ Session saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving session: {e}")