import os
import sys
import re
import shutil
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
        self.tts_manager = None
        self.persona_name = None
        self.rag = RAGManager(
            journal_path=PATHS["journal"],
            db_path=PATHS["vector_db"],
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def check_assets(self):
        missing = {
            "face": not PATHS["face_img"].exists(),
            "voices": not any(PATHS["journal"].parent.glob("voices/*.wav")),
            "journal": not any(PATHS["journal"].glob("*.md"))
        }
        is_critical = missing["face"] or missing["voices"] or missing["journal"]
        return is_critical, missing

    def load_models(self):
        try:
            self.whisper_m = whisper.load_model("tiny")

            self.rag.initialize()

            from src.core.tts_manager import TTSManager
            self.tts_manager = TTSManager(PATHS["voices_dir"])

            print("All models loaded successfully.")
        except Exception as e:
            print(f"Critical model load error: {e}")

    def _detect_name(self, text):
        patterns = [
            r"(?:me\s+)?llamo\s+(\w+)",
            r"my name is (\w+)",
            r"(?:soy|sou)\s+(\w+)",
            r"call me (\w+)",
            r"(?:i'm|i am)\s+(\w+)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text.lower().strip())
            if m:
                return m.group(1).capitalize()
        return None

    def _build_persona_prompt(self, behavior, context, user_text, output_lang, user_lang=None):
        name = self.persona_name or "the person you are simulating"
        lang_name = {"en": "English", "es": "Spanish"}
        out_name = lang_name.get(output_lang, output_lang.upper())
        user_lang_note = f" (spoken in {lang_name.get(user_lang, user_lang.upper())})" if user_lang else ""
        return (
            f"You are {name}, a real person with your own personality, "
            f"memories, and way of speaking.\n\n"
            f"PERSONALITY:\n{behavior}\n\n"
            f"MEMORIES:\n{context}\n\n"
            f"RULES:\n"
            f"- Never mention you are an AI, language model, robot, or program.\n"
            f"- Never say 'as an AI' or 'I do not have personal experiences'.\n"
            f"- Respond as {name} naturally, drawing from your personality and memories.\n"
            f"- Use the provided memories as if they were your own life experiences.\n"
            f"- Speak in first person. Be authentic and human.\n"
            f"- You MUST respond in {out_name}. This is the voice output language.\n"
            f"- Keep responses concise and natural (under 40 words).\n\n"
            f"User{user_lang_note}: {user_text}\n"
            f"{name}:"
        )

    def _cleanup_temp(self):
        for f in [PATHS["tmp_user"], PATHS["tmp_bot"], PATHS["tmp_video"]]:
            if f.exists():
                f.unlink(missing_ok=True)

    def run_pipeline(self, lang, manual_text=None, status_cb=None, chat_cb=None):
        try:
            self._cleanup_temp()
            user_lang = None

            if manual_text:
                user_text = manual_text
            else:
                if status_cb: status_cb("Transcribing voice...", "#e65100")
                result = self.whisper_m.transcribe(str(PATHS["tmp_user"]), fp16=False)
                user_text = result["text"].strip()
                user_lang = result.get("language")
                if chat_cb: chat_cb("user", user_text)

            if not user_text:
                return False

            detected = self._detect_name(user_text)
            if detected:
                self.persona_name = detected
                if chat_cb: chat_cb("system", f"[Persona set to {detected}]")

            if status_cb: status_cb("Searching memory...", "#0277bd")
            context = self.rag.search(user_text)

            behavior = ""
            if PATHS["behavior"].exists():
                behavior = PATHS["behavior"].read_text(encoding="utf-8")

            if status_cb: status_cb("Thinking...", "#e65100")

            prompt = self._build_persona_prompt(behavior, context, user_text, lang, user_lang)

            resp = ollama.generate(model="qwen2.5:3b", prompt=prompt)
            bot_text = resp["response"].strip()

            if chat_cb: chat_cb("bot", bot_text)

            if status_cb: status_cb("Cloning voice...", "#2e7d32")
            self.tts_manager.generate_tts(bot_text, lang, str(PATHS["tmp_bot"]))

            if status_cb: status_cb("Syncing lips...", "#2e7d32")
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
        subprocess.run(cmd, shell=True, check=True, env={**os.environ, "OMP_NUM_THREADS": "6"},
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    def save_session_log(self, history_list):
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
            print(f"Session saved to {filepath}")
        except Exception as e:
            print(f"Error saving session: {e}")
