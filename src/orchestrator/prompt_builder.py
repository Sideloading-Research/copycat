"""PromptBuilder — constructs the LLM prompt from behaviour, RAG context, time, and rules."""

from src.config import cfg


# ISO 639-1 language code -> English name.
ALL_LANGS: dict[str, str] = {
    "en": "English", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "pt": "Portuguese",
    "nl": "Dutch", "ru": "Russian", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese", "ar": "Arabic",
    "hi": "Hindi", "tr": "Turkish", "pl": "Polish",
    "sv": "Swedish", "da": "Danish", "fi": "Finnish",
    "cs": "Czech", "hu": "Hungarian", "ro": "Romanian",
    "el": "Greek", "he": "Hebrew", "th": "Thai",
    "vi": "Indonesian", "id": "Indonesian", "ms": "Malay",
    "uk": "Ukrainian", "ca": "Catalan", "gl": "Galician",
    "eu": "Basque", "bn": "Bengali", "ta": "Tamil",
    "te": "Telugu", "mr": "Marathi", "gu": "Gujarati",
}


class PromptBuilder:
    """Builds the system prompt for the LLM.

    Combines personality, RAG context, temporal/spatial context,
    and output rules into a single prompt string.
    """

    @staticmethod
    def available_langs(voices_dir) -> dict[str, str]:
        """Return ``{code: name}`` only for languages that have a ``.wav``."""
        codes = {p.stem for p in voices_dir.glob("*.wav")}
        return {k: v for k, v in ALL_LANGS.items() if k in codes}

    def build(
        self,
        persona_name: str | None,
        behavior_text: str,
        rag_context: str,
        user_text: str,
        output_lang: str,
        user_lang: str | None = None,
        context_info: str = "",
    ) -> str:
        name = persona_name or "the person you are simulating"
        out_lang = ALL_LANGS.get(output_lang, output_lang)
        lang_note = (
            f" (spoken in {ALL_LANGS.get(user_lang, user_lang)})"
            if user_lang else ""
        )
        now_block = (
            f"\n## Current situation\n{context_info}"
        ) if context_info else ""
        rules = cfg.prompt_rules.format(out_lang=out_lang)
        return (
            f"You are {name}.\n"
            f"Personality: {behavior_text}\n"
            f"Memories: {rag_context}\n"
            f"{now_block}\n"
            f"## Rules\n{rules}\n"
            f"User{lang_note}: {user_text}\n"
            f"{name}:"
        )
