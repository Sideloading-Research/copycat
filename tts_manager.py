"""Voice cloning manager using Pocket-TTS."""

import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from pocket_tts import TTSModel, export_model_state

LANG_CONFIG = {
    "es": "spanish_24l",
    "en": "english",
}

PRESET_VOICES = {
    "es": "lola",
    "en": "alba",
}

VOICE_CLONING_HELP = (
    "Voice cloning weights not downloaded.\n"
    "  1. Accept terms at https://huggingface.co/kyutai/pocket-tts\n"
    "  2. Install HF CLI: curl -LsSf https://hf.co/cli/install.sh | bash\n"
    "  3. Login: hf auth login\n"
    "  Falling back to preset voice '{}'."
)


class TTSManager:
    """Manages Pocket-TTS model lifecycle and cached voice states."""

    def __init__(self, voice_es_path: str, voice_en_path: str,
                 latent_es_path: str = "", latent_en_path: str = ""):
        self.models = {}
        self.voice_states = {}

        self._load_voice_states(voice_es_path, voice_en_path)

    def _get_model(self, language: str) -> TTSModel:
        if language not in self.models:
            cfg = LANG_CONFIG[language]
            self.models[language] = TTSModel.load_model(
                language=cfg,
                quantize=False,
                lsd_decode_steps=5,
                temp=0.7,
            )
        return self.models[language]


    def _extract_state(self, lang: str, wav_path: str):
        safe_path = os.path.join("voices", f"{lang}_pocket.safetensors")
        if os.path.exists(safe_path):
            model = self._get_model(lang)
            return model.get_state_for_audio_prompt(safe_path)

        model = self._get_model(lang)

        try:
            state = model.get_state_for_audio_prompt(wav_path, truncate=True)
            try:
                export_model_state(state, safe_path)
            except Exception:
                pass
            return state
        except ValueError:
            preset = PRESET_VOICES.get(lang)
            msg = VOICE_CLONING_HELP.format(preset)
            print(f"  ⚠️  {msg}")
            if preset:
                return model.get_state_for_audio_prompt(preset)
            raise

    def _load_voice_states(self, voice_es_path: str, voice_en_path: str):
        configs = [
            ("es", voice_es_path),
            ("en", voice_en_path),
        ]
        for lang, wav_path in configs:
            if os.path.exists(wav_path):
                self.voice_states[lang] = self._extract_state(lang, wav_path)
            else:
                model = self._get_model(lang)
                self.voice_states[lang] = model.get_state_for_audio_prompt(
                    PRESET_VOICES[lang])

    def generate_tts(self, text: str, language: str, output_path: str):
        if language not in self.voice_states:
            raise ValueError(f"No voice state for language '{language}'.")

        model = self._get_model(language)
        audio = model.generate_audio(self.voice_states[language], text)
        audio = audio.numpy()
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
        import soundfile as sf
        sf.write(output_path, audio, model.sample_rate, subtype="PCM_16")
