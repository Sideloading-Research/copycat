"""
OpenVoice v2 Voice-Cloning Manager.

Loads the ToneColorConverter and MeloTTS base models once, computes speaker 
conditioning latents from reference audio, then caches them in memory.
"""

import os
from pathlib import Path
import torch
import soundfile as sf
import warnings

# Suppress some common OpenVoice warnings
warnings.filterwarnings("ignore")

from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS as MeloTTS

class TTSManager:
    """Manages OpenVoice v2 model lifecycle and cached speaker-latent inference."""

    def __init__(self, voice_es_path: str, voice_en_path: str, latent_es_path: str, latent_en_path: str):
        self.voice_es_path = voice_es_path
        self.voice_en_path = voice_en_path
        # We don't necessarily need the pth latent paths for OpenVoice as we extract them dynamically fast,
        # but we keep the signature for compatibility with app.py
        
        self.device = "cpu"
        self.tone_color_converter = None
        self.melo_models = {}
        self.target_se = {}
        self.source_se = {}
        
        self._init_models()
        self._load_or_compute_latents()

    def _init_models(self):
        """Initializes ToneColorConverter and MeloTTS base models."""
        ckpt_converter = 'checkpoints_v2/converter'
        if not os.path.exists(ckpt_converter):
            print(f"[ERROR] OpenVoice checkpoints not found in {ckpt_converter}. Please run setup.sh.")
            return

        self.tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=self.device)
        self.tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
        
        # Load base MeloTTS models
        # Note: OpenVoice v2 uses 'EN_V2' for English default, 'ES' for Spanish
        self.melo_models['es'] = MeloTTS(language='ES', device=self.device)
        self.melo_models['en'] = MeloTTS(language='EN_V2', device=self.device)

    def _load_or_compute_latents(self):
        """Extracts and caches the target voice embeddings (SE)."""
        if not self.tone_color_converter:
            return

        configs = [
            ("es", self.voice_es_path),
            ("en", self.voice_en_path)
        ]
        
        for lang, voice_path in configs:
            if os.path.exists(voice_path):
                # Extract target speaker embedding
                se, audio_name = se_extractor.get_se(voice_path, self.tone_color_converter, vad=True)
                self.target_se[lang] = se
                
                # Get the default source speaker embedding from MeloTTS for that language
                # OpenVoice v2 comes with base_speakers config, but extracting from a dummy generated audio is the standard way
                dummy_text = "Prueba de audio" if lang == "es" else "Audio test"
                src_path = f"tmp_dummy_{lang}.wav"
                
                # Default speaker ID
                speaker_id = self.melo_models[lang].hps.data.spk2id.get('ES', 0) if lang == 'es' else self.melo_models[lang].hps.data.spk2id.get('EN-Default', 0)
                if not isinstance(speaker_id, int):
                    speaker_id = list(self.melo_models[lang].hps.data.spk2id.values())[0]

                self.melo_models[lang].tts_to_file(dummy_text, speaker_id, src_path, speed=1.0)
                src_se, _ = se_extractor.get_se(src_path, self.tone_color_converter, vad=True)
                self.source_se[lang] = src_se
                
                if os.path.exists(src_path):
                    os.remove(src_path)

    def generate_tts(self, text: str, language: str, output_path: str):
        """Synthesise speech from text using OpenVoice v2 tone color conversion.

        Args:
            text: Text string to vocalise.
            language: Language code ('es' or 'en').
            output_path: Path for the resulting WAV file.
        """
        if language not in self.melo_models:
            raise ValueError(f"Language '{language}' not configured in TTSManager.")
            
        if language not in self.target_se:
            raise ValueError(f"Reference voice for '{language}' missing. Please provide {self.voice_es_path if language=='es' else self.voice_en_path}.")

        # 1. Generate base audio with MeloTTS
        src_path = f"tmp_base_{language}.wav"
        model = self.melo_models[language]
        
        speaker_id = list(model.hps.data.spk2id.values())[0]
        model.tts_to_file(text, speaker_id, src_path, speed=1.0)
        
        # 2. Convert tone to target voice
        with torch.inference_mode():
            self.tone_color_converter.convert(
                audio_src_path=src_path, 
                src_se=self.source_se[language], 
                tgt_se=self.target_se[language], 
                output_path=output_path,
                message="@MyShell"
            )
            
        # Clean up tmp file
        if os.path.exists(src_path):
            os.remove(src_path)
