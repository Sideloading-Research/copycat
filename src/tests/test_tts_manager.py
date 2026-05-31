import os
import sys
import unittest
import shutil
from pathlib import Path
import torch
import soundfile as sf

# Patch torch.load for PyTorch 2.6
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from tts_manager import TTSManager

class TestTTSManager(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path("/home/marco/Escritorio/copycat")
        self.voices_dir = self.base_dir / "voices"
        self.voice_es = str(self.voices_dir / "es.wav")
        self.voice_en = str(self.voices_dir / "en.wav")
        
        # Temporary directory for testing cache
        self.temp_cache_dir = self.base_dir / "temp_test_cache"
        self.temp_cache_dir.mkdir(exist_ok=True)
        self.latent_es_path = self.temp_cache_dir / "es_latents.pth"
        self.latent_en_path = self.temp_cache_dir / "en_latents.pth"
        
        # Output audio file
        self.out_wav = self.base_dir / "_test_tts_manager_out.wav"
        if self.out_wav.exists():
            self.out_wav.unlink()

    def tearDown(self):
        # Clean up temporary tests files
        if self.temp_cache_dir.exists():
            shutil.rmtree(self.temp_cache_dir)
        if self.out_wav.exists():
            self.out_wav.unlink()

    def test_tts_manager_caching_and_inference(self):
        # 1. Initialize manager (first run, should generate cache)
        print("\n[TEST] Initializing TTSManager (first run - generation)...")
        manager = TTSManager(
            voice_es_path=self.voice_es,
            voice_en_path=self.voice_en,
            latent_es_path=str(self.latent_es_path),
            latent_en_path=str(self.latent_en_path)
        )
        
        # Verify latents are cached in memory
        self.assertIn("es", manager.latents)
        self.assertIn("en", manager.latents)
        self.assertIn("gpt_cond_latent", manager.latents["es"])
        self.assertIn("speaker_embedding", manager.latents["es"])
        
        # Verify latents are saved to disk
        self.assertTrue(self.latent_es_path.exists(), "ES latent file was not created on disk.")
        self.assertTrue(self.latent_en_path.exists(), "EN latent file was not created on disk.")
        
        # 2. Run inference using Spanish latents
        print("[TEST] Running inference on Spanish...")
        test_text = "Prueba de generación rápida con latentes cached."
        manager.generate_tts(
            text=test_text,
            language="es",
            output_path=str(self.out_wav)
        )
        
        self.assertTrue(self.out_wav.exists(), "Output WAV file was not created.")
        self.assertGreater(self.out_wav.stat().st_size, 0, "Output WAV file is empty.")
        
        # Check sampling rate/audio valid
        data, samplerate = sf.read(self.out_wav)
        self.assertEqual(samplerate, 24000, "WAV samplerate is not 24000.")
        self.assertGreater(len(data), 0, "WAV has no audio samples.")
        
        # 3. Re-initialize manager (second run, should load from cache)
        print("[TEST] Re-initializing TTSManager (second run - loading cached)...")
        import time
        start_time = time.time()
        manager_cached = TTSManager(
            voice_es_path=self.voice_es,
            voice_en_path=self.voice_en,
            latent_es_path=str(self.latent_es_path),
            latent_en_path=str(self.latent_en_path)
        )
        duration = time.time() - start_time
        print(f"[TEST] Loading cached latents took {duration:.4f} seconds.")
        
        # Verify it still works and has latents loaded
        self.assertIn("es", manager_cached.latents)
        self.assertIn("en", manager_cached.latents)

if __name__ == "__main__":
    unittest.main()
