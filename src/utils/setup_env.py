import os
import torch

# ── THREAD CONFIGURATION (Optimization for Ryzen 7 4700U) ──
# We reserve 2 cores for the GUI system and assign 6 to the AI process..
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
os.environ.setdefault("OLLAMA_NUM_THREADS", "6")

# ── PARCHE DE COMPATIBILIDAD TORCH 2.6+ ──
# Esto evita errores al cargar pesos antiguos de XTTS/Wav2Lip
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)

torch.load = _patched_load

print("✅ Configured CPU environment and Torch patches.")