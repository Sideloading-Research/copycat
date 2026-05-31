import os
import warnings
import torch

warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message=".*FutureWarning.*")

# ── THREAD CONFIGURATION (Optimization for Ryzen 7 4700U) ──
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
os.environ.setdefault("OLLAMA_NUM_THREADS", "6")

# ── TORCH 2.6+ COMPATIBILITY PATCH ──
# Prevents errors when loading legacy XTTS/Wav2Lip weights
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)

torch.load = _patched_load

print("Configured CPU environment and Torch patches.")
