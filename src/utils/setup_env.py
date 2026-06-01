"""Runtime environment configuration and compatibility patches.

Importing this module performs one-time setup:
- Suppresses noisy library warnings.
- Limits OpenMP / MKL threads to prevent CPU oversubscription.
- Patches ``torch.load`` for compatibility with legacy weight files.
"""

import os
import warnings
import torch

warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message=".*FutureWarning.*")

# ── Thread limits (2 instead of 8 to avoid thermal throttling) ──
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
os.environ.setdefault("OLLAMA_NUM_THREADS", "2")

# ── Torch 2.6+ compatibility patch ──
# PyTorch 2.6 introduced ``weights_only=True`` as default, which rejects
# legacy pickle formats used by older XTTS / Wav2Lip checkpoints.  This
# monkey-patch forces ``weights_only=False`` so those checkpoints load.
_orig_load = torch.load


def _patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)


torch.load = _patched_load

print("Configured CPU environment and Torch patches.")
