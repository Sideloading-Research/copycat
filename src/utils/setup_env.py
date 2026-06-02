"""Runtime environment configuration and compatibility patches.

Importing this module performs one-time setup:
- Suppresses noisy library warnings.
- Limits OpenMP / MKL threads (values from ``config.py``).
- Patches ``torch.load`` for compatibility with legacy weight files.
"""

import os
import warnings
import torch
from src.config import cfg

warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message=".*FutureWarning.*")

# Thread limits — read from centralized config.
os.environ.setdefault("OMP_NUM_THREADS", cfg.omp_num_threads)
os.environ.setdefault("MKL_NUM_THREADS", cfg.mkl_num_threads)
os.environ.setdefault("OPENBLAS_NUM_THREADS", cfg.openblas_num_threads)
os.environ.setdefault("KMP_BLOCKTIME", cfg.kmp_blocktime)
os.environ.setdefault("KMP_AFFINITY", cfg.kmp_affinity)
os.environ.setdefault("OLLAMA_NUM_THREADS", cfg.ollama_num_threads)

# Torch 2.6+ compatible: legacy checkpoints need ``weights_only=False``.
_orig_load = torch.load


def _patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)


torch.load = _patched_load

print("Configured CPU environment and Torch patches.")
