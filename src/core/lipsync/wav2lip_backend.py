"""Wav2Lip backend — lip-sync a face image to audio via subprocess.

Invokes ``inference.py`` from the vendored Wav2Lip fork as a
subprocess.  Each call loads the model from disk (~5-10 s overhead).
"""

import os
import subprocess
import sys
from src.config import cfg


class Wav2LipBackend:
    """Lip-sync wrapper that shells out to the Wav2Lip inference script.

    Usage::

        lipsync = Wav2LipBackend()
        lipsync.render("/path/to/face.jpeg", "/path/to/audio.wav", "/tmp/out.mp4")
    """

    def __init__(self):
        self._script = str(cfg.wav2lip_script)
        self._checkpoint = str(cfg.wav2lip_pth)

    def render(self, face_path: str, audio_path: str, output_path: str) -> None:
        cmd = (
            f"{sys.executable} {self._script} "
            f"--checkpoint_path {self._checkpoint} "
            f"--face {face_path} "
            f"--audio {audio_path} "
            f"--outfile {output_path} "
            f"{'--nosmooth' if cfg.lipsync_nosmooth else ''}"
        )
        subprocess.run(
            cmd,
            shell=True,
            check=True,
            env={**os.environ, "OMP_NUM_THREADS": str(cfg.lipsync_threads)},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
