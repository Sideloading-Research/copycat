"""Re-export of ``XTTSBackend`` for backward compatibility.

New code should import directly from ``src.core.tts.xtts_backend``.
"""

from src.core.tts.xtts_backend import XTTSBackend

# Kept for compatibility.
TTSManager = XTTSBackend
