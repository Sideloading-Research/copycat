"""Pipeline orchestrator — coordinates STT → RAG → LLM → TTS → lip-sync.

All external dependencies are injected via constructor so the
orchestrator never imports concrete backends.
"""

import time
import datetime

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

import numpy as np
import soundfile as sf

from src.config import cfg
from src.core.interfaces import STTBackend, LLMBackend, TTSBackend, VectorDB, LipSyncBackend
from src.core.location import get_context_string, has_internet
from src.orchestrator.prompt_builder import PromptBuilder
from src.orchestrator.name_detector import NameDetector
from src.orchestrator.session_logger import SessionLogger


class PipelineOrchestrator:
    """Coordinates a single turn of the avatar pipeline.

    Usage::

        orch = PipelineOrchestrator(stt=..., llm=..., tts=..., vector_db=..., lipsync=...)
        orch.run_turn(lang="es")
    """

    def __init__(
        self,
        stt: STTBackend,
        llm: LLMBackend,
        tts: TTSBackend,
        vector_db: VectorDB,
        lipsync: LipSyncBackend,
        prompt_builder: PromptBuilder | None = None,
        name_detector: NameDetector | None = None,
        logger: SessionLogger | None = None,
    ):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.vdb = vector_db
        self.lipsync = lipsync
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.name_detector = name_detector or NameDetector()
        self.logger = logger or SessionLogger()
        self.persona_name: str | None = None

    def run_turn(
        self,
        lang: str,
        manual_text: str | None = None,
        mic_enabled: bool = True,
        status_cb=None,
        chat_cb=None,
    ) -> bool:
        """Execute one full pipeline turn.

        Parameters
        ----------
        lang : str
            Output language code (e.g. ``"es"``, ``"en"``).
        manual_text : str or None
            If provided, skip STT and use this text directly.
        status_cb : callable or None
            Called with ``(message, colour)`` for status updates.
        chat_cb : callable or None
            Called with ``(role, text)`` to append to the chat transcript.

        Returns
        -------
        bool
            ``True`` on success, ``False`` on error.
        """
        t_start = time.time()
        stats = {
            "timestamp": datetime.datetime.now().isoformat(),
            "lang": lang,
            "n_inferences": 0,
        }

        try:
            self._cleanup_temp()
            user_lang = None

            # ── STT ──────────────────────────────────────────────
            if manual_text:
                user_text = manual_text
            else:
                self._status(status_cb, "Transcribing voice...", "#e65100")
                # Skip Whisper entirely when the recording is silence.
                if self._is_silence(str(cfg.tmp_user), cfg.stt_silence_threshold):
                    self._status(status_cb, "Ready", "#2e7d32")
                    return False
                t_stt = time.time()
                user_text = self.stt.transcribe(str(cfg.tmp_user))
                stats["tt_stt_ms"] = int((time.time() - t_stt) * 1000)
                user_lang = getattr(self.stt, "last_lang", None)
                stats["n_inferences"] += 1
                self._chat(chat_cb, "user", user_text)

            if not user_text:
                self._status(status_cb, "Ready", "#2e7d32")
                return False

            # ── Name detection ───────────────────────────────────
            detected = self.name_detector.detect(user_text)
            if detected:
                self.persona_name = detected
                self._chat(chat_cb, "system", f"[Persona set to {detected}]")

            # ── RAG ──────────────────────────────────────────────
            self._status(status_cb, "Searching memory...", "#0277bd")
            t_rag = time.time()
            context = self.vdb.search_priority(
                user_text,
                priority_source=cfg.behavior_name,
                k=cfg.rag_k,
                max_chars=cfg.rag_priority_max_chars,
            )
            stats["tt_rag_ms"] = int((time.time() - t_rag) * 1000)
            stats["n_chunks"] = cfg.rag_k
            stats["n_inferences"] += 1

            # ── Behaviour ────────────────────────────────────────
            behavior = ""
            if cfg.behavior.exists():
                behavior = cfg.behavior.read_text(encoding="utf-8")
                if len(behavior) > cfg.max_behavior_summary_chars:
                    behavior = behavior[:cfg.max_behavior_summary_chars] + "\n...[summary]"

            # ── Prompt ───────────────────────────────────────────
            self._status(status_cb, "Thinking...", "#e65100")
            context_info = get_context_string(lang=lang)
            prompt = self.prompt_builder.build(
                self.persona_name, behavior, context, user_text, lang, user_lang,
                context_info=context_info,
            )

            # ── LLM ──────────────────────────────────────────────
            t_llm = time.time()
            bot_text = self.llm.generate(prompt)
            stats["tt_llm_ms"] = int((time.time() - t_llm) * 1000)
            stats["n_inferences"] += 1
            self._chat(chat_cb, "bot", bot_text)

            # ── TTS ──────────────────────────────────────────────
            if mic_enabled:
                self._status(status_cb, "Cloning voice...", "#2e7d32")
                t_tts = time.time()
                self.tts.synthesize(bot_text, lang, str(cfg.tmp_bot))
                stats["tt_tts_ms"] = int((time.time() - t_tts) * 1000)
                stats["n_inferences"] += 1

            # ── Lip-sync ─────────────────────────────────────────
            if mic_enabled:
                self._status(status_cb, "Syncing lips...", "#2e7d32")
                t_lip = time.time()
                self.lipsync.render(str(cfg.face_img), str(cfg.tmp_bot), str(cfg.tmp_video))
                stats["tt_lipsync_ms"] = int((time.time() - t_lip) * 1000)
                stats["n_inferences"] += 1

            # ── Logging ──────────────────────────────────────────
            stats["input_text"] = user_text
            stats["output_text"] = bot_text
            stats["tt_total_ms"] = int((time.time() - t_start) * 1000)
            stats["internet"] = has_internet()
            stats["context_info"] = context_info
            self.logger.log_turn(stats)

            # ── Report ───────────────────────────────────────────
            tt = (time.time() - t_start) * 1000
            stats["tt_total_ms"] = int(tt)
            _print_turn_report(stats, mic_enabled)

            self._status(status_cb, "Ready", "#2e7d32")
            return True

        except Exception as e:
            error_msg = f"Pipeline Error: {str(e)}"
            self._chat(chat_cb, "system", error_msg)
            self._status(status_cb, "Error", "#c62828")
            print(error_msg)
            return False

    # ── helpers ──────────────────────────────────────────────────

    def cleanup_temp(self):
        """Delete output temp files from a previous pipeline run."""
        for f in [cfg.tmp_bot, cfg.tmp_video]:
            if f.exists():
                f.unlink(missing_ok=True)

    _cleanup_temp = cleanup_temp  # alias for internal use

    @staticmethod
    def _status(cb, msg, colour):
        if cb:
            cb(msg, colour)

    @staticmethod
    def _chat(cb, role, text):
        if cb:
            cb(role, text)

    @staticmethod
    def _is_silence(audio_path: str, threshold: float = 0.02) -> bool:
        """Return ``True`` if the WAV file is mostly silence.

        Computes the RMS energy of the audio and compares it to a
        low threshold.  Catches read errors gracefully.
        """
        try:
            data, _ = sf.read(audio_path)
            if len(data) == 0:
                return True
            rms = np.sqrt(np.mean(data**2))
            return float(rms) < threshold
        except Exception:
            return False


# ── module-level helpers ──────────────────────────────────────────


def _print_turn_report(stats: dict, mic_enabled: bool):
    """Print timing and optional HW-resource summary to stdout."""
    total_s = stats.get("tt_total_ms", 0) / 1000.0
    sep = "─" * 48

    print(f"\n{sep}")
    print(f" Turn complete in {total_s:.2f} s")

    def _line(label, key):
        ms = stats.get(key)
        if ms is not None:
            print(f"  ├─ {label:<10s} {ms / 1000:.2f} s")

    _line("STT", "tt_stt_ms")
    _line("RAG", "tt_rag_ms")
    _line("LLM", "tt_llm_ms")
    _line("TTS", "tt_tts_ms")
    _line("Lip-sync", "tt_lipsync_ms")

    # ── HW resources (best-effort) ───────────────────────────────
    if _HAS_PSUTIL:
        mem = _psutil.virtual_memory()
        cpu = _psutil.cpu_percent(interval=0.1)
        print(f"  │")
        print(f"  ├─ RAM  {mem.percent:.1f}%  ({mem.used / 1e9:.1f} / {mem.total / 1e9:.1f} GB)")
        print(f"  ├─ CPU  {cpu:.1f}%")
        # Temperature — only available on some platforms.
        try:
            temps = _psutil.sensors_temperatures()
            if "coretemp" in temps:
                t = max(c.current for c in temps["coretemp"])
                print(f"  └─ Temp {t:.1f}°C")
            elif "cpu_thermal" in temps:
                t = temps["cpu_thermal"][0].current
                print(f"  └─ Temp {t:.1f}°C")
        except Exception:
            pass
    else:
        print(f"  │")
        print(f"  └─ Install ``psutil`` for HW stats (RAM/CPU/temp)")

    print(sep)
