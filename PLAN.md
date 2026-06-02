# PLAN DE REFACTORIZACIÓN — COPYCAT

## HARDWARE OBJETIVO

| Componente | Valor |
|---|---|
| CPU | AMD Ryzen 7 4700U (8 cores, laptop) |
| RAM | 14 GB (~4 GB libre con carga típica) |
| OS | Debian 12 Bookworm x86_64 |
| GPU | No |

El proyecto debe caber holgadamente en 14 GB RAM y no asumir GPU.

---

## ESTRUCTURA DEL PROYECTO

```
copycat0/
├── src/
│   ├── main.py              ← entry point
│   ├── doc_loader.py         ← reemplazo de langchain-community (sunset)
│   ├── core/
│   │   ├── engine.py         ← GOD CLASS: 7+ responsabilidades
│   │   ├── rag.py            ← RAGManager (ChromaDB)
│   │   ├── tts_manager.py    ← TTSManager (XTTS v2)
│   │   ├── audio.py          ← AudioHandler (sounddevice)
│   │   └── location.py       ← geolocalización + tiempo
│   ├── ui/
│   │   ├── main_window.py    ← VIEW + CONTROLLER mezclados
│   │   ├── splash.py         ← pantalla de carga
│   │   └── settings.py       ← diálogo de config
│   ├── utils/
│   │   ├── paths.py          ← PATHS centralizado
│   │   └── setup_env.py      ← threads + parches torch
│   ├── tests/                ← 3 tests (1 roto, 1 script, 1 manual)
│   └── Wav2Lip/              ← fork de Wav2Lip (subprocess)
├── data/
│   ├── behavior/behavior.txt
│   ├── journal/              ← 70+ archivos .md (formato inconsistente)
│   ├── voices/{en,es}.wav + .pth
│   ├── picture/face.jpeg
│   └── vector_db/chroma.sqlite3 + .file_hashes.json
├── requirements.txt
├── run.sh / setup.sh
└── likeFiveYearsOld.md       ← documentación técnica detallada
```

---

## DIAGNÓSTICO CRÍTICO

### 🔴 BLOQUEANTES (deben refactorizarse primero)

| # | Problema | Archivo:línea | Impacto |
|---|---|---|---|
| 1 | **CopycatEngine es GOD CLASS** | `engine.py:17` | Orquesta, construye prompts, detecta nombres, loggea, limpia temp, sincroniza behavior, persiste sesiones, chequea assets. 7 responsabilidades en 322 líneas. |
| 2 | **Sin interfaces abstractas** | `engine.py:189-235` | Whisper, Ollama, TTS, Wav2Lip llamados directamente. Cambiar STT/LLM/TTS = editar engine.py. |
| 3 | **GUI = View + Controller** | `main_window.py:349-372` | `_process_pipeline` lanza threads, orquesta pipeline, reproduce audio/video. Para añadir CLI o API hay que duplicar lógica. |
| 4 | **No hay Config** | 10 archivos | Model names, chunk_size (x2), thread counts, rutas, k RAG, max_chars dispersos. `gemma3:4b` hardcoded en `engine.py:235` pero `setup.sh` instala `qwen2.5:3b`. |
| 5 | **Subprocess Wav2Lip** | `engine.py:283-297` | Carga modelo desde cero cada turno (~5-10s overhead). `OMP_NUM_THREADS=6` hardcoded. stdout silenciado → errores invisibles. |

### 🟡 GRAVES

| # | Problema | Archivo:línea | Impacto |
|---|---|---|---|
| 6 | **Splash thread muere en silencio** | `main.py:50` | `load_task` sin try/except. Si modelo falla, splash nunca cierra, app se cuelga. |
| 7 | **AudioHandler thread-unsafe** | `audio.py:17-18` | `recording` flag y `audio_chunks` list sin locks. Race condition si dos pipelines se solapan. |
| 8 | **Dependencias contradictorias** | `requirements.txt:31` vs `Wav2Lip/requirements.txt:5` | `torch~=2.12.0+cpu` vs `torch==1.1.0`. Incompatible. |
| 9 | **Tests no ejecutables** | `tests/` | `test_rag.py` es script sin assert. `test_tts_manager.py` importa `from tts_manager` (path roto). `test_voice.py` manual con prompts interactivos. |
| 10 | **Wav2Lip usa Python embedido** | `inference.py:17-22` | Duplica el monkey-patch de `torch.load` que ya está en `setup_env.py`. |

### 🟢 MEJORABLES

| # | Problema | Archivo | Impacto |
|---|---|---|---|
| 11 | `chunk_size=800` duplicado | `rag.py:129,179` | DRY violation. Si cambia uno y no el otro, inconsistencia. |
| 12 | `import time` duplicado en location | `location.py:26,106` | `from datetime import datetime` + luego `import time as _time`. Sobra. |
| 13 | Nomenclatura archivos journal caótica | `data/journal/` | 4+ formatos distintos: `YYYYMMDD.md`, `YYYY-MM-DD-dia.md`, `Español.md`, con typos. RAG funciona pero es frágil. |
| 14 | Sin memoria de conversación | `engine.py` | Cada turno es stateless. No hay sliding window de contexto. |
| 15 | `likeFiveYearsOld.md` queda grande | ~443 líneas | Mezcla docs técnica, educativa y roadmap. Separar. |

---

## PLAN DE REFACTORIZACIÓN (por orden de impacto)

### Fase 1 — Infraestructura (desacoplamiento base)

```
src/
├── config.py                 ← nuevo: Config dataclass con TODOS los valores
├── core/
│   ├── interfaces.py         ← nuevo: STTBackend, LLMBackend, TTSBackend, VectorDB, LipSync
│   ├── stt/
│   │   └── whisper_backend.py  ← nuevo: implementa STTBackend
│   ├── llm/
│   │   └── ollama_backend.py   ← nuevo: implementa LLMBackend
│   ├── tts/
│   │   └── xtts_backend.py     ← nuevo: implementa TTSBackend (extraído de TTSManager)
│   ├── vector_db/
│   │   └── chroma_backend.py   ← nuevo: implementa VectorDB (extraído de RAGManager)
│   ├── lipsync/
│   │   └── wav2lip_backend.py  ← nuevo: implementa LipSync (envuelve subprocess o API)
│   ├── rag.py                 ← refactor: solo lógica RAG, sin I/O directo
│   ├── tts_manager.py         ← eliminar: reemplazar por xtts_backend.py
│   └── location.py            ← ok, tocar solo lo mínimo
├── orchestrator/
│   ├── pipeline.py            ← nuevo: PipelineOrchestrator con dependencias inyectadas
│   ├── prompt_builder.py      ← nuevo: construye prompts
│   ├── name_detector.py       ← nuevo: detecta nombres
│   └── session_logger.py      ← nuevo: loggea sesiones
├── ui/
│   ├── main_window.py         ← refactor: solo View
│   └── controller.py          ← nuevo: PipelineController separado
└── main.py                    ← refactor: try/except en splash thread
```

1. **Crear `src/config.py`** — `@dataclass Config` con: model names, paths, chunk_size, k RAG, max_chars, thread counts, timeouts. Unificar `gemma3:4b` vs `qwen2.5:3b`.

2. **Definir `src/core/interfaces.py`** — Protocolos:
   - `STTBackend.transcribe(audio_path) -> str`
   - `LLMBackend.generate(prompt, **kwargs) -> str`
   - `TTSBackend.synthesize(text, lang, output_path)`
   - `VectorDB.search(query, k, max_chars) -> str`
   - `LipSync.render(face_path, audio_path, output_path)`

3. **Implementar backends** uno por uno, manteniendo la misma funcionalidad.

4. **Extraer `PipelineOrchestrator`** de CopycatEngine — recibe backends por constructor. Separa `PromptBuilder`, `NameDetector`, `SessionLogger`.

5. **Separar `PipelineController`** de `MainWindow` — controller recibe señales UI, lanza pipeline. MainWindow solo dibuja.

### Fase 2 — Robustez

6. **Thread safety en AudioHandler** — `threading.Event` para recording, `threading.Lock` para audio_chunks.

7. **Wrap splash thread** — `try/except` en `main.py:45-46` con callback a splash de error.

8. **Wav2Lip como proceso long-lived** — socket o API local en vez de subprocess cada turno. O al menos cachear el modelo en un proceso servidor.

### Fase 3 — Escalabilidad y rendimiento

9. **Conversación con sliding window** — últimos N intercambios inyectados en prompt.

10. **Hybrid RAG** — BM25 + dense vector (ya mencionado en README TODO).

11. **Cargar/descargar modelos bajo demanda** — XTTS v2 pesa ~1.8 GB. Solo mantener en RAM cuando se necesita.

12. **VAD** — Voice Activity Detection para stop automático de grabación.

13. **~~Auto-detección de idioma~~** — **DESCARTADO por ahora.** El sistema debe ser agnóstico al idioma dentro de los límites de lenguajes clonables con XTTS y comprensibles por el LLM (Qwen/Gemma soportan multidioma, no es problema). En lugar de auto-detectar, el frontal debe generar dinámicamente las opciones de voz de salida disponibles y asociar la respuesta traducida al idioma seleccionado. No se ejecuta en esta refactorización.

### Fase 4 — Calidad y mantenibilidad

14. **Tests unitarios** para cada backend (mock dependencies pesadas).

15. **Unificar formato de archivos journal** — normalizar a `YYYY-MM-DD.md`.

16. **Dependency injection en todos los constructores** — facilitar testing y swapping.

---

## MÉTRICAS POST-REFACTORIZACIÓN ESPERADAS

| Métrica | Antes | Después |
|---|---|---|
| engine.py LOCs | 322 | ~80 (solo orquestación pura) |
| Archivos con config hardcoded | 10 | 1 (`config.py`) |
| Tiempo cambio de STT/LLM/TTS | editar engine.py | cambiar 1 línea en constructor |
| Posibilidad de CLI/server | no | sí (sin GUI) |
| Tests unitarios | 0 válidos | 1 por backend |
| Thread safety | no | sí |

---

## Pruebas 1 — Fallos detectados y corregidos

### Fallo 1: `_play_after_pipeline` sondea archivos en vez de usar callback

**Síntoma:** `_toggle_voice_interaction` lanza `_play_after_pipeline()` inmediatamente después de `stop_recording()`, pero el pipeline corre en un hilo separado. El método `_wait_and_play` hace *polling* (`while not exists: sleep(0.5)`) para detectar cuándo termina el pipeline, lo que añade latencia y consume CPU innecesariamente.

**Corrección:** Se añadió el parámetro `on_complete` a `process_text()`, `stop_recording()` y `_run_pipeline()`. Este callback se invoca desde `_run_pipeline` con el booleano `success` al finalizar. Desde `main_window`, `_on_pipeline_complete(success)` reproduce audio+vídeo directamente si `success` y `mic_enabled`, eliminando el *polling*.

**Archivos afectados:**
- `src/ui/controller.py:68,87,118`
- `src/ui/main_window.py:245,277,284`

### Fallo 2: `import threading` dentro de funciones

**Síntoma:** `_play_after_pipeline` importaba `threading` dentro del closure `_wait_and_play`, y `_send_text_manual` no importaba `threading` en absoluto. Esto funciona pero es mala práctica y puede pasar desapercibido si se mueve el código.

**Corrección:** Se movió `import threading` al inicio de `main_window.py`.

**Archivos afectados:**
- `src/ui/main_window.py:7`

### Fallo 3: Etiquetas STT/lip-sync visibles con micrófono desactivado

**Síntoma:** Cuando `mic_enabled=False`, el pipeline de texto escrito sigue ejecutándose pero mostraba etiquetas irrelevantes como «Transcribing voice...» y «Syncing lips...» en la barra de estado.

**Corrección:** Se añadió un filtro en `_update_status` que descarta esas etiquetas cuando `self.mic_enabled` es `False`.

**Archivos afectados:**
- `src/ui/main_window.py:198`

---

## PLAN 3 — Hard cut of TTS/lip-sync when microphone is disabled

**Problem:** When `mic_enabled=False` the pipeline still ran TTS (XTTS v2, ~1.8 GB RAM) and lip-sync (Wav2Lip, ~5-10 s per call) even though the generated audio/video was never played. Bug 3 only hid the status labels, but the computational cost remained.

**Fix:** Added the `mic_enabled` parameter (default `True`) that propagates from `PipelineController` → `CopycatEngine.run_pipeline()` → `PipelineOrchestrator.run_turn()`. When `False`, the orchestrator skips the TTS and lip-sync blocks entirely, avoiding:

- XTTS v2 voice synthesis
- Wav2Lip video generation
- Writing temporary audio/video files

**Affected files:**
- `src/orchestrator/pipeline.py:49-53,141,149`
- `src/core/engine.py:111,125`
- `src/ui/controller.py:123`

### Bug 4: TTS and lip-sync ran even with microphone disabled

**Symptom:** With `mic_enabled=False` and sending manual text, the pipeline still generated audio (TTS) and video (lip-sync) in the background, wasting CPU, RAM and time.

**Fix:** Added the `mic_enabled` flag to the signature of `PipelineOrchestrator.run_turn()`, `CopycatEngine.run_pipeline()` and passes it from `PipelineController._run_pipeline()`. The orchestrator only runs TTS and lip-sync when `mic_enabled=True`.

**Affected files:**
- `src/orchestrator/pipeline.py:53,141,149`
- `src/core/engine.py:111,125`
- `src/ui/controller.py:123`

---

## PLAN 4 — Silence detection, CLI report and HW monitoring

### Bug 5: Silent audio transcription hangs indefinitely

**Symptom:** When sending silent audio (no voice, just background noise or pure silence), Whisper would process seemingly forever. The UI status stayed stuck on "Transcribing voice..." and would not recover until a new message was sent.

**Root cause:** There was no audio energy check before feeding the entire file to Whisper. With low or zero-energy audio, Whisper could take very long or return empty text with no feedback.

**Fix:** Added the `_is_silence(audio_path, threshold)` method in `PipelineOrchestrator` that computes the RMS (root mean square) of the audio and compares it to a configurable threshold (`stt_silence_threshold` in `config.py`). If the audio falls below the threshold, Whisper is skipped entirely and the status is restored to "Ready".

**Affected files:**
- `src/orchestrator/pipeline.py:101-104,216-230`
- `src/config.py:45`

### New feature: Per-interaction timing report in CLI

**Description:** Every time the pipeline completes a turn (input → processing → output), a time breakdown is printed to the terminal in seconds:

```
────────────────────────────────────────────────
 Turn complete in 3.42 s
  ├─ STT       0.85 s
  ├─ RAG       0.12 s
  ├─ LLM       1.50 s
  ├─ TTS       0.00 s
  └─ Lip-sync  0.00 s
  │
  ├─ RAM  42.3%  (5.9 / 14.0 GB)
  ├─ CPU  35.2%
  └─ Temp 72.3°C
────────────────────────────────────────────────
```

**Implementation:** Timestamps are taken before and after each pipeline stage (STT, RAG, LLM, TTS, lip-sync) and stored in the `stats` dict. The `_print_turn_report()` function formats them and writes to `stdout`.

**Affected files:**
- `src/orchestrator/pipeline.py:105-107,236-281`

### New feature: HW resource monitoring (RAM, CPU, temperature)

**Description:** Alongside the timing report, hardware resource statistics are shown using `psutil` (optional, cross-platform, well-documented library). If `psutil` is not installed, a hint is printed without failing. If `psutil` is present but cannot read temperature (platform-dependent), that value is silently skipped.

**Library used:** [`psutil`](https://psutil.readthedocs.io/) — supports Linux, Windows and macOS for RAM and CPU. Temperature (`sensors_temperatures()`) only works on Linux, fails silently on other platforms.

**Affected files:**
- `src/orchestrator/pipeline.py:12-16,257-273`
- `requirements.txt:35`
