# Teaching: Refactorización de Copycat — Fase 1

## Principios Aplicados

### 1. Separación de Responsabilidades (Single Responsibility Principle)

**Antes:** `CopycatEngine` hacía TODO:
- Orquestar el pipeline
- Construir prompts
- Detectar nombres
- Loguear sesiones
- Sincronizar behavior
- Limpiar temp
- Persistir sesiones

**Después:** Cada responsabilidad tiene su propia clase:

```
CopycatEngine (facade)
├── PipelineOrchestrator  → coordina el flujo STT→RAG→LLM→TTS→lip-sync
├── PromptBuilder         → construye el prompt del LLM
├── NameDetector          → extrae nombres del texto
├── SessionLogger         → persiste stats y transcripciones
├── WhisperSTT            → speech-to-text
├── OllamaLLM             → LLM via Ollama
├── XTTSBackend           → voice cloning
├── ChromaVectorDB        → RAG vectorial
└── Wav2LipBackend        → lip-sync
```

**Por qué importa:** Cada clase tiene una razón para cambiar. Si quieres cambiar de Whisper a faster-whisper, solo tocas `WhisperSTT`. Si quieres cambiar de ChromaDB a FAISS, solo tocas `ChromaVectorDB`.

---

### 2. Inyección de Dependencias (Dependency Injection)

**Antes:** El engine importaba directamente `whisper`, `ollama`, etc.

```python
# ANTES — acoplamiento rígido
import whisper
import ollama
result = whisper.transcribe(...)
resp = ollama.generate(...)
```

**Después:** Las dependencias se reciben por constructor.

```python
# DESPUÉS — desacoplado
class PipelineOrchestrator:
    def __init__(self, stt: STTBackend, llm: LLMBackend, ...):
        self.stt = stt
        self.llm = llm
```

**Por qué importa:** Ahora puedes pasar implementaciones distintas sin modificar el orquestador. Quieres probar con OpenAI? Creas `OpenAILLM()` y lo pasas. Quieres un mock para tests? Pasas un `MockLLM()`.

---

### 3. Programación por Interfaces (Protocolos)

Usamos `Protocol` de `typing` (duck typing estático) en lugar de clases abstractas:

```python
@runtime_checkable
class STTBackend(Protocol):
    def transcribe(self, audio_path: str, fp16: bool = False) -> str: ...
```

**Por qué Protocol y no ABC:** No necesitas herencia. Cualquier objeto con un método `transcribe(audio_path, fp16)` **es** un STTBackend. Esto sigue el principio de "duck typing" de Python pero con verificación en runtime si usas `isinstance()`.

---

### 4. Configuración Centralizada

**Antes:** 50+ valores hardcodeados en 10 archivos.

**Después:** Un solo archivo `config.py` con un dataclass `Config`.

```python
@dataclass
class Config:
    llm_model: str = "gemma3:4b"
    rag_chunk_size: int = 800
    tts_threads: int = 2
    ...
```

El singleton `cfg` se importa donde sea necesario:
```python
from src.config import cfg
print(cfg.llm_model)
```

**Por qué importa:** Cambiar cualquier parámetro = editar 1 archivo. No más buscar `chunk_size=800` en múltiples lugares.

---

### 5. Separación View/Controller (MVC ligero)

**Antes:** `MainWindow` manejaba layout, eventos Y lanzaba threads del pipeline.

**Después:**
- `MainWindow` → solo dibuja widgets y reenvía eventos al controller.
- `PipelineController` → maneja la lógica: load models, process text, record voice.

```python
# main_window.py — SOLO VIEW
self.controller.process_text(msg, lang, chat_cb=..., status_cb=...)

# controller.py — SOLO LÓGICA
class PipelineController:
    def process_text(self, text, lang, chat_cb, status_cb):
        # lanza thread, ejecuta pipeline, notifica vía callbacks
```

**Por qué importa:** Puedes construir una CLI o API web reutilizando `PipelineController` sin tocar una línea de Tkinter.

---

### 6. Thread Safety

**Antes:** `AudioHandler` usaba `self.recording = False` (flag simple sin locks).

**Después:** Usa `threading.Event` para el flag y `threading.Lock` para el buffer:

```python
self._recording = threading.Event()
self._lock = threading.Lock()

@property
def recording(self) -> bool:
    return self._recording.is_set()

@recording.setter
def recording(self, value: bool):
    if value: self._recording.set()
    else: self._recording.clear()
```

**Por qué importa:** Dos hilos no pueden pisarse mutuamente. El flag es atómico (Event), el buffer está protegido por Lock.

---

### 7. Error Handling en Splash

**Antes:** El thread de carga no tenía try/except → si un modelo fallaba, el splash se quedaba congelado para siempre.

**Después:** Todo el bloque de carga está envuelto en try/except:

```python
def _load_all_models(splash, app):
    try:
        ...
    except Exception as e:
        splash.set_progress(0.0, f"Failed: {e}")
        splash.after(3000, splash.close)
```

**Por qué importa:** El usuario nunca ve una pantalla congelada. Si algo falla, el error se muestra y la app se cierra limpiamente.

---

### 8. Hard Cut of TTS/Lip-sync When Microphone is Disabled

**Before:** The `mic_enabled` flag existed in `PipelineController` and `MainWindow`, but never reached the pipeline. Even with the microphone button off, sending text through chat still ran:
- TTS (XTTS v2 with ~1.8 GB RAM model)
- Lip-sync (Wav2Lip with ~5-10 s CPU per call)

Audio and video files were generated but never played back (`_on_pipeline_complete` already checked `mic_enabled`). Complete computational waste.

**After:** The `mic_enabled` flag propagates from `PipelineController` into `CopycatEngine.run_pipeline()` and from there to `PipelineOrchestrator.run_turn()`. The orchestrator wraps TTS and lip-sync blocks in `if mic_enabled:`:

```python
# orchestrator/pipeline.py
def run_turn(self, ..., mic_enabled=True):
    ...
    if mic_enabled:
        self.tts.synthesize(bot_text, lang, str(cfg.tmp_bot))
    if mic_enabled:
        self.lipsync.render(str(cfg.face_img), str(cfg.tmp_bot), str(cfg.tmp_video))
```

**Why it matters:** XTTS v2 weighs ~1.8 GB in RAM and Wav2Lip takes ~5-10 s per inference. Running them unnecessarily when the user only wants text is an avoidable waste of energy, time and resources. With this change the cut is real: no audio means no TTS or lip-sync, period.

---

### 9. Silence Detection (Energy-Based VAD)

**Before:** Whisper received every recording, even total silence. The speech-to-text model would process the full audio and could take seconds or return empty text, leaving the UI stuck on "Transcribing voice..." with no feedback.

**After:** Before invoking Whisper, the RMS energy of the audio file is computed. If it falls below a configurable threshold (`stt_silence_threshold = 0.02` in `config.py`), it is considered silence and transcription is skipped entirely:

```python
# orchestrator/pipeline.py
@staticmethod
def _is_silence(audio_path: str, threshold: float = 0.02) -> bool:
    data, _ = sf.read(audio_path)
    if len(data) == 0:
        return True
    rms = np.sqrt(np.mean(data**2))
    return float(rms) < threshold
```

**Why it matters:** Whisper is a large model (even `tiny`) that consumes CPU and time. Transcribing silence wastes resources and confuses the user. Detecting silence by energy is dirt-cheap (milliseconds) compared to seconds of Whisper, and the threshold is configurable to adapt to the environment.

---

### 10. Per-Interaction Performance Report in CLI

Every pipeline turn prints a breakdown to the terminal with:
- Time per stage (STT, RAG, LLM, TTS, lip-sync)
- RAM and CPU usage (via `psutil`, optional)
- CPU temperature (Linux only, via `psutil.sensors_temperatures()`)

Example:
```
────────────────────────────────────────────────
 Turn complete in 3.42 s
  ├─ STT       0.85 s
  ├─ RAG       0.12 s
  ├─ LLM       1.50 s
  ├─ TTS       0.95 s
  └─ Lip-sync  0.78 s
  │
  ├─ RAM  42.3%  (5.9 / 14.0 GB)
  ├─ CPU  35.2%
  └─ Temp 72.3°C
────────────────────────────────────────────────
```

**Library used: `psutil`** — De-facto standard for system monitoring in Python. Cross-platform (Linux, Windows, macOS), well documented, no heavy dependencies. `psutil` is optional: if not installed, the report prints without HW data and suggests `pip install psutil`.

**Why it matters:** Knowing how long each stage takes helps identify bottlenecks (e.g. LLM dominates the time). HW monitoring helps detect overheating or RAM saturation without external tools.

---

## Estructura Final (después de Fase 1)

```
src/
├── config.py                          ← NUEVO: Config centralizada
├── core/
│   ├── interfaces.py                  ← NUEVO: Protocolos STT, LLM, TTS, VectorDB, LipSync
│   ├── engine.py                      ← REFACTOR: facade que wirea backends
│   ├── audio.py                       ← REFACTOR: thread-safe con Event + Lock
│   ├── location.py                    ← REFACTOR: imports limpios
│   ├── stt/whisper_backend.py         ← NUEVO: Whisper implementa STTBackend
│   ├── llm/ollama_backend.py          ← NUEVO: Ollama implementa LLMBackend
│   ├── tts/xtts_backend.py            ← NUEVO: XTTS v2 implementa TTSBackend
│   ├── vector_db/chroma_backend.py    ← NUEVO: ChromaDB implementa VectorDB
│   └── lipsync/wav2lip_backend.py     ← NUEVO: Wav2Lip implementa LipSyncBackend
├── orchestrator/
│   ├── pipeline.py                    ← NUEVO: PipelineOrchestrator
│   ├── prompt_builder.py              ← NUEVO: construye prompts
│   ├── name_detector.py              ← NUEVO: detecta nombres
│   └── session_logger.py             ← NUEVO: loggea stats/sesiones
└── ui/
    ├── controller.py                  ← NUEVO: PipelineController (desacoplado de UI)
    ├── main_window.py                 ← REFACTOR: solo View
    ├── splash.py                      ← igual
    └── settings.py                    ← igual
```

---

## Cómo Probar que Funciona

```bash
# 1. Verificar que todos los módulos importan correctamente
cd /path/to/copycat
source venv/bin/activate
python3 -c "
from src.config import cfg
from src.core.interfaces import STTBackend
from src.core.stt.whisper_backend import WhisperSTT
from src.core.llm.ollama_backend import OllamaLLM
from src.core.tts.xtts_backend import XTTSBackend
from src.core.vector_db.chroma_backend import ChromaVectorDB
from src.core.lipsync.wav2lip_backend import Wav2LipBackend
from src.orchestrator.pipeline import PipelineOrchestrator
from src.core.engine import CopycatEngine
from src.ui.controller import PipelineController
print('OK — todos los módulos importan correctamente')
"

# 2. Verificar que los protocolos se cumplen
python3 -c "
from src.core.interfaces import STTBackend, LLMBackend, TTSBackend, VectorDB, LipSyncBackend
from src.core.stt.whisper_backend import WhisperSTT
from src.core.llm.ollama_backend import OllamaLLM
from src.core.tts.xtts_backend import XTTSBackend
from src.core.vector_db.chroma_backend import ChromaVectorDB
from src.core.lipsync.wav2lip_backend import Wav2LipBackend

assert isinstance(WhisperSTT(), STTBackend)
assert isinstance(OllamaLLM(), LLMBackend)
assert isinstance(XTTSBackend(), TTSBackend)
assert isinstance(ChromaVectorDB(), VectorDB)
assert isinstance(Wav2LipBackend(), LipSyncBackend)
print('OK — todos los backends cumplen los protocolos')
"

# 3. Ejecutar la app (requiere display)
./run.sh
```

---

## Próximos Pasos (Fase 2+)

| Fase | Qué | Por qué |
|------|-----|---------|
| 2 | Wav2Lip como proceso long-lived | Eliminar overhead de 5-10s por turno cargando modelo |
| 2 | Sliding window de conversación | El LLM necesita contexto de turns anteriores |
| 3 | Hybrid RAG (dense + BM25) | Mejor recall en búsqueda |
| 3 | Cargar/descargar modelos on-demand | XTTS v2 pesa 1.8 GB en RAM → descargar cuando no se usa |
| 4 | Tests unitarios por backend | Garantizar que cada backend funciona independientemente |

---

## Glosario de Términos

| Término | Significado |
|---------|-------------|
| **Protocol** | Interfaz implícita de Python. Cualquier objeto con los métodos definidos "implementa" el protocolo. No requiere herencia. |
| **Dependency Injection** | Pasar las dependencias por constructor en lugar de crearlas dentro de la clase. |
| **MVC** | Model-View-Controller. Patrón que separa datos (model), presentación (view) y lógica (controller). |
| **Facade** | Clase que envuelve un subsistema complejo con una API simple. |
| **DRY** | Don't Repeat Yourself. No duplicar código. |
| **Thread Safety** | Garantía de que el código funciona correctamente cuando múltiples hilos acceden simultáneamente. |
