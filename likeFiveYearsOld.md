# Copycat — Explicación como si tuvieras 5 años

## El proyecto con piezas de Lego

Copycat es un programa que **te clona** para hablar por ti.
Tú dices algo, él busca en tu diario, piensa una respuesta,
y responde con **tu cara y tu voz** moviendo los labios.

---

## La tubería (pipeline) — paso a paso

```
[1] MICRÓFONO     →  transforma aire en electricidad
       │                  (tu voz grabada)
       ↓
[2] Whisper        →  convierte sonido en texto (STT)
       │                  (Speech-To-Text)
       ↓
[3] ChromaDB       →  busca en tu diario lo más parecido
       │                  a lo que preguntaste (RAG)
       ↓
[4] Ollama LLM     →  lee el contexto + tu pregunta
       │                  y escribe una respuesta corta
       ↓
[5] XTTS v2        →  clona tu voz y dice la respuesta
       │                  en voz alta (Text-To-Speech)
       ↓
[6] Wav2Lip        →  mueve los labios de tu foto
       │                  para que sincronicen con el audio
       ↓
[7] ALTAVOCES      →  te habla con tu cara y tu voz
```

---

## Cada pieza y por qué está ahí

| Pieza | ¿Qué hace? | ¿Por qué esta? | Alternativas que podrías usar |
|-------|-----------|----------------|-------------------------------|
| **Whisper tiny** | Convierte tu voz en texto | El más pequeñito → corre rápido en CPU | `whisper base/small` (mejor precisión pero más lento), `faster-whisper` (mismo modelo, 2-3x más rápido en CPU), `speechbrain`, `vosk` |
| **ChromaDB** | Guarda tu diario como "memoria" numérica para buscarla | Base de datos de vectores. Guarda frases como números para buscar "lo más parecido" | `FAISS` (más rápido, sin persistencia automática), `Qdrant`, `Pinecone` (online, no libre), `sqlite-vss` |
| **all-MiniLM-L6-v2** | Convierte texto en números (vectores) | El modelo de embeddings más ligero que funciona bien (80 MB, CPU) | `all-mpnet-base-v2` (mejor calidad, 3x más lento), `e5-small`, `bge-small` |
| **Ollama + Qwen2.5:3b** | Lee contexto del diario + pregunta y escribe respuesta | Ollama maneja el modelo. Qwen2.5:3b es pequeño (3B parámetros) pero bilingüe | `llama3.2:3b`, `phi3:mini` (más rápido), `mistral:7b` (mejor pero más lento), `gemma2:2b` |
| **XTTS v2** | Clona tu voz y dice la respuesta | El único open-source que suena bien y soporta español+inglés | `bark` (suena robótico), `styletts2` (inglés solo), `elevenlabs` (online, no libre) |
| **Wav2Lip** | Mueve los labios de tu foto | El estándar open-source para lip-sync con una sola foto | `wav2lip-hq` (mejor calidad), `sadtalker` (más moderno, más módems) |
| **CustomTkinter** | Dibuja la ventana con botones, avatar, texto | Tkinter modernizado con tema oscuro. Viene con Python | `PyQt6`, `DearPyGui`, `PyWebView` (para interfaz web) |

---

## ¿Qué es lento y qué es rápido?

| Paso | Tiempo típico | ¿Razonable? |
|------|--------------|-------------|
| Whisper tiny (STT) | 1-3 segundos | ✅ Rápidísimo |
| ChromaDB (búsqueda) | < 0.5 segundos | ✅ Instantáneo |
| Ollama (respuesta) | 5-15 segundos | ✅ Aceptable |
| **XTTS v2 (clonar voz)** | **30-90 segundos** | 🐢 **El cuello de botella** |
| Wav2Lip (mover labios) | 15-25 segundos | 🐢 Aceptable |

**XTTS v2 NO es malo.** Es el mejor clonador de voz offline que existe.
Pero tiene un GPT-2 dentro que genera el audio token por token,
como un chat escribiendo letra por letra. En CPU eso es inevitablemente lento.

Para acelerarlo necesitarías:
- Una **GPU** (NVIDIA con 6GB+ VRAM) → XTTS va 10x más rápido
- O aceptar **menos calidad** de voz con otro modelo

---

## Aclaración importante: STT ≠ TTS

**Whisper** (STT) y **XTTS** (TTS) son dos cosas completamente distintas:

```
Whisper     → ESCRIBE lo que oye   (voz → texto)   → STT
XTTS v2     → HABLA lo que lee     (texto → voz)   → TTS
```

**faster-whisper** NO clona voces. Es solo una versión más rápida de Whisper.
Donde Whisper tarda 3s en transcribir, faster-whisper tarda 1s.
Pero no tiene nada que ver con clonar tu voz — eso lo hace XTTS v2.

**Si cambiaras a faster-whisper:** la transcripción iría más rápida,
pero la clonación de voz (XTTS) seguiría tardando lo mismo.

---

## El truco clave de optimización

Los modelos de IA en CPU pasan el tiempo multiplicando tablas enormes de números
(operaciones de matrices). De ahí vienen estos ajustes:

| Variable | ¿Qué hace? |
|----------|-----------|
| `OMP_NUM_THREADS=6` | Usa 6 núcleos de 8 para multiplicar matrices |
| `MKL_NUM_THREADS=6` | Lo mismo para Intel MKL (si PyTorch lo usa) |
| `KMP_BLOCKTIME=0` | Los hilos se duermen al terminar → no se pelean |
| `KMP_AFFINITY=compact` | Cada hilo anclado a un núcleo fijo → mejor caché |

Poner 6 en un CPU de 8 núcleos deja 2 núcleos libres para que el sistema
operativo, el sonido y la interfaz gráfica no se peleen con la IA.

---

## Búsquedas útiles para encontrar alternativas tú mismo

| Necesitas | Busca en Google |
|-----------|----------------|
| STT más rápido en CPU | "faster-whisper vs whisper CPU benchmark 2025" |
| Embeddings ligeros | "best lightweight sentence transformers CPU 2025" |
| TTS offline que hable español | "XTTS v2 alternatives offline TTS spanish 2025" |
| Lip-sync con foto | "sadTalker vs wav2Lip CPU speed comparison" |
| Base de datos de vectores | "chromaDB vs FAISS comparison 2025" |
| LLM local pequeño y rápido | "best small LLM ollama CPU 2025 reddit" |

La regla de oro: busca siempre `"X vs Y CPU benchmark 2025"`
y mira cuánta RAM pide cada uno. Con 14 GB no puedes tener dos modelos
que pesen 6 GB cada uno.

---

## El programa visto desde arriba

```
app.py (el jefe)
│
├── Al iniciar (en segundo plano)
│   ├── Whisper (STT) ──── EN PARALELO ──── Embeddings (RAG)
│   ├── ChromaDB (índice del diario)
│   └── XTTS v2 (el gordo, 6 GB)
│
├── Cuando pulsas botón
│   ├── Graba audio del micrófono
│   └── Lanze tubería en hilo separado:
│       ├── Whisper  →  texto
│       ├── ChromaDB →  contexto del diario
│       ├── Ollama   →  respuesta
│       ├── XTTS v2  →  archivo de audio .wav
│       ├── Wav2Lip  →  video .mp4 con labios sincronizados
│       └── Reproduce audio + video
│
└── Interfaz gráfica (CustomTkinter)
    ├── Avatar (tu foto o el video con labios moviéndose)
    ├── Barra de estado (verde = listo, rojo = error)
    ├── Caja de log (texto de lo que está pasando)
    └── Botones: Escucha (español) / Listen (inglés)
```
