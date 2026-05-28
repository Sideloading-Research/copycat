# Copycat — Local AI Avatar Chatbot with Lip-Sync

Un asistente personal AI **offline, solo CPU** que te responde con **tu propia cara y voz**.

**Pipeline:** Voz → Whisper (STT) → RAG (tu diario) → LLM (Qwen2.5) → XTTS v2 (clonar voz) → Wav2Lip (sincronizar labios) → Video

---

## Requisitos mínimos

| Hardware | Mínimo |
|----------|--------|
| CPU | 4 núcleos |
| RAM | 10 GB |
| Disco | 10 GB libres |
| GPU | No necesaria |

**Linux** (Debian 12 testeado), Windows, macOS.

---

## Setup en 1 comando

```bash
git clone <este-repo> copycat
cd copycat
bash setup.sh
```

El script lo hace **todo**:
- Instala paquetes del sistema (`ffmpeg`, `python3-venv`, etc.)
- Crea un `venv` e instala todas las dependencias Python
- Descarga automáticamente `wav2lip.pth` (~416 MB)
- Descarga el modelo `qwen2.5:3b` con Ollama
- Prepara las carpetas `voices/` y `diario/`

> **Nota:** El primer uso descarga ~2 GB adicionales (XTTS v2, Whisper, sentence-transformer) automáticamente.

---

## Preparar assets

Antes de ejecutar, coloca estos archivos en la raíz del proyecto:

```
copycat/
├── face.jpeg          ← Tu foto frontal (cara completa, bien iluminada)
├── voices/
│   ├── es.wav         ← Voz referencia español (6-15s, audio limpio)
│   └── en.wav         ← Voz referencia inglés (6-15s, audio limpio)
└── diario/
    ├── 2024-07-15.md  ← Tus entradas de diario en .md
    └── ...
```

**Tips:**
- **Foto:** Frontal, sin obstrucciones, mínimo 256×256
- **Voz:** Graba 6–15s con el mismo micrófono que usarás para hablar
- **Diario:** Cuantas más entradas, mejor responde el RAG

---

## Ejecutar

```bash
source venv/bin/activate
python3 app.py
```

Espera a que aparezca _"Listo. Pulsa Escucha o Listen."_ y presiona el botón.

---

## Personalizar

| Cambio | Dónde |
|--------|-------|
| Modelo LLM | `app.py:79` — cambiar `OLLAMA_MODEL` |
| Voz de referencia | Reemplazar `voices/es.wav` o `voices/en.wav` |
| Foto | Reemplazar `face.jpeg` |
| Conocimiento | Agregar `.md` a `diario/` (se reindexa automáticamente) |

---

## Troubleshooting

| Síntoma | Solución |
|---------|----------|
| _"Face not detected!"_ | Usa foto frontal bien iluminada |
| Wav2Lip crashea | Reduce `--wav2lip_batch_size` en `app.py` |
| Sin micrófono | `python -c "import sounddevice; print(sounddevice.query_devices())"` |
| Error en ChromaDB | Borra `chroma_db/` y reinicia |

---

## Tech Stack

| Componente | Modelo |
|---|---|
| STT | [Whisper tiny](https://github.com/openai/whisper) |
| RAG | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) + [ChromaDB](https://www.trychroma.com/) |
| LLM | [Qwen2.5-3B](https://ollama.ai/library/qwen2.5:3b) via [Ollama](https://ollama.ai/) |
| TTS | [XTTS v2](https://github.com/coqui-ai/TTS) |
| Lip-sync | [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) |
| GUI | [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) |
