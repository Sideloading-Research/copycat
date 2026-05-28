#!/usr/bin/env bash
# setup.sh — Copycat: complete one-command setup
# Run: bash setup.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "==> [1/7] System packages"
sudo apt-get update -qq
sudo apt-get install -y \
    python3-venv python3-pip python3-dev \
    ffmpeg git wget curl \
    portaudio19-dev libasound2-dev libsndfile1-dev \
    build-essential libffi-dev libssl-dev \
    libgl1 libglib2.0-0

echo "==> [2/7] Python virtualenv"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools

echo "==> [3/7] Python packages (main + Wav2Lip)"
pip install -r requirements.txt

echo "==> [4/7] XTTS v2 (Coqui TTS) — descarga ~1.8 GB en primer uso"
pip install TTS

echo "==> [5/7] Wav2Lip weights (wav2lip.pth, ~416 MB)"
mkdir -p Wav2Lip/checkpoints
if [ ! -f "Wav2Lip/checkpoints/wav2lip.pth" ]; then
    wget -q --show-progress \
        "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip.pth" \
        -O "Wav2Lip/checkpoints/wav2lip.pth"
else
    echo "    Ya existe, se omite."
fi

echo "==> [6/7] Ollama + LLM"
if command -v ollama &>/dev/null; then
    ollama pull qwen2.5:3b
else
    echo "    ollama no encontrado. Instálalo desde https://ollama.com y luego: ollama pull qwen2.5:3b"
fi

echo "==> [7/7] Directorios de usuario"
mkdir -p voices diario

echo ""
echo "======================================================"
echo "  ✅  Setup completado"
echo "======================================================"
echo ""
echo "Antes de ejecutar:"
echo "  1. Copia tu foto frontal a:  face.jpeg"
echo "  2. Copia voces de referencia:"
echo "       voices/es.wav   (español, 6-15s)"
echo "       voices/en.wav   (inglés, 6-15s)"
echo "  3. Agrega entradas de diario:  diario/*.md"
echo ""
echo "Para ejecutar:"
echo "    source venv/bin/activate"
echo "    python3 app.py"
echo "======================================================"
