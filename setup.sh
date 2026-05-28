#!/usr/bin/env bash
# setup.sh — Copycat project: full bootstrap on Debian 12
# Run once from ~/Escritorio/copycat/
# Requirements: curl, sudo, ollama already installed system-wide

set -euo pipefail

PROJECT_DIR="$HOME/Escritorio/copycat"
mkdir -p "$PROJECT_DIR/voices" "$PROJECT_DIR/diario" "$PROJECT_DIR/Wav2Lip"
cd "$PROJECT_DIR"

echo "==> [1/7] System packages"
sudo apt-get update -qq
sudo apt-get install -y \
    python3-venv python3-pip python3-dev \
    ffmpeg git wget curl \
    portaudio19-dev libasound2-dev libsndfile1-dev \
    build-essential libffi-dev libssl-dev \
    libgl1 libglib2.0-0           # required by opencv headless

echo "==> [2/7] Python venv"
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip wheel setuptools

echo "==> [3/7] Core Python packages"
pip install \
    customtkinter \
    pillow \
    sounddevice \
    soundfile \
    numpy \
    opencv-python-headless \
    openai-whisper \
    ollama \
    langchain \
    langchain-community \
    langchain-chroma \
    langchain-huggingface \
    langchain-text-splitters \
    sentence-transformers \
    chromadb

echo "==> [4/7] XTTS v2 (Coqui TTS)"
# NOTE: downloads ~1.8 GB model on first run of app.py, not here
pip install TTS

echo "==> [5/7] Wav2Lip"
if [ ! -f "Wav2Lip/inference.py" ]; then
    git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git Wav2Lip
fi
pip install -r Wav2Lip/requirements.txt

echo "==> [6/7] Wav2Lip weights"
mkdir -p Wav2Lip/checkpoints
WAV2LIP_PTH="Wav2Lip/checkpoints/wav2lip.pth"
if [ ! -f "$WAV2LIP_PTH" ]; then
    wget -q --show-progress \
        "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip.pth" \
        -O "$WAV2LIP_PTH"
else
    echo "    wav2lip.pth already present, skipped."
fi

echo "==> [7/7] Ollama model"
if command -v ollama &>/dev/null; then
    ollama pull qwen2.5:3b
else
    echo "    [WARN] ollama not found. Install from https://ollama.com and run: ollama pull qwen2.5:3b"
fi

echo ""
echo "======================================================"
echo "Setup complete."
echo ""
echo "Next steps:"
echo "  1. Copy face.jpeg into $PROJECT_DIR/"
echo "  2. Copy voices/es.mp3 and voices/en.mp3 into $PROJECT_DIR/voices/"
echo "  3. Add .md files to $PROJECT_DIR/diario/"
echo "  4. Run:"
echo "       cd $PROJECT_DIR"
echo "       source venv/bin/activate"
echo "       python3 app.py"
echo "======================================================"
