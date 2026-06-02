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
pip install --upgrade pip wheel setuptools uv

echo "==> [2.5/7] Installing CPU-only PyTorch (saves ~3 GB of CUDA libraries)"
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "==> [3/7] Python packages"
uv pip install -r requirements.txt

echo "==> [4/7] Wav2Lip weights (~416 MB)"
mkdir -p src/Wav2Lip/checkpoints
if [ ! -f "src/Wav2Lip/checkpoints/wav2lip.pth" ]; then
    wget -q --show-progress \
        "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip.pth" \
        -O "src/Wav2Lip/checkpoints/wav2lip.pth"
else
    echo "    Already exists, skipping."
fi

echo "==> [5/7] Ollama + LLM"
if command -v ollama &>/dev/null; then
    ollama pull qwen2.5:3b
else
    echo "    ollama not found. Install from https://ollama.com and run: ollama pull qwen2.5:3b"
fi

echo "==> [6/7] User directory structure"
mkdir -p data/picture data/voices data/journal data/behavior data/checkpoints data/vector_db tmp

echo "==> [7/7] Finishing up"
chmod +x run.sh

echo ""
echo "======================================================"
echo "  ✅  Setup complete"
echo "======================================================"
echo ""
echo "Assets to place before running:"
echo "  1. data/picture/face.jpeg   — front-facing portrait (256×256+ px)"
echo "  2. data/voices/[lang].wav   — Where lang means the international code of 2 letters, e,g en.wav (6-15s, clean)"
echo "  4. data/journal/*.md        — diary entries (auto-indexed)"
echo "  5. data/behavior/behavior.txt — personality (optional)"
echo ""
echo "Run:"
echo "    ./run.sh"
echo "======================================================"
