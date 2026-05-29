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
    ffmpeg git wget curl unzip \
    portaudio19-dev libasound2-dev libsndfile1-dev \
    build-essential libffi-dev libssl-dev \
    libgl1 libglib2.0-0

echo "==> [2/7] Python virtualenv"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools uv

echo "==> [2.5/7] Installing CPU-only PyTorch"
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "==> [3/7] Python packages"
uv pip install -r requirements.txt

echo "==> [4/7] Wav2Lip weights (wav2lip_gan.pth, ~416 MB)"
mkdir -p Wav2Lip/checkpoints
if [ ! -f "Wav2Lip/checkpoints/wav2lip_gan.pth" ]; then
    wget -q --show-progress \
        "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth" \
        -O "Wav2Lip/checkpoints/wav2lip_gan.pth"
else
    echo "    Already exists, skipping."
fi

echo "==> [5/7] Ollama + LLM"
if command -v ollama &>/dev/null; then
    ollama pull qwen2.5:3b
else
    echo "    ollama not found. Install from https://ollama.com and run: ollama pull qwen2.5:3b"
fi

echo "==> [6/7] HuggingFace login (needed for Pocket-TTS voice cloning)"
if command -v hf &>/dev/null; then
    if hf auth whoami &>/dev/null; then
        echo "    ✅ Already logged in as $(hf auth whoami 2>&1 | head -1)"
    else
        echo "    ⚠️  Not logged in. Run: hf auth login"
        echo "       (get token from https://huggingface.co/settings/tokens)"
    fi
else
    echo "    ⚠️  HF CLI not found. Install with: curl -LsSf https://hf.co/cli/install.sh | bash"
    echo "       Then: hf auth login"
fi

echo "==> [7/7] User directories"
mkdir -p voices diario

echo ""
echo "======================================================"
echo "  ✅  Setup complete"
echo "======================================================"
echo ""
echo "Before running:"
echo "  1. Copy your front-facing photo to:  face.jpeg"
echo "  2. Copy reference voice files:"
echo "       voices/es.wav   (Spanish, 6-15s)"
echo "       voices/en.wav   (English, 6-15s)"
echo "  3. Add diary entries:  diario/*.md"
echo ""
echo "To run:"
echo "    source venv/bin/activate"
echo "    python3 app.py"
echo "======================================================"
