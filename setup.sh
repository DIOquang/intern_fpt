#!/bin/bash
# ============================================================
# setup.sh — Run this once on Lightning.ai Studio to prepare
# the environment before launching app.py
# ============================================================
set -e

echo "=== [1/5] Installing Python dependencies ==="
# Install PyTorch with CUDA first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q

# Install gradio stack
pip install "gradio==4.44.1" "gradio-client==1.3.0" "starlette<0.38.0" "fastapi<0.113.0" -q

# Patch gradio/oauth.py: HfFolder was removed in huggingface_hub >= 1.0
python - <<'PYEOF'
import pathlib, importlib.util
spec = importlib.util.find_spec('gradio')
if spec:
    oauth_path = pathlib.Path(spec.origin).parent / 'oauth.py'
    if oauth_path.exists():
        src = oauth_path.read_text()
        old = 'from huggingface_hub import HfFolder, whoami'
        new = (
            'try:\n'
            '    from huggingface_hub import HfFolder, whoami\n'
            'except ImportError:\n'
            '    from huggingface_hub import whoami\n'
            '    class HfFolder:\n'
            '        @staticmethod\n'
            '        def get_token(): return None\n'
            '        @staticmethod\n'
            '        def save_token(token): pass\n'
            '        @staticmethod\n'
            '        def delete_token(): pass\n'
        )
        if old in src:
            oauth_path.write_text(src.replace(old, new))
            print('  gradio oauth.py patched OK')
        else:
            print('  oauth.py already patched')
PYEOF

# Install remaining deps (no gradio-image-prompter on Lightning)
pip install facexlib basicsr realesrgan gfpgan -q
pip install numpy opencv-python-headless Pillow scikit-image requests tqdm pyyaml scipy addict future lmdb setuptools ffmpeg-python psutil -q

echo "=== [2/5] Installing basicsr from local source ==="
cd basicsr-1.4.2
pip install -e . -q
cd ..

echo "=== [3/5] Patching Gradio for compatibility ==="
python patch_gradio.py

echo "=== [4/5] Downloading model weights ==="
mkdir -p weights weights/CodeFormer weights/facelib

# Helper: check if file is a real binary (>1MB), not an LFS pointer
is_real_weight() { [ -f "$1" ] && [ "$(wc -c < "$1")" -gt 1048576 ]; }

# RealESRGAN
is_real_weight weights/RealESRGAN_x2.pth || curl -L "https://huggingface.co/sberbank-ai/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth" -o weights/RealESRGAN_x2.pth --progress-bar
is_real_weight weights/RealESRGAN_x4.pth || curl -L "https://huggingface.co/sberbank-ai/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth" -o weights/RealESRGAN_x4.pth --progress-bar
is_real_weight weights/RealESRGAN_x8.pth || curl -L "https://huggingface.co/sberbank-ai/Real-ESRGAN/resolve/main/RealESRGAN_x8.pth" -o weights/RealESRGAN_x8.pth --progress-bar

# CodeFormer weights
is_real_weight weights/codeformer.pth || curl -L "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth" -o weights/codeformer.pth --progress-bar

# facelib weights
is_real_weight weights/facelib/detection_Resnet50_Final.pth || curl -L "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" -o weights/facelib/detection_Resnet50_Final.pth --progress-bar
is_real_weight weights/facelib/parsing_parsenet.pth || curl -L "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth" -o weights/facelib/parsing_parsenet.pth --progress-bar
is_real_weight weights/facelib/detection_mobilenet0.25_Final.pth || curl -L "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth" -o weights/facelib/detection_mobilenet0.25_Final.pth --progress-bar

# Copy facelib weights to facexlib package dir so it can find them at runtime
FACELIB_DIR=$(python -c "import facexlib, os; print(os.path.join(os.path.dirname(facexlib.__file__), 'weights'))")
mkdir -p "$FACELIB_DIR"
cp weights/facelib/*.pth "$FACELIB_DIR/" 2>/dev/null || true

# CodeFormer repo (source code only, no weights download script needed)
if [ ! -d "third_party/CodeFormer" ]; then
    mkdir -p third_party
    git clone --depth 1 https://github.com/sczhou/CodeFormer.git third_party/CodeFormer
    cd third_party/CodeFormer
    pip install -r requirements.txt -q
    # Sync codeformer.pth into repo weights dir (used by inference_codeformer.py)
    mkdir -p weights/CodeFormer weights/facelib
    cp ../../weights/codeformer.pth weights/CodeFormer/codeformer.pth
    cp ../../weights/facelib/*.pth weights/facelib/ 2>/dev/null || true
    cd ../..
elif [ ! -f "third_party/CodeFormer/weights/CodeFormer/codeformer.pth" ]; then
    mkdir -p third_party/CodeFormer/weights/CodeFormer third_party/CodeFormer/weights/facelib
    cp weights/codeformer.pth third_party/CodeFormer/weights/CodeFormer/codeformer.pth
    cp weights/facelib/*.pth third_party/CodeFormer/weights/facelib/ 2>/dev/null || true
fi

echo "=== [5/5] Setup complete! Run: python app.py ==="
