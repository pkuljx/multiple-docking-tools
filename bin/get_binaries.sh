#!/usr/bin/env bash
# Download the smina and GNINA static binaries into this bin/ folder.
# They are NOT stored in git (gnina is ~2 GB). Run once after cloning:
#     bash bin/get_binaries.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

SMINA_URL="https://sourceforge.net/projects/smina/files/smina.static/download"
GNINA_URL="https://github.com/gnina/gnina/releases/download/v1.3.3/gnina.cuda12.8.static"

echo "[1/2] smina (static, ~9 MB) ..."
curl -L --fail -o "$HERE/smina" "$SMINA_URL"
chmod +x "$HERE/smina"

echo "[2/2] gnina v1.3.3 (CUDA 12.8 static, ~2 GB) ..."
curl -L --fail -o "$HERE/gnina.bin" "$GNINA_URL"
chmod +x "$HERE/gnina.bin" "$HERE/gnina"

echo
echo "Done."
echo "  smina     -> $HERE/smina"
echo "  gnina.bin -> $HERE/gnina.bin   (call it via the 'gnina' wrapper in this folder)"
echo
echo "GNINA needs an NVIDIA GPU + driver supporting CUDA 12.8, plus libcudnn.so.9"
echo "and the CUDA 12.8 runtime — all provided by the 'dock' conda env; the 'gnina'"
echo "wrapper adds \$CONDA_PREFIX/lib to LD_LIBRARY_PATH automatically."
