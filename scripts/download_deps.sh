#!/usr/bin/env bash
set -euo pipefail

mkdir -p model tokenizer

# MODIFIED: Switched to a more compatible GGUF model file.
# This model is based on BAAI/bge-base-en-v1.5, which is recommended by the bert.cpp developers.
echo "Downloading compatible model..."
curl -L https://huggingface.co/hotwater/bge-base-en-v1.5-gguf/resolve/main/bge-base-en-v1.5-q4_k_m.gguf -o model/bert.bin

# Download tokenizer files
echo "Downloading tokenizer..."
curl -L https://huggingface.co/distilbert/distilbert-base-uncased/resolve/main/tokenizer.json -o tokenizer/tokenizer.json
curl -L https://huggingface.co/distilbert/distilbert-base-uncased/resolve/main/config.json -o tokenizer/config.json

echo "Dependencies downloaded to $(pwd)/model and $(pwd)/tokenizer"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"
BERTCPP_REPO="https://github.com/ggerganov/bert.cpp.git"
# Using the commit that is known to work with the submodule structure
BERTCPP_COMMIT="b4330a33a82c8abe6b37e8c37223de84a7d30a6c"

mkdir -p "${EXTERNAL_DIR}"

# Clone bert.cpp first
echo "Cloning bert.cpp repository..."
if [ ! -d "${EXTERNAL_DIR}/bert.cpp/.git" ]; then
    git clone "$BERTCPP_REPO" "${EXTERNAL_DIR}/bert.cpp"
fi
git -C "${EXTERNAL_DIR}/bert.cpp" fetch --all
git -C "${EXTERNAL_DIR}/bert.cpp" checkout "$BERTCPP_COMMIT"

# Initialize and update the git submodule for ggml within bert.cpp
# This ensures the correct version of ggml is used, matching what bert.cpp expects.
echo "Initializing and updating ggml submodule..."
git -C "${EXTERNAL_DIR}/bert.cpp" submodule update --init --recursive

echo "External dependencies downloaded and set up in ${EXTERNAL_DIR}"
