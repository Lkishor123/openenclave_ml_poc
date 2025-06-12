#!/usr/bin/env bash
set -euo pipefail

mkdir -p model tokenizer

# Download GGML Distilbert model
curl -L https://huggingface.co/mradermacher/distilbert-base-nli-mean-tokens-GGUF/resolve/main/distilbert-base-nli-mean-tokens.Q4_K_M.gguf -o model/bert.bin

# Download tokenizer files
curl -L https://huggingface.co/distilbert/distilbert-base-uncased/resolve/main/tokenizer.json -o tokenizer/tokenizer.json
curl -L https://huggingface.co/distilbert/distilbert-base-uncased/resolve/main/config.json -o tokenizer/config.json

echo "Dependencies downloaded to $(pwd)/model and $(pwd)/tokenizer"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"
BERTCPP_REPO="https://github.com/ggerganov/bert.cpp.git"
BERTCPP_COMMIT="b4330a33a82c8abe6b37e8c37223de84a7d30a6c"

mkdir -p "${EXTERNAL_DIR}"

# Clone bert.cpp first
if [ ! -d "${EXTERNAL_DIR}/bert.cpp/.git" ]; then
    git clone "$BERTCPP_REPO" "${EXTERNAL_DIR}/bert.cpp"
fi
git -C "${EXTERNAL_DIR}/bert.cpp" fetch --all
git -C "${EXTERNAL_DIR}/bert.cpp" checkout "$BERTCPP_COMMIT"

# MODIFIED: Initialize and update the git submodule for ggml within bert.cpp
# This ensures the correct version of ggml is used.
git -C "${EXTERNAL_DIR}/bert.cpp" submodule update --init --recursive

echo "Dependencies downloaded to ${EXTERNAL_DIR}"
