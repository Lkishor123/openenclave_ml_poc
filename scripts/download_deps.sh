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
GGML_REPO="https://github.com/ggerganov/ggml.git"
# MODIFIED: Updated ggml commit to a version compatible with the bert.cpp commit.
GGML_COMMIT="a3c634c436f5631a3d24e548cc6a1202e85e3d7a"
BERTCPP_REPO="https://github.com/ggerganov/bert.cpp.git"
BERTCPP_COMMIT="7f084fa6bdab9939b383495666617ceabe1522db"

mkdir -p "${EXTERNAL_DIR}"

clone_repo() {
    local repo=$1
    local dest=$2
    local commit=$3
    if [ ! -d "$dest/.git" ]; then
        git clone "$repo" "$dest"
    fi
    git -C "$dest" fetch --all
    git -C "$dest" checkout "$commit"
}

# Clone bert.cpp first
clone_repo "$BERTCPP_REPO" "${EXTERNAL_DIR}/bert.cpp" "$BERTCPP_COMMIT"
# Clone ggml as a subdirectory of bert.cpp
clone_repo "$GGML_REPO" "${EXTERNAL_DIR}/bert.cpp/ggml" "$GGML_COMMIT"


echo "Dependencies downloaded to ${EXTERNAL_DIR}"
