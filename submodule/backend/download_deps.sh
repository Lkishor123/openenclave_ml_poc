#!/usr/bin/env bash
set -euo pipefail

# This script is not strictly necessary but is a good practice
# to ensure the user has the required tool for the git operations below.
if ! command -v git &> /dev/null
then
    echo "git could not be found. Please install it first."
    exit
fi

mkdir -p model tokenizer

# --- Model Download ---
curl -L https://github.com/Lkishor123/openenclave_ml_poc/releases/download/untagged-7ec4b9a98d1a90bf5157/bge-base-en-v1.5.tgz -o model/bge-base-en-v1.5.tgz
echo "Extracting bge-base-en-v1.5 model..."
tar -xzf model/bge-base-en-v1.5.tgz -C model
cp model/bge-base-en-v1.5/ggml-model-f16.gguf model/bert.bin

# --- Tokenizer Download ---
echo "tokenizer for bge-base-en-v1.5..."
cp model/bge-base-en-v1.5/tokenizer.json tokenizer/tokenizer.json
cp model/bge-base-en-v1.5/config.json tokenizer/config.json
cp model/bge-base-en-v1.5/vocab.txt tokenizer/vocab.txt
cp model/bge-base-en-v1.5/tokenizer_config.json tokenizer/tokenizer_config.json


echo "Dependencies downloaded to $(pwd)/model and $(pwd)/tokenizer"

# --- External Library Download ---
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"
BERTCPP_REPO="https://github.com/Lkishor123/bert.cpp.git"

mkdir -p "${EXTERNAL_DIR}"

# --- Robust Git Checkout and Submodule Initialization ---
echo "Cloning and setting up the LATEST version of bert.cpp and its ggml submodule..."
BERTCPP_DIR="${EXTERNAL_DIR}/bert.cpp"
if [ ! -d "$BERTCPP_DIR/.git" ]; then
    git clone "$BERTCPP_REPO" "$BERTCPP_DIR"
fi
# Change into the directory to run subsequent git commands reliably
cd "$BERTCPP_DIR"
git fetch --all
# MODIFIED: Checkout the main branch to get the latest version and ensure it's up-to-date.
git checkout 7f084fa6bdab9939b383495666617ceabe1522db
# This ensures the correct version of ggml is used by initializing the submodule AFTER checkout
git submodule update --init --recursive
cd "$ROOT_DIR" # Return to the project root

echo "External dependencies downloaded and set up in ${EXTERNAL_DIR}"
