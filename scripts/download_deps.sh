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
# REVERTED: Reverted to using a simple curl command with the correct, direct download link
# for the model file. The -L flag handles redirects.
echo "Downloading compatible model..."
curl -L https://huggingface.co/CompendiumLabs/bge-base-en-v1.5-gguf/resolve/main/bge-base-en-v1.5-q4_k_m.gguf -o model/bert.bin

# MODIFIED: Added a check to ensure the model file was downloaded correctly.
MODEL_SIZE_THRESHOLD=1000000 # 1MB
MODEL_SIZE=$(stat -c%s "model/bert.bin")
if [ "$MODEL_SIZE" -lt "$MODEL_SIZE_THRESHOLD" ]; then
    echo "Error: Downloaded model file is too small ($MODEL_SIZE bytes)."
    echo "This might be an HTML error page instead of the model. Please check the URL."
    exit 1
fi
echo "Model downloaded successfully."


# --- Tokenizer Download ---
echo "Downloading tokenizer for bge-base-en-v1.5..."
curl -L https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/tokenizer.json -o tokenizer/tokenizer.json
curl -L https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/config.json -o tokenizer/config.json
curl -L https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/vocab.txt -o tokenizer/vocab.txt
curl -L https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/tokenizer_config.json -o tokenizer/tokenizer_config.json


echo "Dependencies downloaded to $(pwd)/model and $(pwd)/tokenizer"

# --- External Library Download ---
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"
BERTCPP_REPO="https://github.com/ggerganov/bert.cpp.git"
# MODIFIED: Removed the specific commit hash to use the latest version from the main branch.

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
git checkout master
git pull origin master
# This ensures the correct version of ggml is used by initializing the submodule AFTER checkout
git submodule update --init --recursive
cd "$ROOT_DIR" # Return to the project root

echo "External dependencies downloaded and set up in ${EXTERNAL_DIR}"
