#!/bin/bash
set -euo pipefail

GGML_DIR=/opt/ggml
MODEL_DIR=/app/model
MODEL_ID="distilbert-base-uncased-finetuned-sst-2-english"

# Clone and build GGML
mkdir -p "${GGML_DIR}"

git clone --depth 1 https://github.com/ggerganov/ggml.git /tmp/ggml
cd /tmp/ggml
make
mkdir -p ${GGML_DIR}/lib ${GGML_DIR}/include
cp libggml.a ${GGML_DIR}/lib/
cp -r include/ ${GGML_DIR}/include/

# Convert Hugging Face model to GGML format
cd examples/bert
pip3 install --no-cache-dir transformers
python3 convert-bert.py ${MODEL_ID} /tmp/bert.bin

mkdir -p "${MODEL_DIR}"
mv /tmp/bert.bin "${MODEL_DIR}/bert.bin"

# cleanup
test -d /tmp/ggml && rm -rf /tmp/ggml
