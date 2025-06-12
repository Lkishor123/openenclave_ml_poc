#!/usr/bin/env bash
set -euo pipefail

mkdir -p model tokenizer

# Download GGML DistilBERT model
curl -L https://huggingface.co/mradermacher/distilbert-base-nli-mean-tokens-GGUF/resolve/main/distilbert-base-nli-mean-tokens.Q4_K_M.gguf -o model/bert.bin

# Download tokenizer files
curl -L https://huggingface.co/distilbert/distilbert-base-uncased/resolve/main/tokenizer.json -o tokenizer/tokenizer.json
curl -L https://huggingface.co/distilbert/distilbert-base-uncased/resolve/main/config.json -o tokenizer/config.json

echo "Dependencies downloaded to $(pwd)/model and $(pwd)/tokenizer"
