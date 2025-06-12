#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"
GGML_REPO="https://github.com/ggerganov/ggml.git"
GGML_COMMIT="d58ce750aac398568457db9c69882492c515d488"
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

clone_repo "$GGML_REPO" "${EXTERNAL_DIR}/ggml" "$GGML_COMMIT"
clone_repo "$BERTCPP_REPO" "${EXTERNAL_DIR}/bert.cpp" "$BERTCPP_COMMIT"

echo "Dependencies downloaded to ${EXTERNAL_DIR}" 
