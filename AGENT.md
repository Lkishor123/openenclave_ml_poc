This repository contains a Go backend, a C++ host/enclave using Open Enclave, and build scripts.  
When adding new code or documentation:

- Keep code formatting consistent with the surrounding files.
- Run `scripts/download_deps.sh` before building to ensure external libraries are available.
- The host application links against `bert.cpp` and `ggml`; avoid adding dependencies on extra dependencies.
- Docker builds expect the model at `model/bert.bin` and tokenizer files under `tokenizer/`.
