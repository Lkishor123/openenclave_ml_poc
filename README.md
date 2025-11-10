# Openenclave_ML_POC

## Download Dependencies

Run `scripts/download_deps.sh` to fetch a prebuilt GGML model and tokenizer. **Run this script before invoking CMake** so the build can locate `model/bert.bin`:

```bash
scripts/download_deps.sh
```

After the script finishes you will have `model/bert.bin` and a
`tokenizer/` directory containing the tokenizer configuration.

All components—including the Go backend and Docker image—expect the
model to reside at this unified path. The backend and Docker setup now
look for the model exclusively at this path.

### Architecture Overview

The project uses the [GGML](https://github.com/ggml-org/ggml) library to run a
BERT model inside an Open Enclave. The host application loads `bert.bin`, passes
tokenized input to the enclave, and prints the resulting logits.

The script also clones the required GGML and bert.cpp repositories used during
the build.

![openenvlave](https://github.com/user-attachments/assets/d136d379-376f-40ec-86d6-30c421394d34)

## Build

Generate the enclave keys and edge routines:

```bash
openssl genrsa -out enclave/enclave_private.pem -3 3072

# Generate trusted code
oeedger8r --trusted common/enclave.edl --trusted-dir build/edl_generated \
    --search-path /opt/openenclave/include

# Generate untrusted code
oeedger8r --untrusted common/enclave.edl --untrusted-dir build/edl_generated \
    --search-path /opt/openenclave/include
```

Configure and compile:

```bash
cd openenclave_ml_poc
mkdir build
cd build
# Ensure scripts/download_deps.sh has been run so model/bert.bin exists
cmake .. -DGGML_ROOT_DIR=/opt/ggml -DCMAKE_BUILD_TYPE=Debug
make
make run
```

### Run Host

From the `build` directory you can directly invoke the host binary:

```bash
./host/ml_host_prod_go ../model/bert.bin ../enclave/enclave_prod.signed.so --use-stdin
```

## Docker Build

```bash
docker build -t confidential-ml-app .
docker run --rm -p 8080:8080 --device /dev/sgx_enclave confidential-ml-app
```

### For Simulation Mode

Modify `backend/main.go` to pass the `--simulate` flag and then run:

```bash
docker run --rm -p 8080:8080 confidential-ml-app
```

## SupaBase Test:
```
curl -i -X POST \
  'https://xxxxxxxxxx.supabase.co/auth/v1/token?grant_type=password' \
  -H 'apikey: REPLACE_WITH_ANON_KEY' \
  -H 'Authorization: Bearer REPLACE_WITH_ANON_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"email":"user@example.com","password":"SuperSecret123"}'
```