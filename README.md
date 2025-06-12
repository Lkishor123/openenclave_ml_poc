# Openenclave_ML_POC

## Download Dependencies

Run the helper script to download a prebuilt GGML model and tokenizer files:

```bash
scripts/download_deps.sh
```

After the script finishes you will have `model/bert.bin` and a
`tokenizer/` directory containing the tokenizer configuration.

### Architecture Overview

The project uses the [GGML](https://github.com/ggml-org/ggml) library to run a
BERT model inside an Open Enclave. The host application loads `bert.bin`, passes
tokenized input to the enclave, and prints the resulting logits.

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
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
make run
```

### Running the Host Manually

After building, you can test the GGML path directly:

```bash
./build/host/ml_host_prod_go model/bert.bin enclave/enclave_prod.signed.so --use-stdin
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
