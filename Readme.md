## Download Dependencies

Run `scripts/download_deps.sh` to fetch the GGML model and tokenizer.

## Build Steps

```bash
openssl genrsa -out enclave/enclave_private.pem -3 3072
oeedger8r --trusted common/enclave.edl --trusted-dir build/edl_generated \
    --search-path /opt/openenclave/include
oeedger8r --untrusted common/enclave.edl --untrusted-dir build/edl_generated \
    --search-path /opt/openenclave/include

mkdir build
cd build
cmake .. -DGGML_ROOT_DIR=/opt/ggml -DCMAKE_BUILD_TYPE=Debug
make
```

### Run Host

```bash
./host/ml_host_prod_go ../model/bert.bin ../enclave/enclave_prod.signed.so --use-stdin
```

### Docker

```bash
docker build -t confidential-ml-app .
docker run --rm -p 8080:8080 --device /dev/sgx_enclave confidential-ml-app
```

For simulation mode, add the `--simulate` flag in `backend/main.go`.

