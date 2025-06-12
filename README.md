# Openenclave_ML_POC

## Prepare the DistilBERT model (GGML)

This project uses a GGML formatted version of
`distilbert-base-uncased-finetuned-sst-2-english`.
To create it:

```bash
# Clone ggml and install dependencies
git clone https://github.com/ggerganov/ggml.git
cd ggml/examples/bert
pip install transformers

# Convert the Hugging Face model to GGML
python convert-bert.py distilbert-base-uncased-finetuned-sst-2-english \ 
    ../../distilbert.ggml
```

Copy the resulting `distilbert.ggml` into the `model/` directory of this repo.

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
